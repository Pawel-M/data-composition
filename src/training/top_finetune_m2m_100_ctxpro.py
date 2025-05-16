import argparse
import os
import random
import time
from copy import deepcopy

import datasets
import evaluate
import numpy as np
import re
import torch
import torchinfo
import transformers
from datasets import DatasetDict
from transformers import AutoTokenizer

from common.common_functions import score_contextual
from config_utils import load_configs, Struct
from data.loading import prepare_dataset, load_ctxpro_opensubtitles_dataset, load_dataset
from modeling.freezing_m2m_100 import FreezingM2M100ForConditionalGeneration
from modeling.nllb_200_config import LANG_MAP
from training.m2m_seq2seq_trainer import M2MSeq2SeqTrainer
from training.prepare_data import load_train_dataset
from training.utils import get_sacrebleu_metric_fn


def replace_words(text, replacements, p):
    def replace_match(match):
        return random.choice(replacements) if random.random() < p else match.group(0)

    return re.sub(r'\b\w+\b', replace_match, text)


def calculate_pcxmi(
        examples,
        model,
        src_lang,
        tgt_lang,
        src_ctx_size,
        tgt_ctx_size,
        tokenizer,
        max_length,
        device,
):
    sources = [t[src_lang] for t in examples['translation']]
    targets = [t[tgt_lang] for t in examples['translation']]
    sources_context = [t[src_lang] for t in examples['context']]
    targets_context = [t[tgt_lang] for t in examples['context']]

    with torch.no_grad():
        ctx_token_logprob, ctx_tgt_ids, ctx_tgt_attention_mask, ctx_attentions = score_contextual(
            model=model,
            tokenizer=tokenizer,
            device=device,
            source_context_size=src_ctx_size,
            target_context_size=tgt_ctx_size,
            max_length=max_length,
            sources=sources,
            targets=targets,
            source_contexts=sources_context,
            target_contexts=targets_context,
            output_attentions=True,
        )

        sent_token_logprob, sent_tgt_ids, sent_tgt_attention_mask = score_contextual(
            model=model,
            tokenizer=tokenizer,
            device=device,
            source_context_size=0,
            target_context_size=0,
            max_length=max_length,
            sources=sources,
            targets=targets,
            source_contexts=sources_context,
            target_contexts=targets_context,
        )

    ctx_tgt_lens = ctx_tgt_attention_mask.sum(dim=-1)
    sent_tgt_lens = sent_tgt_attention_mask.sum(dim=-1)
    shift_tgt_lens = shift_tgt_attention_mask.sum(dim=-1)

    tgt_lens_diffs = ctx_tgt_lens - sent_tgt_lens
    shift_tgt_lens_diffs = shift_tgt_lens - sent_tgt_lens

    token_pcxmis = []
    pcxmis = []
    max_token_pcxmis = []
    avg_token_pcxmis = []
    for b in range(sent_token_logprob.shape[0]):
        ctx_logprobs = ctx_token_logprob[b, tgt_lens_diffs[b]:tgt_lens_diffs[b] + sent_tgt_lens[b]]
        sent_logprobs = sent_token_logprob[b, :sent_tgt_lens[b]]

        ctx_probs = ctx_logprobs.exp()
        sent_probs = sent_logprobs.exp()

        token_pcxmi = ctx_logprobs - sent_logprobs
        pcxmi = ctx_logprobs.sum() - sent_logprobs.sum()


        token_pcxmis.append(token_pcxmi)
        pcxmis.append(pcxmi)
        max_token_pcxmis.append(token_pcxmi.max().item())
        avg_token_pcxmis.append(token_pcxmi.mean().item())
        pass

    pass

    annotated_current = [False] * len(examples['translation'])

    if 'annotation' in examples:
        annotated_current = [a[tgt_lang]['current_sentence'] for a in examples['annotation']]

    pass
    examples['token_pcxmi'] = token_pcxmis
    examples['pcxmi'] = pcxmis
    examples['max_token_pcxmi'] = max_token_pcxmis
    examples['avg_token_pcxmi'] = avg_token_pcxmis

    examples['annotated_current'] = annotated_current
    examples['target'] = targets
    examples['source'] = sources
    return examples


def calculate_pcxmi_dataset(dataset,
                            model,
                            tokenizer,
                            src_lang,
                            tgt_lang,
                            src_ctx_size,
                            tgt_ctx_size,
                            max_length,
                            batch_size,
                            device,):
    translated_datasets = {}

    for split, ds in dataset.items():
        print(f'Calculating P-CXMI on "{split}" split with {len(ds)} examples...')
        translated_dataset = ds.map(
            calculate_pcxmi,
            batched=True,
            batch_size=batch_size,
            # remove_columns=ds.column_names,
            fn_kwargs={
                'model': model,
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'src_ctx_size': src_ctx_size,
                'tgt_ctx_size': tgt_ctx_size,
                'tokenizer': tokenizer,
                'max_length': max_length,
                'device': device,
            },
            keep_in_memory=True,
        )
        translated_datasets[split] = translated_dataset

    return DatasetDict(translated_datasets)


def prepare_top_dataset(config, tokenizer, model, device=None):


    # -  # OS random
    # dataset_name: opensubtitles
    # raw_dataset_path: $PROJECT_HOME / Datasets / ctxpro / data / opensubs /
    # dataset_path: $PROJECT_HOME / fine - tuning - ctx - mt / data / hf_opensubtitles
    # dataset_annotation_path: $PROJECT_HOME / Datasets / ctxpro / release
    # train_size: 50_000
    # split_seed: 1
    # sample_ctx_size: false
    # lang_pairs:
    # - "en-de"
    # - "de-en"
    # - "en-es"
    # - "es-en"
    # - "en-fr"
    # - "fr-en"
    # - "en-pl"
    # - "pl-en"
    # - "en-ru"
    # - "ru-en"

    def get_valid_filename(name):
        """
        From: https://github.com/django/django/blob/main/django/utils/text.py
        Return the given string converted to a string that can be used for a clean
        filename. Remove leading and trailing spaces; convert other spaces to
        underscores; and remove anything that is not an alphanumeric, dash,
        underscore, or dot.
        """
        s = str(name).strip().replace(" ", "_")
        s = re.sub(r"(?u)[^-\w.]", "", s)
        if s in {"", ".", ".."}:
            raise Exception("Could not derive file name from '%s'" % name)
        return s

    metrics_config = Struct(**config.metrics_arguments)
    metrics_limit_size = metrics_config.limit_size
    model_name = get_valid_filename(config.model_path)

    train_datasets = []
    for dataset_config in config.datasets:
        dataset_config = Struct(**dataset_config)
        if dataset_config.dataset_name != 'opensubtitles':
            raise ValueError(f'Only "opensubtitles" dataset is supported, but got {dataset_config.dataset_name}')

        combined_dataset_file = f'combined_pcxmi_dataset_{dataset_config.train_size}_{metrics_limit_size}_{model_name}'
        combined_dataset_file = os.path.join(config.dataset_path, combined_dataset_file)
        if os.path.exists(combined_dataset_file):
            print(f'Loading tokenized dataset from {combined_dataset_file}...')
            tokenized_dataset = datasets.load_from_disk(combined_dataset_file)
            print(f'Number of training examples: {len(tokenized_dataset)}')
            return tokenized_dataset

        for lang_pair in dataset_config.lang_pairs:
            lang_dataset_config = deepcopy(dataset_config)
            src_lang, tgt_lang = lang_pair.split('-')
            lang_dataset_config.src_lang = src_lang
            lang_dataset_config.tgt_lang = tgt_lang
            lang_dataset_config.src_ctx_size = config.src_ctx_size
            lang_dataset_config.tgt_ctx_size = config.tgt_ctx_size

            src_lang_code = LANG_MAP[src_lang]
            tgt_lang_code = LANG_MAP[tgt_lang]
            tokenizer.src_lang = src_lang_code
            tokenizer.tgt_lang = tgt_lang_code
            print(f'Using languages: {src_lang} ({src_lang_code}) -> {tgt_lang} ({tgt_lang_code})')

            train_dataset = load_dataset(lang_dataset_config.dataset_name, lang_dataset_config)
            # train_dataset = load_train_dataset(tokenizer, config)

            train_dataset = calculate_pcxmi_dataset(
                train_dataset,
                model=model,
                tokenizer=tokenizer,
                src_lang=lang_dataset_config.src_lang,
                tgt_lang=lang_dataset_config.tgt_lang,
                src_ctx_size=config.src_ctx_size,
                tgt_ctx_size=config.tgt_ctx_size,
                max_length=config.max_length,
                batch_size=metrics_config.batch_size,
                device=device,
            )

            train_dataset = train_dataset['train']
            if len(train_dataset) > metrics_limit_size:
                print(f'Filtering dataset to top {metrics_limit_size} (in terms of {metrics_config.metric}) examples')
                train_dataset = train_dataset.sort(metrics_config.metric, reverse=True)
                selected_train_dataset = train_dataset.select(range(metrics_limit_size))
                random_size = metrics_config.get('random_size', 0)
                if random_size > 0:
                    print(
                        f'Adding {random_size} random examples to the dataset (not included in the top {metrics_limit_size})')
                    random_indices = np.random.choice(range(metrics_limit_size, len(train_dataset)), random_size,
                                                      replace=False)
                    random_dataset = train_dataset.select(random_indices)
                    train_dataset = datasets.concatenate_datasets([selected_train_dataset, random_dataset])
                else:
                    train_dataset = selected_train_dataset

            print(f'Dataset loaded: {len(train_dataset)} examples')

            print('Preparing dataset')
            dataset_name = f'{dataset_config.dataset_name}_top'
            prepared_train_dataset = prepare_dataset(
                tokenizer=tokenizer,
                tokenizer_name=config.tokenizer_name,
                dataset=train_dataset,
                dataset_name=dataset_name,
                base_data_dir=dataset_config.dataset_path,
                train_size=None,
                valid_size=None,
                test_size=None,
                split_seed=None,
                src_lang=lang_dataset_config.src_lang,
                tgt_lang=lang_dataset_config.tgt_lang,
                src_ctx_size=config.src_ctx_size,
                tgt_ctx_size=config.tgt_ctx_size,
                max_length=config.max_length,
                save_load_dataset=False,
                tokenize_fn=None,
                tokenize_kwargs=None,
            )
            train_datasets.append(prepared_train_dataset)
    train_datasets = datasets.concatenate_datasets(train_datasets).shuffle(config.seed)

    print(f'Saving tokenized dataset to {combined_dataset_file}...')
    train_datasets.save_to_disk(combined_dataset_file)

    print(f'Number of training examples: {len(train_datasets)}, number of datasets: {len(config.datasets)}')
    return train_datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', default=None, type=str,
                        help='config yaml files, parameters of the later configs overwrite the previous ones')
    args = parser.parse_args()
    config = load_configs(args.configs)
    print('config', config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    # from transformers.models.m2m_100.modeling_m2m_100 import M2M100Model

    model_name = config.model_path
    tokenizer_name = config.tokenizer_path if config.tokenizer_path else model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = FreezingM2M100ForConditionalGeneration.from_pretrained(model_name)
    torchinfo.summary(model)

    if config.freeze:
        model.freeze_model(config)

    # train_dataset, _ = load_train_dataset(tokenizer, config, tokenizer_lang_code_map=LANG_MAP)

    model.to(device)
    train_dataset = prepare_top_dataset(config, tokenizer, model, device)

    # # TODO: refactor loading multiple datasets
    # dataset_config = Struct(**config.datasets[0])
    # if 'src_lang' not in dataset_config:
    #     dataset_config.src_lang = config.src_lang
    # if 'tgt_lang' not in dataset_config:
    #     dataset_config.tgt_lang = config.tgt_lang
    # if 'src_ctx_size' not in dataset_config:
    #     dataset_config.src_ctx_size = config.src_ctx_size
    # if 'tgt_ctx_size' not in dataset_config:
    #     dataset_config.tgt_ctx_size = config.tgt_ctx_size
    #
    # train_dataset = load_dataset(dataset_config.dataset_name, dataset_config)
    # # train_dataset = load_train_dataset(tokenizer, config)
    #
    # metrics_config = Struct(**config.metrics_arguments)
    # train_dataset = calculate_pcxmi_dataset(
    #     train_dataset,
    #     model=model,
    #     tokenizer=tokenizer,
    #     src_lang=config.src_lang,
    #     tgt_lang=config.tgt_lang,
    #     src_ctx_size=config.src_ctx_size,
    #     tgt_ctx_size=config.tgt_ctx_size,
    #     max_length=config.max_length,
    #     batch_size=metrics_config.batch_size,
    #     # device=device,
    # )
    #
    # train_dataset = train_dataset['train']
    # metrics_limit_size = metrics_config.limit_size
    # if len(train_dataset) > metrics_limit_size:
    #     print(f'Filtering dataset to top {metrics_limit_size} (in terms of {metrics_config.metric}) examples')
    #     train_dataset = train_dataset.sort(metrics_config.metric, reverse=True)
    #     selected_train_dataset = train_dataset.select(range(metrics_limit_size))
    #     random_size = metrics_config.get('random_size', 0)
    #     if random_size > 0:
    #         print(
    #             f'Adding {random_size} random examples to the dataset (not included in the top {metrics_limit_size})')
    #         random_indices = np.random.choice(range(metrics_limit_size, len(train_dataset)), random_size,
    #                                           replace=False)
    #         random_dataset = train_dataset.select(random_indices)
    #         train_dataset = datasets.concatenate_datasets([selected_train_dataset, random_dataset])
    #     else:
    #         train_dataset = selected_train_dataset
    #
    # print(f'Dataset loaded: {len(train_dataset)} examples')
    #
    # print('Preparing dataset')
    # dataset_name = f'{dataset_config.dataset_name}_top'
    # train_dataset = prepare_dataset(
    #     tokenizer=tokenizer,
    #     tokenizer_name=tokenizer_name,
    #     dataset=train_dataset,
    #     dataset_name=dataset_name,
    #     base_data_dir=dataset_config.dataset_path,
    #     train_size=None,
    #     valid_size=None,
    #     test_size=None,
    #     split_seed=None,
    #     src_lang=config.src_lang,
    #     tgt_lang=config.tgt_lang,
    #     src_ctx_size=config.src_ctx_size,
    #     tgt_ctx_size=config.tgt_ctx_size,
    #     max_length=config.max_length,
    #     save_load_dataset=False,
    #     tokenize_fn=None,
    #     tokenize_kwargs=None,
    # )

    print(f'Number of GPUs available: {torch.cuda.device_count()}')

    training_args = transformers.Seq2SeqTrainingArguments(
        'checkpoints',
        push_to_hub=False,
        logging_strategy="epoch",
        eval_strategy='no',
        eval_steps=None,
        save_strategy='epoch',
        predict_with_generate=True,
        **config.training_arguments,
    )

    compute_metric = get_sacrebleu_metric_fn(tokenizer)

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=tokenizer.pad_token_id)
    trainer = M2MSeq2SeqTrainer(model,
                                training_args,
                                data_collator=data_collator,
                                train_dataset=train_dataset,
                                # eval_dataset=eval_dataset,
                                tokenizer=tokenizer,
                                compute_metrics=compute_metric, )
    # trainer = PostBackwardM2MSeq2SeqTrainer(model,
    #                                         training_args,
    #                                         data_collator=data_collator,
    #                                         train_dataset=train_dataset,
    #                                         # eval_dataset=eval_dataset,
    #                                         tokenizer=tokenizer,
    #                                         compute_metrics=compute_metric, )

    print('Training started...')
    start_time = time.time()

    trainer.train(resume_from_checkpoint=False)

    training_time = time.time() - start_time
    hours = training_time // 3600
    minutes = (training_time % 3600) // 60
    print(f'Training completed in {hours}:{minutes} hours.')

    torchinfo.summary(model)

    model.save_pretrained('./model')
    tokenizer.save_pretrained('./tokenizer')

    # Remove the checkpoint directory
    run_name = training_args.output_dir
    print(f"Removing checkpoint directory '{run_name}'...")
    import shutil

    shutil.rmtree(run_name)
