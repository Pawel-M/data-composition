import argparse
import os
import time

import datasets
import evaluate
import numpy as np
import torch
import torchinfo
import transformers
from transformers import AutoTokenizer

from config_utils import load_configs
from data.loading import prepare_dataset, load_ctxpro_opensubtitles_dataset
from modeling.freezing_m2m_100 import FreezingM2M100ForConditionalGeneration
from modeling.nllb_200_config import LANG_MAP
from training.m2m_seq2seq_trainer import M2MSeq2SeqTrainer
from training.utils import get_sacrebleu_metric_fn


def filter_phenomena(example, filtered_phenomena, tgt_lang):
    annotations = example['annotation'][tgt_lang]
    phenomena = annotations['phenomena']['phenomenon']
    return any([phenomenon in phenomena for phenomenon in filtered_phenomena])


def load_opensubtitles_datasets(
        lang_pairs,
        src_ctx_size,
        tgt_ctx_size,
        sample_ctx_size,
        raw_data_dir,
        base_data_dir,
        processed_dataset_dir,
        annotation_data_dir,
        tokenizer,
        tokenizer_name,
        filtered_phenomena,
        # dataset_splits: [valid,test]
        # train_size,
        # valid_size,
        # split_seed,
        max_length,
):
    raw_data_dir = os.path.expanduser(raw_data_dir)
    base_data_dir = os.path.expanduser(base_data_dir)
    annotation_data_dir = os.path.expanduser(annotation_data_dir)

    # phenomena_files_config = load_phenomena_files_config(annotation_data_dir)

    DATASET_CTX_SIZE_MAP = {
        0: 1,
        1: 1,
    }

    train_datasets = []
    for lang_pair in lang_pairs:
        # for tgt_lang in langs:
        #     if src_lang == tgt_lang:
        #         continue
        src_lang, tgt_lang = lang_pair.split('-')

        src_lang_code = LANG_MAP[src_lang]
        tgt_lang_code = LANG_MAP[tgt_lang]
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code
        print(f'Using languages: {src_lang} ({src_lang_code}) -> {tgt_lang} ({tgt_lang_code})')

        # src_phenomena_files = phenomena_files_config.get(f'{tgt_lang}-{src_lang}', None)
        # tgt_phenomena_files = phenomena_files_config.get(f'{src_lang}-{tgt_lang}', None)

        if sample_ctx_size:
            assert src_ctx_size == tgt_ctx_size
            context_sizes = [(s, s) for s in range(src_ctx_size + 1)]
        else:
            context_sizes = [(src_ctx_size, tgt_ctx_size)]
        for src_cs, tgt_cs in context_sizes:
            dataset_src_ctx_size = DATASET_CTX_SIZE_MAP[src_cs]
            dataset_tgt_ctx_size = DATASET_CTX_SIZE_MAP[tgt_cs]
            dataset = load_ctxpro_opensubtitles_dataset(
                raw_data_dir=raw_data_dir,
                base_data_dir=base_data_dir,
                processed_dataset_dir=processed_dataset_dir,
                annotation_data_dir=annotation_data_dir,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=dataset_src_ctx_size,
                tgt_ctx_size=dataset_tgt_ctx_size,
                # src_phenomena_file_paths=src_phenomena_files,
                # tgt_phenomena_file_paths=tgt_phenomena_files,
            )

            if filtered_phenomena is not None:
                print(f'Filtering examples with phenomena: {filtered_phenomena}')
                dataset = dataset.filter(
                    filter_phenomena,
                    fn_kwargs={
                        'filtered_phenomena': filtered_phenomena,
                        'tgt_lang': tgt_lang,
                    },
                    keep_in_memory=True,
                )
                print('Filtered dataset:', dataset)

            tokenized_train_dataset = prepare_dataset(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                dataset=dataset['dev'],
                dataset_name='os-ctxpro-train',
                base_data_dir=processed_dataset_dir,
                train_size=None,
                valid_size=None,
                test_size=None,
                split_seed=None,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=src_cs,
                tgt_ctx_size=tgt_cs,
                max_length=max_length,
                save_load_dataset=filtered_phenomena is None,
            )

            train_datasets.append(tokenized_train_dataset)

    if len(train_datasets) > 1:
        train_dataset = datasets.interleave_datasets(train_datasets)
        # eval_dataset = datasets.interleave_datasets(eval_datasets)
        # test_dataset = datasets.interleave_datasets(test_datasets)
    else:
        train_dataset = train_datasets[0]
        # eval_dataset = eval_datasets[0]
        # test_dataset = test_datasets[0]

    print(f'Dataset loaded: {len(train_dataset)} examples')

    return train_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', default=None, type=str,
                        help='config yaml files, parameters of the later configs overwrite the previous ones')
    args = parser.parse_args()
    config = load_configs(args.configs)
    print('config', config)

    # from transformers.models.m2m_100.modeling_m2m_100 import M2M100Model

    model_name = config.model_path
    tokenizer_name = config.tokenizer_path if config.tokenizer_path else model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = FreezingM2M100ForConditionalGeneration.from_pretrained(model_name)
    torchinfo.summary(model)

    if config.freeze:
        model.freeze_model(config)

    train_dataset = load_opensubtitles_datasets(
        lang_pairs=config.lang_pairs,
        src_ctx_size=config.src_ctx_size,
        tgt_ctx_size=config.tgt_ctx_size,
        sample_ctx_size=config.sample_ctx_size,
        raw_data_dir=config.raw_dataset_path,
        base_data_dir=config.base_dataset_path,
        processed_dataset_dir=config.processed_dataset_path,
        annotation_data_dir=config.dataset_annotation_path,
        max_length=config.max_length,
        tokenizer=tokenizer,
        tokenizer_name=config.tokenizer_name,
        filtered_phenomena=config.get('filtered_phenomena'),
    )

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
