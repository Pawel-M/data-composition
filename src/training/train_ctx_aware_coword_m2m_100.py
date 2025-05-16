import argparse
import functools
import shutil
import time

import torch
import torchinfo
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.trainer_utils import set_seed

from config_utils import load_configs
from modeling.coword_m2m_100 import CoWordM2M100Config, CoWordM2M100ForConditionalGeneration
from modeling.nllb_200_config import LANG_MAP
from training.m2m_seq2seq_trainer import M2MSeq2SeqTrainer
from training.prepare_data import load_train_dataset
from training.utils import get_sacrebleu_metric_fn
from training.weighting_m2m_seq2seq_trainer import WeightingDataCollatorForSeq2Seq, WeightingM2MSeq2SeqTrainer
from common.common_functions import trim_context, concatenate_current_and_context

def tokenize_all_with_context_and_sentence(tokenizer,
                                           sources, targets,
                                           sources_context, targets_context,
                                           source_context_size, target_context_size,
                                           max_length=None,
                                           return_tensors=None,
                                           num_additional_src_tokens=0,
                                           num_additional_tgt_tokens=0, ):
    src_context, sources_context_sizes = trim_context(sources_context, source_context_size)
    tgt_context, targets_context_sizes = trim_context(targets_context, target_context_size)

    sep_token = tokenizer.sep_token
    all_sources = concatenate_current_and_context(sources, src_context, sep_token)
    all_targets = concatenate_current_and_context(targets, tgt_context, sep_token)
    all_tokenized = tokenizer(text=all_sources,
                              text_target=all_targets,
                              max_length=max_length,
                              padding=False,
                              truncation=True,
                              # return_tensors='pt',
                              return_tensors=return_tensors,
                              return_token_type_ids=False, )

    all_source_sentences = []
    all_target_sentences = []
    for src, tgt, src_tok, tgt_tok, src_ctxs, tgt_ctxs in zip(sources, targets, all_tokenized['input_ids'],
                                                              all_tokenized['labels'], src_context, tgt_context):
        index = 0
        source_sentences = []
        for src_ctx in src_ctxs + [src]:
            tokenized_src_ctx = tokenizer(text=src_ctx, return_tensors='pt',
                                          max_length=max_length, truncation=True)

            src_ctx_len = tokenized_src_ctx['input_ids'].shape[1] - num_additional_src_tokens
            if num_additional_src_tokens > 0 and index == 0:
                src_ctx_len += 1
            source_sentences.extend([index + 1] * src_ctx_len)
            index += 1

        index = 0
        target_sentences = []
        for tgt_ctx in tgt_ctxs + [tgt]:
            tokenized_tgt_ctx = tokenizer(text_target=tgt_ctx, return_tensors='pt',
                                          max_length=max_length, truncation=True)
            tgt_ctx_len = tokenized_tgt_ctx['input_ids'].shape[1] - num_additional_tgt_tokens
            if num_additional_src_tokens > 0 and index == 0:
                tgt_ctx_len += 1
            target_sentences.extend([index + 1] * tgt_ctx_len)
            index += 1

        all_source_sentences.append(source_sentences)
        all_target_sentences.append(target_sentences)
    return all_tokenized, sources_context_sizes, targets_context_sizes, all_source_sentences, all_target_sentences


def tokenize_with_sentences(examples,
                            tokenizer,
                            src_lang, tgt_lang,
                            src_ctx_size, tgt_ctx_size,
                            max_length,
                            forced_bos_token_id=None,
                            num_additional_src_tokens=1,
                            num_additional_tgt_tokens=1,
                            ):
    source = [t[src_lang] for t in examples['translation']]
    target = [t[tgt_lang] for t in examples['translation']]
    source_context = [c[src_lang] for c in examples['context']]
    target_context = [c[tgt_lang] for c in examples['context']]
    tokenized, _, _, source_sentences, target_sentences = tokenize_all_with_context_and_sentence(
        tokenizer,
        source, target,
        source_context, target_context,
        src_ctx_size, tgt_ctx_size,
        max_length=max_length,
        return_tensors=None,
        num_additional_src_tokens=num_additional_src_tokens,
        num_additional_tgt_tokens=num_additional_tgt_tokens,
    )

    tokenized['source_sentences'] = source_sentences
    tokenized['target_sentences'] = target_sentences
    if forced_bos_token_id is not None:
        tokenized['forced_bos_token_id'] = [forced_bos_token_id] * len(examples['translation'])

    return tokenized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', default=None, type=str,
                        help='config yaml files, parameters of the later configs overwrite the previous ones')
    parser.add_argument("--resume-from-checkpoint", default=False, action='store_true',
                        help='Resume training from the last checkpoint')
    args = parser.parse_args()
    config = load_configs(args.configs)
    print('config', config)

    seed = config.get('seed', 1)
    set_seed(seed)
    print('seed', seed)

    model_name = config.model_path
    tokenizer_name = config.tokenizer_path if config.tokenizer_path else model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'sep_token': '</sep>'})

    model_config_kwargs = config.get('model_config_arguments', {})
    model_config = CoWordM2M100Config.from_pretrained(
        model_name,
        mask_token_id=tokenizer.mask_token_id,
        **model_config_kwargs)
    print('Model config:', model_config)

    model = CoWordM2M100ForConditionalGeneration.from_pretrained(model_name, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    torchinfo.summary(model)

    dataset_name_suffix = 'annot_sentence'
    os_dataset_name_suffix = dataset_name_suffix
    tokenize_fn = functools.partial(
        tokenize_with_sentences,
    )
    train_dataset, _ = load_train_dataset(
        tokenizer,
        config,
        tokenize_fn=tokenize_fn,
        dataset_name_suffix=dataset_name_suffix,
        os_dataset_name_suffix=os_dataset_name_suffix,
        tokenizer_lang_code_map=LANG_MAP,
    )

    # train_dataset, _ = load_train_dataset(tokenizer, config, tokenizer_lang_code_map=LANG_MAP)

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

    # data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=tokenizer.pad_token_id)
    # trainer = M2MSeq2SeqTrainer(model,
    #                             training_args,
    #                             data_collator=data_collator,
    #                             train_dataset=train_dataset,
    #                             # eval_dataset=eval_dataset,
    #                             tokenizer=tokenizer,
    #                             compute_metrics=compute_metric, )

    data_collator = WeightingDataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=tokenizer.pad_token_id)
    trainer = WeightingM2MSeq2SeqTrainer(model,
                                         training_args,
                                         data_collator=data_collator,
                                         train_dataset=train_dataset,
                                         # eval_dataset=eval_dataset,
                                         tokenizer=tokenizer,
                                         compute_metrics=compute_metric, )

    print('Training started...')
    start_time = time.time()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

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
    shutil.rmtree(run_name)
