import argparse
import functools
import os
import time
import shutil
from functools import reduce

import datasets
import numpy as np
import torch
import torchinfo
import transformers
from transformers import AutoTokenizer, Seq2SeqTrainer
from transformers.trainer_utils import set_seed

from common.common_functions import trim_context, concatenate_current_and_context
from config_utils import load_configs, Struct
from data.loading import load_opensubtitles_dataset, prepare_dataset, load_iwslt2017_dataset
from modeling.weighting_opus_mt import WeightingMarianMTModel, WeightingMarianConfig
from training.prepare_data import load_train_dataset
from training.utils import get_sacrebleu_metric_fn
from training.weighting_m2m_seq2seq_trainer import WeightingM2MSeq2SeqTrainer, WeightingDataCollatorForSeq2Seq


def _np_search_sequence(a, seq, distance=1):
    """
    From https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    :param a:
    :param seq:
    :param distance:
    :return:
    """
    return np.where(reduce(lambda a, b: a & b, (
        (np.concatenate([(a == s)[i * distance:], np.zeros(i * distance, dtype=np.uint8)], dtype=np.uint8)) for i, s in
        enumerate(seq))))[0]


def get_expected_mask(tokenizer, tokenized, expected):
    exp_masks = []
    for i, (tgt, exp_i) in enumerate(zip(tokenized, expected)):
        if exp_i is None or len(exp_i) == 0:
            exp_masks.append([0] * len(tgt))
            continue

        tokenized_expected = tokenizer(text_target=exp_i)['input_ids']
        tokenized_expected_upper = tokenizer(text_target=[e[0].upper() + e[1:] for e in exp_i])['input_ids']
        tokenized_expected_pre = tokenizer(text_target=['"' + e for e in exp_i])['input_ids']
        tokenized_expected_upper_pre = tokenizer(text_target=['"' + e[0].upper() + e[1:] for e in exp_i])['input_ids']
        tgt_array = np.array(tgt)
        exp_mask = np.zeros(tgt_array.size, dtype=np.uint8)
        for exp, expu, exp_p, expu_p in zip(tokenized_expected, tokenized_expected_upper,
                                            tokenized_expected_pre, tokenized_expected_upper_pre):
            exp = exp[:-1]
            expu = expu[:-1]
            exp_p = exp_p[:-1]
            expu_p = expu_p[:-1]
            exp_indices = _np_search_sequence(tgt_array, exp)
            expu_indices = _np_search_sequence(tgt_array, expu)
            if exp_indices.size == 0 and expu_indices.size == 0:
                print(f'Expected "{exp_i}" not found in target: "{tokenizer.decode(tgt)}". Trying preceded...')
                exp_indices = _np_search_sequence(tgt_array, exp_p)
                expu_indices = _np_search_sequence(tgt_array, expu_p)
                if exp_indices.size == 0 and expu_indices.size == 0:
                    print(f'Expected "{exp_p}" not found in target: "{tokenizer.decode(tgt)}"!')
                    continue

            indices = np.concatenate((exp_indices, expu_indices))
            index_argmax = np.argmax(indices)
            lens = [len(exp)] * exp_indices.size + [len(expu)] * expu_indices.size
            max_index = indices[index_argmax]
            max_len = lens[index_argmax]
            np.put_along_axis(exp_mask, np.arange(max_len) + max_index, [1] * max_len, axis=0)
            pass

        exp_masks.append(exp_mask.tolist())
    return exp_masks


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
        target_sentences = []

        for src_ctx in src_ctxs + [src]:
            tokenized_src_ctx = tokenizer(text=src_ctx, return_tensors='pt',
                                          max_length=max_length, truncation=True)
            src_ctx_len = tokenized_src_ctx['input_ids'].shape[1] - num_additional_src_tokens
            source_sentences.extend([index + 1] * src_ctx_len)
            index += 1

        # source_sentences.extend([0] * (len(src_tok) - len(source_sentences)))

        index = 0
        for tgt_ctx in tgt_ctxs + [tgt]:
            tokenized_tgt_ctx = tokenizer(text_target=tgt_ctx, return_tensors='pt',
                                          max_length=max_length, truncation=True)
            tgt_ctx_len = tokenized_tgt_ctx['input_ids'].shape[1] - num_additional_tgt_tokens
            target_sentences.extend([index + 1] * tgt_ctx_len)
            index += 1
        # target_sentences.extend([0] * (len(tgt) - len(target_sentences)))

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
                            filtered_phenomena=None,
                            expected_in_current_only=False,
                            ):

    expected_max_current_location = 0 if expected_in_current_only else tgt_ctx_size
    source = [t[src_lang] for t in examples['translation']]
    target = [t[tgt_lang] for t in examples['translation']]
    source_context = [c[src_lang] for c in examples['context']]
    target_context = [c[tgt_lang] for c in examples['context']]
    expected = []
    for annot in examples['annotation']:
        phenomena = annot[tgt_lang]['phenomena']
        expected_list = []
        for exp, current_loc, p in zip(phenomena['expected'], phenomena['current_location'], phenomena['phenomenon']):
            if filtered_phenomena is not None and p not in filtered_phenomena:
                continue
            if current_loc is not None and current_loc <= expected_max_current_location:
                expected_list.append(exp)
        expected.append(expected_list)
    # expected = [[e for e in a[tgt_lang]['phenomena']['expected']] for a in examples['annotation']]
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

    # tokenized['source_sentences'] = source_sentences
    # tokenized['target_sentences'] = target_sentences
    expected_mask = get_expected_mask(tokenizer, tokenized['labels'], expected)
    tokenized['exp_mask'] = expected_mask
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
    random_initialization = config.get('random_initialization', False)

    from transformers.models.marian.modeling_marian import MarianMTModel

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model_config_kwargs = config.get('model_config_arguments', {})
    model_config = WeightingMarianConfig.from_pretrained(model_name, **model_config_kwargs)
    print('Model config:', model_config)

    if random_initialization:
        print('Random initialization of the model weights...')
        model = WeightingMarianMTModel(model_config)
    else:
        model = WeightingMarianMTModel.from_pretrained(model_name, config=model_config)

    tokenizer.add_special_tokens({'sep_token': '</sep>'})
    model.resize_token_embeddings(len(tokenizer))

    torchinfo.summary(model)

    PHENOMENA_MAP = {
        'gender': 'g',
        'auxiliary': 'x',
        'formality': 'f',
        'inflection': 'i',
        'animacy': 'a',
    }


    dataset_name_suffix = 'annot'
    os_dataset_name_suffix = dataset_name_suffix
    phenomena = None
    if ('dataset_name' in config and config.dataset_name == 'opensubtitles'):
        if 'annotation_phenomena' in config:
            phenomena = config.annotation_phenomena
    elif 'datasets' in config:
        for d in config.datasets:
            dataset_config = Struct(**d)
            if dataset_config.dataset_name == 'opensubtitles':
                phenomena = dataset_config.annotation_phenomena

    if phenomena is not None:
        for p in phenomena:
            os_dataset_name_suffix += f'_{PHENOMENA_MAP[p]}'

    expected_in_current_only = config.get('expected_in_current_only', False)
    if expected_in_current_only:
        print('Expected in current only:', expected_in_current_only)
        dataset_name_suffix += '_current'
        os_dataset_name_suffix += '_current'

    tokenize_fn = functools.partial(
        tokenize_with_sentences,
        filtered_phenomena=phenomena,
        expected_in_current_only=expected_in_current_only,
    )
    train_dataset, _ = load_train_dataset(
        tokenizer,
        config,
        tokenize_fn=tokenize_fn,
        dataset_name_suffix=dataset_name_suffix,
        os_dataset_name_suffix=os_dataset_name_suffix,
    )

    print(f'Number of GPUs available: {torch.cuda.device_count()}')

    # Compatibility with previous configs (setting default weight decay to 0.01)
    if 'training_arguments' not in config:
        config.training_arguments = {}
    if 'weight_decay' not in config.training_arguments:
        config.training_arguments['weight_decay'] = 0.01

    training_args = transformers.Seq2SeqTrainingArguments(
        'checkpoints',
        push_to_hub=False,
        logging_strategy="epoch",
        eval_strategy='no',
        eval_steps=None,
        save_strategy='epoch',
        predict_with_generate=True,
        # weight_decay=0.01,
        **config.training_arguments,
    )

    compute_metric = get_sacrebleu_metric_fn(tokenizer)

    data_collator = WeightingDataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=tokenizer.pad_token_id)
    trainer = WeightingM2MSeq2SeqTrainer(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metric,
    )

    # print(f"Saving initial model to 'checkpoint-0'...")
    # model.save_pretrained('./checkpoint-0')

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

    # # Move the initial model to the run directory, as the directory is deleted and recreated when the training starts
    # print(f"Saving initial model from 'checkpoint-0' to '{os.path.join(run_name, 'checkpoint-0')}'...")
    # shutil.move('./checkpoint-0', os.path.join(run_name, 'checkpoint-0'))
