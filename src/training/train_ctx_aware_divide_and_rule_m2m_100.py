import argparse
import functools
import shutil
import time

import re

import datasets
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
from common.common_functions import trim_context, concatenate_current_and_context, tokenize_all_with_context


def split_sentence_in_middle_word(sentence):
    """
    Splits a sentence roughly in the middle based on word count,
    preserving multiple spaces. It attempts to split after the middle word.
    """
    # Use regex to find words and spaces
    # This pattern captures sequences of non-space chars (words) and sequences of space chars
    parts = re.findall(r'(\S+|\s+)', sentence)

    if not parts:
        return [""]  # Handle empty string

    # Count how many "word" parts there are
    word_count = sum(1 for part in parts if not part.isspace())

    if word_count == 0:
        return [sentence]  # Only spaces or empty

    # Find the index in 'parts' after which we want to split
    # We want to split after roughly word_count // 2 words
    split_part_index = -1
    current_word_count = 0
    for i, part in enumerate(parts):
        if not part.isspace():
            current_word_count += 1
            if current_word_count >= word_count // 2:
                # Find the next space sequence after this word
                for j in range(i + 1, len(parts)):
                    if parts[j].isspace():
                        split_part_index = j
                        break
                # If the last part was a word and no trailing space
                if split_part_index == -1 and i == len(parts) - 1:
                    split_part_index = len(parts) - 1  # Split after the last part
                break

    # If no suitable split point found (e.g., single word sentence)
    if split_part_index == -1:
        return [sentence]

    # Reconstruct the sentence from the parts
    part1 = "".join(parts[:split_part_index + 1])
    part2 = "".join(parts[split_part_index + 1:])

    return [part1, part2]


def tokenize_divide_and_rule(examples,
                             tokenizer,
                             src_lang, tgt_lang,
                             src_ctx_size, tgt_ctx_size,
                             max_length,
                             max_src_ctx_size=None, max_tgt_ctx_size=None,
                             min_split_length=7,  # from the Divide and Rule paper
                             forced_bos_token_id=None):
    source = []
    target = []
    source_context = []
    target_context = []
    for translation, context in zip(examples['translation'], examples['context']):
        src = translation[src_lang]
        tgt = translation[tgt_lang]
        src_ctx = context[src_lang]
        tgt_ctx = context[tgt_lang]

        if len(src.split()) > min_split_length:
            src_parts = split_sentence_in_middle_word(src)
            tgt_parts = split_sentence_in_middle_word(tgt)

            if not (len(src_parts[0]) == 0 or len(src_parts[0]) == 0
                    or len(tgt_parts[0]) == 0 or len(tgt_parts[0]) == 0):
                src = src_parts[1]
                tgt = tgt_parts[1]

                src_ctx.append(src_parts[0])
                tgt_ctx.append(tgt_parts[0])

                if len(src_ctx) > max_src_ctx_size:
                    src_ctx = src_ctx[-max_src_ctx_size:]
                if len(tgt_ctx) > max_tgt_ctx_size:
                    tgt_ctx = tgt_ctx[-max_tgt_ctx_size:]

        source.append(src)
        target.append(tgt)
        source_context.append(src_ctx)
        target_context.append(tgt_ctx)

    tokenized, _, _ = tokenize_all_with_context(
        tokenizer,
        source, target,
        source_context, target_context,
        src_ctx_size, tgt_ctx_size,
        max_length=max_length,
        return_tensors=None,

    )
    if forced_bos_token_id is not None:
        tokenized['forced_bos_token_id'] = [forced_bos_token_id] * len(examples['translation'])

    return tokenized


if __name__ == '__main__':
    # Example sentences for testing
    sentence1 = "This is a sentence with   multiple   spaces in it."
    sentence2 = "Another  sentence  with varying   spacing."
    sentence3 = "Short sentence."
    sentence4 = "OneLongWordWithoutSpaces"
    sentence5 = "   Leading and trailing spaces   "
    print('Testing split_sentence_in_middle_word function...')
    print(f"'{sentence1}' -> {split_sentence_in_middle_word(sentence1)}")
    print(f"'{sentence2}' -> {split_sentence_in_middle_word(sentence2)}")
    print(f"'{sentence3}' -> {split_sentence_in_middle_word(sentence3)}")
    print(f"'{sentence4}' -> {split_sentence_in_middle_word(sentence4)}")
    print(f"'{sentence5}' -> {split_sentence_in_middle_word(sentence5)}")
    print()

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

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    torchinfo.summary(model)

    dataset_name_suffix = 'divided'
    os_dataset_name_suffix = dataset_name_suffix

    tokenize_fn = functools.partial(
        tokenize_divide_and_rule,
        max_src_ctx_size=config.src_ctx_size,
        max_tgt_ctx_size=config.tgt_ctx_size,
    )

    train_dataset, _, train_datasets = load_train_dataset(
        tokenizer,
        config,
        tokenize_fn=tokenize_fn,
        dataset_name_suffix=dataset_name_suffix,
        os_dataset_name_suffix=os_dataset_name_suffix,
        return_partial_datasets=True,
        tokenizer_lang_code_map=LANG_MAP,
    )

    if not config.only_divided_dataset:
        _, _, train_original_datasets = load_train_dataset(
            tokenizer,
            config,
            return_partial_datasets=True,
            tokenizer_lang_code_map=LANG_MAP,
        )
        train_datasets = train_datasets + train_original_datasets
        seed = config.get('seed', None)
        train_dataset = datasets.concatenate_datasets(train_datasets).shuffle(seed)

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

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=tokenizer.pad_token_id)
    trainer = M2MSeq2SeqTrainer(model,
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
