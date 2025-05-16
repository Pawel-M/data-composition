import argparse
import os
import time

import datasets
import evaluate
import numpy as np
import torch
import torchinfo
import transformers
from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed

from config_utils import load_configs, Struct
from data.loading import prepare_dataset, load_ctxpro_opensubtitles_dataset, load_dataset
from modeling.freezing_opus_mt import FreezingMarianMTModel
from modeling.nllb_200_config import LANG_MAP
from training.m2m_seq2seq_trainer import M2MSeq2SeqTrainer
from training.utils import get_sacrebleu_metric_fn
from training.prepare_data import load_train_dataset
from evaluating.opus_mt_pcxmi import calculate_pcxmi_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', default=None, type=str,
                        help='config yaml files, parameters of the later configs overwrite the previous ones')
    args = parser.parse_args()
    config = load_configs(args.configs)
    print('config', config)

    seed = config.get('seed', 1)
    set_seed(seed)
    print('seed', seed)

    # from transformers.models.m2m_100.modeling_m2m_100 import M2M100Model

    model_name = config.model_path
    tokenizer_name = config.tokenizer_path if config.tokenizer_path else model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = FreezingMarianMTModel.from_pretrained(model_name)
    torchinfo.summary(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if config.freeze:
        model.freeze_model(config)

    # TODO: refactor loading multiple datasets
    dataset_config = Struct(**config.datasets[0])
    if 'src_lang' not in dataset_config:
        dataset_config.src_lang = config.src_lang
    if 'tgt_lang' not in dataset_config:
        dataset_config.tgt_lang = config.tgt_lang
    if 'src_ctx_size' not in dataset_config:
        dataset_config.src_ctx_size = config.src_ctx_size
    if 'tgt_ctx_size' not in dataset_config:
        dataset_config.tgt_ctx_size = config.tgt_ctx_size

    train_dataset = load_dataset(dataset_config.dataset_name, dataset_config)
    # train_dataset = load_train_dataset(tokenizer, config)

    metrics_config = Struct(**config.metrics_arguments)
    train_dataset = calculate_pcxmi_dataset(
        train_dataset,
        results_dir=config.results_dir,
        results_file_stem='opus_mt_pcxmi',
        model=model,
        tokenizer=tokenizer,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        src_ctx_size=config.src_ctx_size,
        tgt_ctx_size=config.tgt_ctx_size,
        max_length=config.max_length,
        batch_size=metrics_config.batch_size,
        device=device,
    )

    train_dataset = train_dataset['train']
    metrics_limit_size = metrics_config.limit_size
    if len(train_dataset) > metrics_limit_size:
        print(f'Filtering dataset to top {metrics_limit_size} (in terms of {metrics_config.metric}) examples')
        train_dataset = train_dataset.sort(metrics_config.metric, reverse=True)
        selected_train_dataset = train_dataset.select(range(metrics_limit_size))
        random_size = metrics_config.get('random_size', 0)
        if random_size > 0:
            print(f'Adding {random_size} random examples to the dataset (not included in the top {metrics_limit_size})')
            random_indices = np.random.choice(range(metrics_limit_size, len(train_dataset)), random_size, replace=False)
            random_dataset = train_dataset.select(random_indices)
            train_dataset = datasets.concatenate_datasets([selected_train_dataset, random_dataset])
        else:
            train_dataset = selected_train_dataset

    print(f'Dataset loaded: {len(train_dataset)} examples')

    print('Preparing dataset')
    dataset_name = f'{dataset_config.dataset_name}_top'
    train_dataset = prepare_dataset(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        dataset=train_dataset,
        dataset_name=dataset_name,
        base_data_dir=dataset_config.dataset_path,
        train_size=None,
        valid_size=None,
        test_size=None,
        split_seed=None,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        src_ctx_size=config.src_ctx_size,
        tgt_ctx_size=config.tgt_ctx_size,
        max_length=config.max_length,
        save_load_dataset=False,
        tokenize_fn=None,
        tokenize_kwargs=None,
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
        seed=seed,
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
