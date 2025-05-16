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
from transformers.trainer_utils import set_seed

from config_utils import load_configs
from data.loading import prepare_dataset, load_ctxpro_opensubtitles_dataset
from modeling.freezing_opus_mt import FreezingMarianMTModel
from modeling.nllb_200_config import LANG_MAP
from training.m2m_seq2seq_trainer import M2MSeq2SeqTrainer
from training.utils import get_sacrebleu_metric_fn
from training.prepare_data import load_train_dataset


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

    if config.freeze:
        model.freeze_model(config)

    train_dataset, _ = load_train_dataset(tokenizer, config)

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
