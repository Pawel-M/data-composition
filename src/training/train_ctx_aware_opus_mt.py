import argparse
import shutil
import time

import torch
import torchinfo
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, MarianConfig
from transformers.trainer_utils import set_seed

from config_utils import load_configs
from training.prepare_data import load_train_dataset
from training.utils import get_sacrebleu_metric_fn

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
    tokenizer.add_special_tokens({'sep_token': '</sep>'})

    model_config_kwargs = config.get('model_config_arguments', {})
    model_config = MarianConfig.from_pretrained(model_name, **model_config_kwargs)
    print('Model config:', model_config)

    if random_initialization:
        print('Random initialization of the model weights...')
        model = MarianMTModel(model_config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=model_config)

    model.resize_token_embeddings(len(tokenizer))
    torchinfo.summary(model)

    train_dataset, _ = load_train_dataset(tokenizer, config)

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
        seed=seed,
        **config.training_arguments,
    )

    compute_metric = get_sacrebleu_metric_fn(tokenizer)

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=tokenizer.pad_token_id)
    trainer = Seq2SeqTrainer(model,
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

    # # Move the initial model to the run directory, as the directory is deleted and recreated when the training starts
    # print(f"Saving initial model from 'checkpoint-0' to '{os.path.join(run_name, 'checkpoint-0')}'...")
    # shutil.move('./checkpoint-0', os.path.join(run_name, 'checkpoint-0'))
