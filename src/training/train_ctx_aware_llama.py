import argparse
import functools
import shutil
import time

import torch
import torchinfo
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, MarianConfig, AutoModelForCausalLM, \
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizerFast
from transformers.trainer_utils import set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from common.common_functions import trim_context
from config_utils import load_configs
from data.loading import load_dataset
from modeling.completion_loss_llama import CompletionLossLlamaForCausalLM
from modeling.llm_utils import generate_full_sequence_prompt
from training.completion_loss_trainer import CompletionLossTrainer
from training.prepare_data import load_train_dataset
from training.utils import get_sacrebleu_metric_fn


def tokenize_for_llm(
        examples,
        tokenizer,
        src_lang, tgt_lang,
        src_ctx_size, tgt_ctx_size,
        max_length,
        prompt_fn,
        forced_bos_token_id=None,
        separator='<sep>',
        eos='</s>',
):
    sources = [t[src_lang] for t in examples['translation']]
    targets = [t[tgt_lang] for t in examples['translation']]
    source_contexts = [c[src_lang] for c in examples['context']]
    target_contexts = [c[tgt_lang] for c in examples['context']]
    source_contexts, effective_src_context_size = trim_context(source_contexts, src_ctx_size)
    target_contexts, effective_tgt_context_size = trim_context(target_contexts, tgt_ctx_size)
    examples, prompts = prompt_fn(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        sources=sources,
        source_contexts=source_contexts,
        targets=targets,
        target_contexts=target_contexts,
        sep=separator,
        eos=eos,
        include_pure_prompt=True,
    )

    tokenized = tokenizer(examples, max_length=max_length, truncation=True)
    prompts_tokenized = tokenizer(prompts, max_length=max_length, truncation=True)
    prompt_masks = [[1] * len(p) + [0] * (len(t) - len(p))
                    for t, p in zip(tokenized['input_ids'], prompts_tokenized['input_ids'])]
    tokenized['token_type_ids'] = prompt_masks
    return tokenized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', default=None, type=str,
                        help='config yaml files, parameters of the later configs overwrite the previous ones')
    parser.add_argument("--resume-from-checkpoint", default=False, action='store_true',
                        help='Resume training from the last checkpoint')
    parser.add_argument("--keep-checkpoints", default=False, action='store_true',
                        help='Do not remove checkpoints after training')
    args = parser.parse_args()
    config = load_configs(args.configs)
    print('config', config)

    seed = config.get('seed', 1)
    set_seed(seed)
    print('seed', seed)

    # bitsandbytes config
    USE_NESTED_QUANT = True
    BNB_4BIT_COMPUTE_DTYPE = "bfloat16"

    assert 'quantization' not in config or config.quantization in ('4bits', '8bits', '16bits')

    if ('use_4bits' in config and config.use_4bits) or config.quantization == '4bits':
        print('Using 4-bit quantization with nested quantization.')
        # 4-bit quantization
        compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=USE_NESTED_QUANT,
        )
        torch_dtype = None

    elif ('use_4bits' in config and not config.use_4bits) or config.quantization == '8bits':
        print('Using 8-bit quantization with nested quantization.')
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = None

    elif config.quantization == '16bits':
        print('Using 16-bit quantization with nested quantization.')
        quantization_config = None
        torch_dtype = torch.bfloat16

    else:
        raise ValueError(f"Unsupported quantization type: {config.quantization}")

    tokenizer_path = config.model_path if config.tokenizer_path is None else config.tokenizer_path
    tokenizer = LlamaTokenizerFast.from_pretrained(
        tokenizer_path,
        padding_side="left",
    )

    model = CompletionLossLlamaForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch_dtype,
        device_map='auto',
        quantization_config=quantization_config,
    )

    sep_token = '<sep>'
    eos_token = tokenizer.eos_token

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # tokenizer.pad_token = "[PAD]"
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.model_input_names.append('token_type_ids')

    print('tokenizer', tokenizer)
    print('model', model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        **config.lora_arguments,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print('perf model', model)

    sep_token = '<sep>'
    eos_token = tokenizer.eos_token

    train_dataset, _ = load_train_dataset(
        tokenizer,
        config,
        tokenize_fn=tokenize_for_llm,
        tokenize_kwargs={
            'prompt_fn': generate_full_sequence_prompt,
            'separator': sep_token,
            'eos': eos_token,
        },
    )

    print(f'Number of GPUs available: {torch.cuda.device_count()}')

    if 'training_arguments' not in config:
        config.training_arguments = {}

    training_args = transformers.TrainingArguments(
        'checkpoints',
        push_to_hub=False,
        logging_strategy="epoch",
        eval_strategy='no',
        eval_steps=None,
        save_strategy='epoch',
        seed=seed,
        **config.training_arguments,
    )

    compute_metric = get_sacrebleu_metric_fn(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Save the tokenizer (can be used to evaluate checkpoints)
    tokenizer.save_pretrained('./tokenizer')

    trainer = CompletionLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print('Training started...')
    start_time = time.time()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    training_time = time.time() - start_time
    hours = training_time // 3600
    minutes = (training_time % 3600) // 60
    print(f'Training completed in {hours}:{minutes} hours.')

    torchinfo.summary(model)

    model.save_pretrained('./model')

    # Remove the checkpoint directory
    run_name = training_args.output_dir
    if args.keep_checkpoints:
        print(f"Keeping checkpoints directory '{run_name}'.")
    else:
        print(f"Removing checkpoint directory '{run_name}'...")
        shutil.rmtree(run_name)
