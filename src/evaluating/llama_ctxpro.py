import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config_utils import load_configs
from data.loading import load_ctxpro_dataset
from evaluating.ctxpro_score import ctxpro_score
from evaluating.translate_llm import translate_dataset
from modeling.llm_utils import generate_full_sequence_prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', default=None, type=str,
                        help='config yaml files, parameters of the later configs overwrite the previous ones')
    args = parser.parse_args()
    config = load_configs(args.configs)
    print('config', config)

    tokenizer_path = config.model_path if config.tokenizer_path is None else config.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="left",
    )

    if 'base_model' in config and config.base_model is not None:
        print(f'Loading perf trained base model {config.base_model}...')
        model_name = config.base_model
        perf = True
    else:
        print(f'Loading model {config.model_path}...')
        model_name = config.model_path
        perf = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=None,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    sep_token = '<sep>'
    eos_token = tokenizer.eos_token

    if not model.config.pad_token_id:
        print('No pad token found in base model, setting it to <pad>.')

        if not tokenizer.pad_token:
            print('No pad token found in tokenizer, setting it to <pad>.')
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    print('tokenizer', tokenizer)
    print('model', model)

    if perf:
        print(f'Loading perf model from {config.model_path}...')
        model = PeftModel.from_pretrained(model, config.model_path)
        model.merge_and_unload()
        print('perf model', model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = torch.compile(model)

    for phenomenon in config.phenomena:
        print(f'Evaluating phenomenon: {phenomenon}')

        dataset = load_ctxpro_dataset(
            raw_data_dir=config.raw_dataset_path,
            base_data_dir=config.dataset_path,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            phenomenon=phenomenon,
            files_base_name=f'{phenomenon}.{config.base_dataset_name}',
            src_ctx_size=config.src_ctx_size,
            tgt_ctx_size=config.tgt_ctx_size,
            splits=config.dataset_splits,
        )

        results_file_stem = f'{config.dataset}.{phenomenon}' if config.results_suffix is None \
            else f'{config.dataset}.{phenomenon}.{config.results_suffix}'

        translations, infos = translate_dataset(
            dataset,
            results_dir=config.results_dir,
            results_file_stem=results_file_stem,
            model=model,
            tokenizer=tokenizer,
            prompt_fn=generate_full_sequence_prompt,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            src_ctx_size=config.src_ctx_size,
            tgt_ctx_size=config.tgt_ctx_size,
            max_prompt_length=config.max_prompt_length,
            max_length=config.max_length,
            num_beams=config.beam_size,
            do_sample=config.do_sample,
            batch_size=config.batch_size,
            separator=sep_token,
            eos=eos_token,
            gold_target_context=config.gold_target_context,
        )
        ctxpro_results = ctxpro_score(
            translations=translations,
            results_dir=config.results_dir,
            raw_dataset_path=config.raw_dataset_path,
            base_dataset_name=config.base_dataset_name,
            phenomenon=phenomenon,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            results_file_stem=results_file_stem
        )
