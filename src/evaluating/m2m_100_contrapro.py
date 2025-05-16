import argparse
import functools

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config_utils import load_config, load_configs
from data.contrapro import load_contrapro
from common.common_functions import score_contrastive
from evaluating.contrapro_score import score_contrapro
from modeling.nllb_200_config import LANG_MAP


def score(model,
          tokenizer,
          results_dir,
          dataset_dir,
          dataset_context_size=None,
          source_context_size=None,
          target_context_size=None,
          filter_context_size=False,
          limit_size=None,
          results_suffix=None,
          max_len=512,
          batch_size=None):
    use_contrapro_dataset = 'contrapro' in dataset_dir.lower()
    print('use_contrapro_dataset', use_contrapro_dataset)

    data = load_contrapro(dataset_dir, dataset_context_size,
                          filter_context_size=filter_context_size,
                          use_json_lines=use_contrapro_dataset,
                          limit_size=limit_size, )

    source_context_size = 0 if source_context_size is None else source_context_size
    target_context_size = 0 if target_context_size is None else target_context_size

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.compile()

    scorer = functools.partial(score_contrastive, model=model,
                               source_context_size=source_context_size, target_context_size=target_context_size,
                               tokenizer=tokenizer, device=device, max_length=max_len)

    results_file_stem = f'contrapro' if results_suffix is None else f'contrapro.{results_suffix}'
    with torch.no_grad():
        results = score_contrapro(
            data=data,
            score_contrastive_fn=scorer,
            results_dir=results_dir,
            results_file_stem=results_file_stem,
            save_results_to_file=True,
            batch_size=batch_size,
        )

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', default=None, type=str,
                        help='config yaml files, parameters of the later configs overwrite the previous ones')
    args = parser.parse_args()
    config = load_configs(args.configs)
    print('config', config)

    tokenizer_path = config.model_path if config.tokenizer_path is None else config.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path)

    src_lang_code = LANG_MAP[config.src_lang]
    tgt_lang_code = LANG_MAP[config.tgt_lang]
    tokenizer.src_lang = src_lang_code
    tokenizer.tgt_lang = tgt_lang_code
    print(f'Using languages: {config.src_lang} ({src_lang_code}) -> {config.tgt_lang} ({tgt_lang_code})')

    # print('tokenizer', tokenizer)
    print('tokenizer special tokens:', tokenizer.special_tokens_map)
    print('model', model)

    if tokenizer.sep_token is None:
        print('Tokenizer does not have sep_token set!!! The empty separator will be used instead.')

    contrapro_ctx_size = config.get('contrapro_ctx_size', max(config.src_ctx_size, config.tgt_ctx_size))

    results = score(
        model=model,
        tokenizer=tokenizer,
        results_dir=config.results_dir,
        dataset_dir=config.contrapro_dir,
        dataset_context_size=contrapro_ctx_size,
        source_context_size=config.src_ctx_size,
        target_context_size=config.tgt_ctx_size,
        filter_context_size=config.filter_context_size,
        results_suffix=config.results_suffix,
        limit_size=config.limit_dataset_size,
        max_len=config.max_length,
        batch_size=config.batch_size,
    )
