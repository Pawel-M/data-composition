import argparse

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config_utils import load_config, load_configs
from data.loading import load_ctxpro_dataset
from evaluating.ctxpro_score import ctxpro_score
from evaluating.translate import translate_dataset
from modeling.nllb_200_config import LANG_MAP

#
# def evaluate_ctxpro(raw_dataset_path,
#                     base_dataset_name,
#                     ds,
#                     split,
#                     phenomenon,
#                     src_lang,
#                     tgt_lang,
#                     results_dir,
#                     results_file_stem):
#     file_name_stem = f'{phenomenon}.{base_dataset_name}.{src_lang}-{tgt_lang}.{split}'
#     json_file_path = os.path.join(raw_dataset_path, 'evalsets', f'{file_name_stem}.json')
#     json_file_path = os.path.expanduser(json_file_path)
#     ctxpro_results = ctxpro_score_script.score(
#         ds['predicted'],
#         json_file_path,
#         pretty=True,
#         complete=True
#     )
#
#     total_correct = sum([_['correct'] for _ in ctxpro_results.values()])
#     total = sum([_['total'] for _ in ctxpro_results.values()])
#     accuracy = total_correct * 100 / total
#
#     results_file_path = os.path.join(results_dir, f'{results_file_stem}.{split}.ctxpro.txt')
#     with open(results_file_path, 'w') as f:
#         f.write(f'{accuracy:.2f}%\t{total_correct}\t{total}\n')
#         for rule, scores in ctxpro_results.items():
#             rule_accuracy = scores['correct'] * 100 / scores['total']
#             f.write(f"{rule} ({scores['form']})\t{rule_accuracy:.2f}%\t{scores['correct']}\t{scores['total']}\n")
#
#     return ctxpro_results


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = torch.compile(model)

    print('tokenizer', tokenizer)
    print('model', model)

    src_lang_code = LANG_MAP[config.src_lang]
    tgt_lang_code = LANG_MAP[config.tgt_lang]
    tokenizer.src_lang = src_lang_code
    tokenizer.tgt_lang = tgt_lang_code
    print(f'Using languages: {config.src_lang} ({src_lang_code}) -> {config.tgt_lang} ({tgt_lang_code})')

    if tokenizer.sep_token is None:
        print('Tokenizer does not have sep_token set!!! The empty separator will be used instead.')

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
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            src_ctx_size=config.src_ctx_size,
            tgt_ctx_size=config.tgt_ctx_size,
            max_length=config.max_length,
            num_beams=config.beam_size,
            batch_size=config.batch_size,
            device=device,
            full_sequence_decoding=config.full_sequence_decoding,
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
        # for split, ds in translations.items():
        #     print(f'Evaluation for {split}')
        #
        #     ctxpro_results = evaluate_ctxpro(
        #         raw_dataset_path=config.raw_dataset_path,
        #         base_dataset_name=config.base_dataset_name,
        #         ds=ds,
        #         split=split,
        #         phenomenon=phenomenon,
        #         src_lang=config.src_lang,
        #         tgt_lang=config.tgt_lang,
        #         results_dir=config.results_dir,
        #         results_file_stem=results_file_stem
        #     )
