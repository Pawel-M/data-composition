import os

from config_utils import expand_path
from ctxpro import score_script as ctxpro_score_script


def ctxpro_score(translations,
                 results_dir,
                 raw_dataset_path,
                 base_dataset_name,
                 phenomenon,
                 src_lang,
                 tgt_lang,
                 results_file_stem):
    ctxpro_results = {}
    for split, ds in translations.items():
        print(f'Evaluation for {split}')

        # ctxpro_results = _evaluate_ctxpro(
        #     raw_dataset_path=raw_dataset_path,
        #     base_dataset_name=base_dataset_name,
        #     ds=ds,
        #     split=split,
        #     phenomenon=phenomenon,
        #     src_lang=config.src_lang,
        #     tgt_lang=config.tgt_lang,
        #     results_dir=config.results_dir,
        #     results_file_stem=results_file_stem
        # )

        file_name_stem = f'{phenomenon}.{base_dataset_name}.{src_lang}-{tgt_lang}.{split}'
        json_file_path = os.path.join(raw_dataset_path, 'evalsets', f'{file_name_stem}.json')
        json_file_path = expand_path(json_file_path)
        ctxpro_result = ctxpro_score_script.score(
            ds['predicted'],
            json_file_path,
            pretty=True,
            complete=True
        )

        ctxpro_results[split] = ctxpro_result

        total_correct = sum([_['correct'] for _ in ctxpro_result.values()])
        total = sum([_['total'] for _ in ctxpro_result.values()])
        accuracy = total_correct * 100 / total

        results_file_path = os.path.join(results_dir, f'{results_file_stem}.{split}.ctxpro.txt')
        with open(results_file_path, 'w') as f:
            f.write(f'{accuracy:.2f}%\t{total_correct}\t{total}\n')
            for rule, scores in ctxpro_result.items():
                rule_accuracy = scores['correct'] * 100 / scores['total']
                f.write(f"{rule} ({scores['form']})\t{rule_accuracy:.2f}%\t{scores['correct']}\t{scores['total']}\n")

    return ctxpro_results

# def _evaluate_ctxpro(raw_dataset_path,
#                      base_dataset_name,
#                      ds,
#                      split,
#                      phenomenon,
#                      src_lang,
#                      tgt_lang,
#                      results_dir,
#                      results_file_stem):
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
