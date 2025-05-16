import os
import warnings
from typing import List

import math
import torch
import tqdm

from data.contrapro import ContrastiveDataPoint


def select_sublist(base_list, include):
    return [base_list[i] for i, inc in enumerate(include) if inc]


def compute_metrics(metric, predictions, references):
    result = metric.compute(predictions=predictions, references=references)
    result = {"bleu": result["score"]}
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def save_results(correct_total,
                 total,
                 # bleu,
                 results_dir,
                 results_file):
    results_file = os.path.join(results_dir, results_file)
    print(f'Saving results to file: {results_file}...')
    with open(results_file, 'w') as f:
        f.write(f'correct {correct_total}\n')
        f.write(f'total {total}\n')
        f.write(f'accuracy {correct_total / total}\n')
        print(f'Accuracy {correct_total / total}')

        # if bleu is not None:
        #     f.write(f'BLEU {round(bleu, 4)} ({bleu})\n')
        #     print(f'BLEU {round(bleu, 4)}')


# def save_attentions_to_file(tokens, correct, total, pronoun_target_logits,
#                             pronoun_self_attentions, pronoun_cross_attentions, pronoun_decoder_attentions,
#                             data, correct_total, bleu,
#                             results_dir, results_suffix=None, part_index=None):
#     results = {
#         'tokens': tokens,
#         'correct': [torch.tensor(c) for c in correct],
#         'total': total,
#         # 'target_encoded_ids': all_target_encoded_ids,
#         'phrase_target_logits': pronoun_target_logits,
#         'phrase_attentions': {
#             'self_attentions': pronoun_self_attentions,
#             'cross_attentions': pronoun_cross_attentions,
#             'decoder_attentions': pronoun_decoder_attentions,
#         },
#         'data': data,
#         'accuracy': correct_total / total,
#         'bleu': bleu,
#     }
#
#     part_str = f'.part-{part_index}' if part_index is not None else ''
#     if results_suffix is not None:
#         results_file = f'results_{results_suffix}{part_str}.pkl'
#     else:
#         results_file = f'results{part_str}.pkl'
#
#     with open(os.path.join(results_dir, results_file), 'wb') as f:
#         pickle.dump(results, f)


def save_predictions(correct,
                     selected,
                     total_logprobs,
                     selected_logprobs,
                     data: List[ContrastiveDataPoint],
                     # predictions,
                     # context_size,
                     results_dir,
                     results_file, ):
    results_file = os.path.join(results_dir, results_file)  # 'results.txt'
    print(f'Saving results to file: {results_file}...')
    with open(results_file, 'w') as f:
        # translations_generated = predictions is not None

        # if predictions is None:
        #     predictions = [None for _ in data]

        print('Writing wrong predictions...')

        f.write('Wrong:\n')
        f.write('S - source\n')
        # if context_size is not None and context_size > 0:
        f.write('SC - source context\n')
        f.write('TC - target context\n')
        f.write('R - reference\n')
        f.write('P - predicted\n')
        # if translations_generated:
        #     f.write('G - generated\n')
        f.write('\n\n')

        i = -1
        d: ContrastiveDataPoint
        for d, c, sel, tp, sp in zip(data, correct, selected, total_logprobs, selected_logprobs):
            s = d.source
            ts = d.targets
            sc = d.source_context
            tc = d.target_context

            i += 1
            if c:
                continue

            # generated = generate_fn(src=s, src_context=sc, tgt_context=tc)
            f.write(f'\tId: {i}\n')
            f.write(f'\tS: {s}\n')

            # if context_size is not None and context_size > 0:
            f.write(f'\tSC: {sc}\n')
            f.write(f'\tTC: {tc}\n')

            f.write(f'\tR: {ts[0]} ({tp[0]})\n')
            f.write(f'\tP: {ts[sel]} ({sp}, {tp})\n')
            # if translations_generated:
            #     f.write(f'\tG: {generated}\n')
            f.write('\n\n')

        print('Writing correct predictions...')

        f.write('\n\n\nCorrect:\n')
        f.write('S - source\n')
        # if context_size is not None and context_size > 0:
        f.write('SC - context\n')
        f.write('P - predicted\n')
        # if translations_generated:
        #     f.write('G - generated\n')
        f.write('\n\n')

        i = -1
        for d, c, sel, tp, sp in zip(data, correct, selected, total_logprobs, selected_logprobs):
            s = d.source
            ts = d.targets
            sc = d.source_context
            tc = d.target_context

            i += 1
            if not c:
                continue

            # generated = generate_fn(src=s, src_context=sc, tgt_context=tc)
            f.write(f'\tId: {i}\n')
            f.write(f'\tS: {s}\n')

            # if context_size is not None and context_size > 0:
            f.write(f'\tSC: {sc}\n')
            f.write(f'\tTC: {tc}\n')

            f.write(f'\tP: {ts[sel]} ({sp}, {tp})\n')
            # if translations_generated:
            #     f.write(f'\tG: {generated}\n')
            f.write('\n\n')


def save_results_correct(correct, results_dir, results_file, ):
    if results_dir is not None:
        results_file = os.path.join(results_dir, results_file)  # 'results.txt'

    print(f'Saving correct to file: {results_file}...')
    with open(results_file, 'w') as f:
        for i, cor in enumerate(correct):
            f.write(f'{1 if cor else 0}\n')


def save_results_logprobs(logprobs, results_dir, results_file, flat=False):
    if results_dir is not None:
        results_file = os.path.join(results_dir, results_file)  # 'results.txt'

    print(f'Saving correct to file: {results_file}...')
    with open(results_file, 'w') as f:
        for i, lps in enumerate(logprobs):
            if flat:
                for lp in lps:
                    f.write(f'{lp}\n')
            else:
                f.write(f'{", ".join([str(lp) for lp in lps])}\n')


def get_results_dir(base_dir, model_name, results_dir_prefix=None, context_size=0, create=True,
                    use_name_slice_index=-1):
    results_dir_name = model_name.split("/")[use_name_slice_index]
    if results_dir_prefix is not None:
        results_dir_name = results_dir_prefix + results_dir_name

    results_dir = f'{base_dir}/{results_dir_name}'
    if context_size is not None and context_size > 0:
        results_dir += f'_ctx-{context_size}'

    if create:
        os.makedirs(results_dir, exist_ok=True)

    return results_dir


def score(data: List[ContrastiveDataPoint],
          score_contrastive_fn,
          ):
    (
        total_logprobs,
        tokens,
        additional_data,
    ) = score_contrastive_fn(data=data)

    num_nans = 0
    correct = []
    selected = []
    ref_logprobs = []
    selected_logprobs = []
    d: ContrastiveDataPoint
    for i, (d, total_logprob) in enumerate(zip(data, total_logprobs)):
        best_logprob = torch.argmax(torch.tensor(total_logprob), dim=0)
        any_nan = torch.tensor(total_logprob).isnan().any()

        is_correct = best_logprob == 0 and not any_nan
        if any_nan:
            warnings.warn(f'Nan logprobs for {i} {d.source} {d.targets[0]} {torch.tensor(total_logprobs)}')
            num_nans += 1

        selected.append(int(best_logprob if not any_nan else -1))
        correct.append(bool(is_correct))
        ref_logprobs.append(total_logprob[0].item())
        selected_logprobs.append(total_logprob[best_logprob].item())

    if num_nans > 0:
        warnings.warn(f'NaNs detected! Number of examples containing NaNs: {num_nans}')

    total_logprobs = [[t.item() for t in tlp] for tlp in total_logprobs]
    return (
        correct,
        selected,
        total_logprobs,
        ref_logprobs,
        selected_logprobs,
        tokens,
    )


def score_contrapro(score_contrastive_fn,
                    results_dir,
                    data: List[ContrastiveDataPoint],
                    results_file_stem=None,
                    save_results_to_file=True,
                    batch_size=None):
    os.makedirs(results_dir, exist_ok=True)
    num_batches = math.ceil(len(data) / batch_size) if batch_size is not None else 1
    if num_batches > 1:
        print(f'Processing {len(data)} examples in {num_batches} batches...')

    all_correct_total = 0
    all_total = 0
    all_correct = []
    all_selected = []
    all_total_logprobs = []
    all_selected_logprobs = []

    pbar = tqdm.tqdm(total=len(data), dynamic_ncols=True, ncols=200)
    for batch_id in range(num_batches):
        if num_batches > 1:
            start = batch_id * batch_size
            end = min((batch_id + 1) * batch_size, len(data))
            data_batch = data[start:end]
        else:
            data_batch = data

        (
            correct,
            selected,
            total_logprobs,
            ref_logprobs,
            selected_logprobs,
            tokens,
        ) = score(data_batch, score_contrastive_fn, )

        batch_correct = sum(correct)
        batch_total = len(data_batch)
        all_correct_total += batch_correct
        all_total += batch_total
        all_correct.extend(correct)
        all_selected.extend(selected)
        all_total_logprobs.extend(total_logprobs)
        all_selected_logprobs.extend(selected_logprobs)

        pbar.update(batch_total)

    pbar.close()

    results_file = f'{results_file_stem}.results.txt'
    save_results(all_correct_total, all_total,
                 # all_bleu,
                 results_dir, results_file)

    save_results_correct(all_correct, results_dir, f'{results_file_stem}.correct.txt')
    save_results_logprobs(all_total_logprobs, results_dir, f'{results_file_stem}.logprobs.txt')
    save_results_logprobs(all_total_logprobs, results_dir, f'{results_file_stem}.scores.txt', flat=True)

    # if save_detailed_results:
    predictions_results_file = f'{results_file_stem}.results.detailed.txt'
    save_predictions(all_correct,
                     all_selected,
                     all_total_logprobs,
                     all_selected_logprobs,
                     data,
                     results_dir,
                     predictions_results_file
                     )

    return {
        'correct': all_correct,
        'total': all_total,
        'data': data,
        'accuracy': all_correct_total / all_total,
    }
