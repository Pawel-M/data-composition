import argparse
import os
from collections import defaultdict
import random
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import torch
from datasets import DatasetDict
from transformers import MarianMTModel
from transformers import MarianTokenizer

from common.common_functions import tokenize_with_context, score_contextual
from config_utils import load_config, load_configs
from data.loading import load_dataset
from evaluating.translate import translate_dataset


def calculate_pcxmi_dataset(dataset,
                            results_dir,
                            results_file_stem,
                            model,
                            tokenizer,
                            src_lang,
                            tgt_lang,
                            src_ctx_size,
                            tgt_ctx_size,
                            max_length,
                            batch_size,
                            device,
                            plot_metrics=False,
                            save_metrics=False,
                            save_correlations=False,):

    def replace_words(text, replacements, p):
        def replace_match(match):
            return random.choice(replacements) if random.random() < p else match.group(0)

        return re.sub(r'\b\w+\b', replace_match, text)

    def get_vocab(sentences):
        vocab = set()
        for s in sentences:
            vocab.update(re.findall(r'\b\w+\b', s))
        return list(vocab)

    def plot_probs(logprobs1, logprobs2, label1, label2, tokens, save_file=None):
        pcxmi = logprobs2 - logprobs1

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.plot(logprobs1.cpu().exp(), marker='.', linestyle='-', label=label1)
        plt.plot(logprobs2.cpu().exp(), marker='.', linestyle='-', label=label2)
        plt.plot(pcxmi.cpu(), marker='.', linestyle='--', label='Token P-CXMI')
        # tokens = tokenizer.convert_ids_to_tokens(sent_tgt_ids[b, :sent_tgt_lens[b]], skip_special_tokens=False)
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.title(
            f'logP1={logprobs1.sum().item():.2f}, logP2={logprobs2.sum().item():.2f}, P-CXMI={pcxmi.sum().item():.2f}')
        plt.legend()
        plt.tight_layout()
        if save_file is not None:
            plt.savefig(os.path.join(results_dir, 'plots', save_file))
        plt.show()

    def calculate_pcxmi(examples, src_ctx_size, tgt_ctx_size, tokenizer, max_length):
        sources = [t[src_lang] for t in examples['translation']]
        targets = [t[tgt_lang] for t in examples['translation']]
        sources_context = [t[src_lang] for t in examples['context']]
        targets_context = [t[tgt_lang] for t in examples['context']]

        half_batch_size = len(sources) // 2
        shifted_source_context = sources_context[half_batch_size:] + sources_context[:half_batch_size]
        shifted_target_context = targets_context[half_batch_size:] + targets_context[:half_batch_size]

        source_vocab = get_vocab(sources + [s for sentences in sources_context for s in sentences])
        target_vocab = get_vocab(targets + [s for sentences in targets_context for s in sentences])
        randomized_source_context_p01 = [[replace_words(s, source_vocab, 0.1) for s in ctx] for ctx in sources_context]
        randomized_target_context_p01 = [[replace_words(s, target_vocab, 0.1) for s in ctx] for ctx in targets_context]
        randomized_source_context_p02 = [[replace_words(s, source_vocab, 0.2) for s in ctx] for ctx in sources_context]
        randomized_target_context_p02 = [[replace_words(s, target_vocab, 0.2) for s in ctx] for ctx in targets_context]
        randomized_source_context_p04 = [[replace_words(s, source_vocab, 0.4) for s in ctx] for ctx in sources_context]
        randomized_target_context_p04 = [[replace_words(s, target_vocab, 0.4) for s in ctx] for ctx in targets_context]

        with torch.no_grad():
            ctx_token_logprob, ctx_tgt_ids, ctx_tgt_attention_mask, ctx_attentions = score_contextual(
                model=model,
                tokenizer=tokenizer,
                device=device,
                source_context_size=src_ctx_size,
                target_context_size=tgt_ctx_size,
                max_length=max_length,
                sources=sources,
                targets=targets,
                source_contexts=sources_context,
                target_contexts=targets_context,
                output_attentions=True,
            )

            shift_token_logprob, shift_tgt_ids, shift_tgt_attention_mask = score_contextual(
                model=model,
                tokenizer=tokenizer,
                device=device,
                source_context_size=src_ctx_size,
                target_context_size=tgt_ctx_size,
                max_length=max_length,
                sources=sources,
                targets=targets,
                source_contexts=shifted_source_context,
                target_contexts=shifted_target_context,
            )

            # rand_token_logprob_p01, rand_tgt_ids_p01, rand_tgt_attention_mask_p01 = score_contextual(
            #     model=model,
            #     tokenizer=tokenizer,
            #     device=device,
            #     source_context_size=src_ctx_size,
            #     target_context_size=tgt_ctx_size,
            #     max_length=max_length,
            #     sources=sources,
            #     targets=targets,
            #     source_contexts=randomized_source_context_p01,
            #     target_contexts=randomized_target_context_p01,
            # )
            #
            # rand_token_logprob_p02, rand_tgt_ids_p02, rand_tgt_attention_mask_p02 = score_contextual(
            #     model=model,
            #     tokenizer=tokenizer,
            #     device=device,
            #     source_context_size=src_ctx_size,
            #     target_context_size=tgt_ctx_size,
            #     max_length=max_length,
            #     sources=sources,
            #     targets=targets,
            #     source_contexts=randomized_source_context_p02,
            #     target_contexts=randomized_target_context_p02,
            # )
            #
            # rand_token_logprob_p04, rand_tgt_ids_p04, rand_tgt_attention_mask_p04 = score_contextual(
            #     model=model,
            #     tokenizer=tokenizer,
            #     device=device,
            #     source_context_size=src_ctx_size,
            #     target_context_size=tgt_ctx_size,
            #     max_length=max_length,
            #     sources=sources,
            #     targets=targets,
            #     source_contexts=randomized_source_context_p04,
            #     target_contexts=randomized_target_context_p04,
            # )

            sent_token_logprob, sent_tgt_ids, sent_tgt_attention_mask = score_contextual(
                model=model,
                tokenizer=tokenizer,
                device=device,
                source_context_size=0,
                target_context_size=0,
                max_length=max_length,
                sources=sources,
                targets=targets,
                source_contexts=sources_context,
                target_contexts=targets_context,
            )

        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

        ctx_tgt_lens = ctx_tgt_attention_mask.sum(dim=-1)
        sent_tgt_lens = sent_tgt_attention_mask.sum(dim=-1)
        shift_tgt_lens = shift_tgt_attention_mask.sum(dim=-1)
        # rand_tgt_lens_p01 = rand_tgt_attention_mask_p01.sum(dim=-1)
        # rand_tgt_lens_p02 = rand_tgt_attention_mask_p02.sum(dim=-1)
        # rand_tgt_lens_p04 = rand_tgt_attention_mask_p04.sum(dim=-1)

        tgt_lens_diffs = ctx_tgt_lens - sent_tgt_lens
        shift_tgt_lens_diffs = shift_tgt_lens - sent_tgt_lens
        # rand_tgt_lens_diffs_p01 = rand_tgt_lens_p01 - sent_tgt_lens
        # rand_tgt_lens_diffs_p02 = rand_tgt_lens_p02 - sent_tgt_lens
        # rand_tgt_lens_diffs_p04 = rand_tgt_lens_p04 - sent_tgt_lens

        token_pcxmis = []
        pcxmis = []
        avg_token_prob_diffs = []
        max_token_prob_diffs = []
        max_token_pcxmis = []
        avg_token_relu_pcxmis = []
        avg_token_pcxmis = []
        max_shift_token_prob_diffs = []
        avg_relu_token_prob_diffs = []
        shift_pcxmis = []
        shift_token_pcxmis = []
        attention_sum_dec_6_4 = []
        attention_sum_second_dec_6_4 = []
        perplexity_diffs = []
        perplexity_proportions = []
        # rand_pcxmis_p01 = []
        # rand_pcxmis_p02 = []
        # rand_pcxmis_p04 = []
        # rand_max_token_prob_diffs_p01 = []
        # rand_max_token_prob_diffs_p02 = []
        # rand_max_token_prob_diffs_p04 = []
        for b in range(sent_token_logprob.shape[0]):
            ctx_logprobs = ctx_token_logprob[b, tgt_lens_diffs[b]:tgt_lens_diffs[b] + sent_tgt_lens[b]]
            sent_logprobs = sent_token_logprob[b, :sent_tgt_lens[b]]
            shift_ctx_logprobs = shift_token_logprob[b,
                                 shift_tgt_lens_diffs[b]:shift_tgt_lens_diffs[b] + sent_tgt_lens[b]]
            # rand_ctx_logprobs_p01 = rand_token_logprob_p01[b, rand_tgt_lens_diffs_p01[b]:rand_tgt_lens_diffs_p01[b] + sent_tgt_lens[b]]
            # rand_ctx_logprobs_p02 = rand_token_logprob_p02[b,
            #                         rand_tgt_lens_diffs_p02[b]:rand_tgt_lens_diffs_p02[b] + sent_tgt_lens[b]]
            # rand_ctx_logprobs_p04 = rand_token_logprob_p04[b,
            #                         rand_tgt_lens_diffs_p04[b]:rand_tgt_lens_diffs_p04[b] + sent_tgt_lens[b]]

            ctx_probs = ctx_logprobs.exp()
            sent_probs = sent_logprobs.exp()
            shift_probs = shift_ctx_logprobs.exp()
            # rand_probs_p01 = rand_ctx_logprobs_p01.exp()
            # rand_probs_p02 = rand_ctx_logprobs_p02.exp()
            # rand_probs_p04 = rand_ctx_logprobs_p04.exp()

            token_pcxmi = ctx_logprobs - sent_logprobs
            pcxmi = ctx_logprobs.sum() - sent_logprobs.sum()
            token_prob_diff = ctx_probs - sent_probs

            # shift P-CXMI
            shift_token_pcxmi = ctx_logprobs - shift_ctx_logprobs
            shift_pcxmi = ctx_logprobs.sum() - shift_ctx_logprobs.sum()
            shift_token_prob_diff = ctx_probs - shift_probs

            # # random P-CXMI
            # rand_token_pcxi_p01 = ctx_logprobs - rand_ctx_logprobs_p01
            # rand_token_pcxi_p02 = ctx_logprobs - rand_ctx_logprobs_p02
            # rand_token_pcxi_p04 = ctx_logprobs - rand_ctx_logprobs_p04
            # rand_pcxmi_p01 = ctx_logprobs.sum() - rand_ctx_logprobs_p01.sum()
            # rand_pcxmi_p02 = ctx_logprobs.sum() - rand_ctx_logprobs_p02.sum()
            # rand_pcxmi_p04 = ctx_logprobs.sum() - rand_ctx_logprobs_p04.sum()
            # rand_token_prob_diff_p01 = ctx_probs - rand_probs_p01
            # rand_token_prob_diff_p02 = ctx_probs - rand_probs_p02
            # rand_token_prob_diff_p04 = ctx_probs - rand_probs_p04

            # attentions
            dec_6_4_attn = ctx_attentions['decoder'][5][b, 3, ...]
            dec_6_4_attn = dec_6_4_attn[tgt_lens_diffs[b]:tgt_lens_diffs[b] + sent_tgt_lens[b], :]
            attention_sum_dec_6_4.append(dec_6_4_attn[:tgt_lens_diffs[b]].sum().item())
            attention_sum_second_dec_6_4.append(dec_6_4_attn[1:tgt_lens_diffs[b]].sum().item())

            # perplexity
            ctx_loss = -ctx_logprobs.masked_select(sent_tgt_attention_mask[b, :sent_tgt_lens[b]].bool()).mean()
            sent_loss = -sent_logprobs.masked_select(sent_tgt_attention_mask[b, :sent_tgt_lens[b]].bool()).mean()
            ctx_perplexity = torch.exp(ctx_loss)
            sent_perplexity = torch.exp(sent_loss)
            perplexity_diff = sent_perplexity - ctx_perplexity
            perplexity_proportion = sent_perplexity / ctx_perplexity

            avg_token_prob_diffs.append(token_prob_diff.mean().item())
            max_token_prob_diffs.append(token_prob_diff.max().item())
            max_shift_token_prob_diffs.append(shift_token_prob_diff.max().item())
            avg_relu_token_prob_diffs.append(token_prob_diff.relu().mean().item())
            token_pcxmis.append(token_pcxmi)
            pcxmis.append(pcxmi)
            max_token_pcxmis.append(token_pcxmi.max().item())
            avg_token_relu_pcxmis.append(token_pcxmi.relu().mean().item())
            avg_token_pcxmis.append(token_pcxmi.mean().item())
            shift_pcxmis.append(shift_pcxmi)
            shift_token_pcxmis.append(shift_token_pcxmi)
            perplexity_diffs.append(perplexity_diff.item())
            perplexity_proportions.append(perplexity_proportion.item())
            # rand_pcxmis_p01.append(rand_pcxmi_p01)
            # rand_pcxmis_p02.append(rand_pcxmi_p02)
            # rand_pcxmis_p04.append(rand_pcxmi_p04)
            # rand_max_token_prob_diffs_p01.append(rand_token_prob_diff_p01.max().item())
            # rand_max_token_prob_diffs_p02.append(rand_token_prob_diff_p02.max().item())
            # rand_max_token_prob_diffs_p04.append(rand_token_prob_diff_p04.max().item())
            pass

            # plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            # plt.plot(sent_logprobs.cpu().exp(), marker='.', linestyle='-', label='Sent Prob')
            # plt.plot(ctx_logprobs.cpu().exp(), marker='.', linestyle='-', label='Ctx Prob')
            # plt.plot(shift_ctx_logprobs.cpu().exp(), marker='.', linestyle='-', label='Shift Prob')
            # plt.plot(token_pcxmi.cpu(), marker='.', linestyle='--', label='Token P-CXMI')
            # plt.plot(shift_token_pcxmi.cpu(), marker='.', linestyle='--', label='Shift Token P-CXMI')
            # tokens = tokenizer.convert_ids_to_tokens(sent_tgt_ids[b, :sent_tgt_lens[b]], skip_special_tokens=False)
            # plt.xticks(range(len(tokens)), tokens, rotation=90)
            # plt.title(f'logP(sent)={sent_logprobs.sum().item():.2f}, logP(ctx)={ctx_logprobs.sum().item():.2f}, P-CXMI={pcxmi.item():.2f}')
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(os.path.join(results_dir, 'plots', f'{results_file_stem}.token_pcxmi.{b}.png'))
            # plt.show()

        pass

        # cxmi = sent_total_logprob - ctx_total_logprob

        annotated_current = [False] * len(examples['translation'])

        if 'annotation' in examples:
            annotated_current = [a[tgt_lang]['current_sentence'] for a in examples['annotation']]

        pass
        examples['token_pcxmi'] = token_pcxmis
        examples['shift_pcxmi'] = shift_pcxmis
        examples['pcxmi'] = pcxmis
        examples['max_token_pcxmi'] = max_token_pcxmis
        examples['avg_token_relu_pcxmi'] = avg_token_relu_pcxmis
        examples['avg_token_pcxmi'] = avg_token_pcxmis
        examples['shift_token_pcxmi'] = shift_token_pcxmis
        examples['max_token_prob_diff'] = max_token_prob_diffs
        examples['avg_token_prob_diff'] = avg_token_prob_diffs
        examples['avg_relu_token_prob_diff'] = avg_relu_token_prob_diffs
        examples['attention_sum_dec_6_4'] = attention_sum_dec_6_4
        examples['attention_sum_second_dec_6_4'] = attention_sum_second_dec_6_4
        examples['perplexity_diff'] = perplexity_diffs
        examples['perplexity_proportion'] = perplexity_proportions
        examples['shift_max_token_prob_diff'] = max_shift_token_prob_diffs
        # examples['rand_pcxmi_p01'] = rand_pcxmis_p01
        # examples['rand_pcxmi_p02'] = rand_pcxmis_p02
        # examples['rand_pcxmi_p04'] = rand_pcxmis_p04
        # examples['rand_max_token_prob_diff_p01'] = rand_max_token_prob_diffs_p01
        # examples['rand_max_token_prob_diff_p02'] = rand_max_token_prob_diffs_p02
        # examples['rand_max_token_prob_diff_p04'] = rand_max_token_prob_diffs_p04
        examples['random'] = [np.random.random() for _ in pcxmis]

        examples['annotated_current'] = annotated_current
        examples['target'] = targets
        examples['source'] = sources
        return examples

    METRIC_MAP = {
        'pcxmi': 'P-CXMI',
        'max_token_pcxmi': 'Max Token P-CXMI',
        'avg_token_relu_pcxmi': 'Avg Token ReLU P-CXMI',
        'shift_pcxmi': 'Shift P-CXMI',
        'max_token_prob_diff': 'Max Token Prob diff',
        'avg_token_prob_diff': 'Avg Token Prob diff',
        'avg_relu_token_prob_diff': 'Avg ReLU Token Prob diff',
        'attention_sum_second_dec_6_4': 'Attention sum second d-6-4',
        'attention_sum_dec_6_4': 'Attention sum d-6-4',
        'perplexity_diff': 'Perplexity diff',
        'perplexity_proportion': 'Perplexity proportion',
        'shift_max_token_prob_diff': 'Shift Max Token Prob diff',
        # 'random': 'Random',
        # 'rand_pcxmi_p01': 'Random P-CXMI p=0.1',
        # 'rand_pcxmi_p02': 'Random P-CXMI p=0.2',
        # 'rand_pcxmi_p04': 'Random P-CXMI p=0.4',
        # 'rand_max_token_prob_diff_p01': 'Random Max Token Prob diff p=0.1',
        # 'rand_max_token_prob_diff_p02': 'Random Max Token Prob diff p=0.2',
        # 'rand_max_token_prob_diff_p04': 'Random Max Token Prob diff p=0.4',
    }

    COLOR_MAP = defaultdict(lambda: 'tab:gray')
    COLOR_MAP.update({
        'none': 'tab:blue',
        'gender': 'tab:orange',
        'formality': 'tab:green',
        'auxiliary': 'tab:red',
        'animacy': 'tab:purple',
        'inflection': 'tab:brown',
    })

    ALPHA_MAP = defaultdict(lambda: 0.5)
    ALPHA_MAP.update({
        'none': 0.3,
        'gender': 0.6,
        'formality': 0.6,
        'auxiliary': 0.6,
        'animacy': 0.6,
        'inflection': 0.6,
    })


    translated_datasets = {}

    for split, ds in dataset.items():
        print(f'Calculating P-CXMI on "{split}" split with {len(ds)} examples...')
        translated_dataset = ds.map(
            calculate_pcxmi,
            batched=True,
            batch_size=batch_size,
            # remove_columns=ds.column_names,
            fn_kwargs={'src_ctx_size': src_ctx_size,
                       'tgt_ctx_size': tgt_ctx_size,
                       'tokenizer': tokenizer,
                       'max_length': max_length, },
            keep_in_memory=True,
        )
        translated_datasets[split] = translated_dataset

        pcxmis = translated_dataset['pcxmi']
        annotated_currents = translated_dataset['annotated_current']

        if save_metrics:
            print(f'Saving results to {results_dir}...')

            pcxmi_file_path = os.path.join(results_dir, f'{results_file_stem}.{split}.pcxmi.txt')
            print(f'Saving P-CXMIs to {pcxmi_file_path}...')
            with open(pcxmi_file_path, 'w') as f:
                for p in pcxmis:
                    f.write(f'{p}\n')

            richness_file_path = os.path.join(results_dir, f'{results_file_stem}.{split}.richness.txt')
            print(f'Saving richness to {richness_file_path}...')
            with open(richness_file_path, 'w') as f:
                for a in annotated_currents:
                    f.write(f'{1.0 if a else 0.0}\n')

        if plot_metrics or save_correlations:
            correlations = {}

            for metric, metric_name in METRIC_MAP.items():
                values = translated_dataset[metric]
                pearson_corr = scipy.stats.pearsonr(values, annotated_currents)

                phenomena = []
                for e in translated_dataset:
                    if e['annotated_current']:
                        phenomenon = e['annotation'][tgt_lang]['phenomena']['phenomenon']
                        phenomena.append(phenomenon[0] if phenomenon else 'none')
                        continue
                    phenomena.append('none')

                none_phenomena_indices = [i for i, x in enumerate(phenomena) if x == 'none']
                none_values = [values[i] for i in none_phenomena_indices]
                none_annotated_currents = [annotated_currents[i] for i in none_phenomena_indices]

                unique_phenomena = set(phenomena)

                metric_correlations = {
                    'metric': metric,
                    'all': pearson_corr[0],
                }
                correlations[metric] = metric_correlations

                if plot_metrics:
                    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

                for phenomenon in unique_phenomena:
                    print(f'Phenomenon: {phenomenon}')
                    p_indices = [i for i, x in enumerate(phenomena) if x == phenomenon]
                    p_values = [values[i] for i in p_indices]
                    p_annotated_currents = [annotated_currents[i] for i in p_indices]
                    p_corr = scipy.stats.pearsonr(p_values + none_values, p_annotated_currents + none_annotated_currents)
                    metric_correlations[phenomenon] = p_corr[0]
                    if plot_metrics:
                        p_annotated_currents_spread = [a + np.random.normal(scale=0.04) for a in p_annotated_currents]
                        plt.plot(p_values, p_annotated_currents_spread, marker='.', linestyle='None',
                                 label=f'{phenomenon} ({p_corr[0]:.2f})',
                                 color=COLOR_MAP[phenomenon], alpha=ALPHA_MAP[phenomenon])

                if plot_metrics:
                    plt.title(f'{split} {metric_name} vs Richness (Pearson correlation: {pearson_corr[0]:.2f})')
                    plt.xlabel(metric_name)
                    plt.ylabel('Richness')
                    plt.legend()
                    plt.savefig(os.path.join(results_dir, 'plots',
                                             f'{results_file_stem}.{split}.{metric}_vs_richness_by_phenomenon.png'))
                    plt.show()

            if save_correlations:
                correlations = pd.DataFrame(correlations).T
                correlations.to_csv(os.path.join(results_dir, f'{results_file_stem}.{split}.correlations.csv'), index=False)
            pass

    return DatasetDict(translated_datasets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', default=None, type=str,
                        help='config yaml files, parameters of the later configs overwrite the previous ones')
    args = parser.parse_args()
    config = load_configs(args.configs)
    print('config', config)

    tokenizer_path = config.model_path if config.tokenizer_path is None else config.tokenizer_path

    tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
    model = MarianMTModel.from_pretrained(config.model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = torch.compile(model)

    print('tokenizer', tokenizer)
    print('model', model)

    if tokenizer.sep_token is None:
        print('Tokenizer does not have sep_token set!!! The empty separator will be used instead.')

    dataset = load_dataset(dataset_name=config.dataset, config=config)

    results_file_stem = f'{config.dataset}' if config.results_suffix is None \
        else f'{config.dataset}.{config.results_suffix}'
    translations = calculate_pcxmi_dataset(
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
        batch_size=config.batch_size,
        device=device,
    )
