import os

import math
import sacrebleu
import comet
import torch
from datasets import DatasetDict
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.utils import logging

from common.common_functions import tokenize_with_context

logger = logging.get_logger(__name__)


class BatchedPrefixConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(self, prefix_allowed_tokens: torch.Tensor, num_beams: int, padding_token_id: int):
        self._prefix_allowed_tokens = prefix_allowed_tokens
        self._num_beams = num_beams
        self._padding_token_id = padding_token_id
        self._prefix_tokens_beamed = prefix_allowed_tokens.repeat_interleave(num_beams, dim=0)
        self._prefix_non_padding_mask = self._prefix_tokens_beamed.ne(padding_token_id)
        self._beam_range = torch.arange(self._prefix_tokens_beamed.shape[0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # mask = torch.full_like(scores, -math.inf)
        mask = torch.full_like(scores, 1)
        token_pos = input_ids.shape[1] - 1
        if token_pos >= self._prefix_allowed_tokens.shape[1]:
            return scores

        prefix_beam = self._prefix_tokens_beamed[:, token_pos]
        mask[self._beam_range, prefix_beam] = 0
        non_pad_mask = self._prefix_non_padding_mask[:, token_pos].unsqueeze(-1).expand(mask.shape).to(scores.dtype)
        mask = mask * non_pad_mask
        mask = mask.masked_fill(mask.to(torch.bool), -math.inf)
        scores_processed = scores + mask
        return scores_processed


def translate_full_sequence(model, tokenizer, device,
                            source_context_size, target_context_size,
                            source, source_context, target_context,
                            num_beams, max_length, ):
    tokenized_src, source_context_lens = tokenize_with_context(
        tokenizer,
        source, source_context, source_context_size,
        is_target=False,
        max_length=max_length,
    )
    tokenized_src = tokenized_src.to(device)

    forced_bos_token_id = None
    if hasattr(tokenizer, 'tgt_lang'):
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)

    generated_tokens = model.generate(
        **tokenized_src,
        num_beams=num_beams,
        do_sample=False,
        max_length=max_length,
        # output_scores=True,
        # min_new_tokens=1,
        forced_bos_token_id=forced_bos_token_id,
    )
    if target_context_size > 0:
        sep_token_id = tokenizer.sep_token_id
        # eos_token_id = tokenizer.eos_token_id
        sep_token_mask = generated_tokens.eq(sep_token_id)
        num_sep_tokens = sep_token_mask.sum(dim=1).tolist()
        if any([n_sep != n_ctx for n_sep, n_ctx in zip(num_sep_tokens, source_context_lens)]):
            logger.warning(f"Number of generated separator tokens does not match the source context size")
        decoded = []
        target_context = []
        for i in range(generated_tokens.shape[0]):
            # full_decoded = tokenizer.convert_ids_to_tokens(generated_tokens[i].tolist())
            sep_tokens_mask_i = sep_token_mask[i]
            sep_token_indices = sep_tokens_mask_i.nonzero(as_tuple=True)
            if num_sep_tokens[i] == 0:
                tokens = generated_tokens[i]
                context_tokens = []
            else:
                try:
                    last_sep_token_index = sep_token_indices[0][-1]
                    tokens = generated_tokens[i, last_sep_token_index:]
                    context_tokens = generated_tokens[i, :last_sep_token_index]
                except Exception as e:
                    print(sep_token_indices)
                    print(sep_token_indices[-1])
                    print(e)
                    print()
            decoded_current = tokenizer.decode(tokens, skip_special_tokens=True)
            decoded_context = tokenizer.decode(context_tokens, skip_special_tokens=True)
            # print('source', tokenizer.decode(tokenized_src['input_ids'][i], skip_special_tokens=False))
            # print('full', full_decoded)
            # print('num sep', num_sep_tokens[i])
            # print('current', decoded_current)
            # print('context', decoded_context)
            # print()
            decoded.append(decoded_current)
            target_context.append(decoded_context)

        return decoded, target_context
    else:
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return decoded, [''] * len(decoded)


def translate_gold_context(model, tokenizer, device,
                           source_context_size, target_context_size,
                           source, source_context, target_context,
                           num_beams, max_length,
                           forced_bos_token_id=None):
    tokenized_src, source_context_lens = tokenize_with_context(
        tokenizer,
        source, source_context, source_context_size,
        is_target=False,
        max_length=max_length,
    )
    tokenized_src = tokenized_src.to(device)
    empty_target = ''
    if type(source) is list:
        empty_target = [''] * len(source)
    tokenized_tgt_context, target_context_lens = tokenize_with_context(tokenizer,
                                                                       empty_target, target_context,
                                                                       target_context_size,
                                                                       is_target=True)
    tokenized_tgt_context = tokenized_tgt_context.to(device)
    tgt_context_ids = tokenized_tgt_context['input_ids']
    tgt_context_ids = torch.where(tgt_context_ids == tokenizer.eos_token_id,
                                  tokenizer.pad_token_id, tgt_context_ids)
    tgt_context_ids = tgt_context_ids[..., :-1]
    tgt_context_mask = tgt_context_ids.ne(tokenizer.pad_token_id)
    tgt_context_sizes = tgt_context_mask.sum(dim=-1)

    tgt_context_tokens = [tokenizer.convert_ids_to_tokens(tgt_context_ids[i].tolist()) for i in
                          range(len(tgt_context_ids))]

    all_tokens = list(range(tokenizer.vocab_size))

    def target_context_fn(batch_id, input_ids):
        token_pos = input_ids.shape[0] - 1
        if token_pos < tgt_context_sizes[batch_id]:
            return [tgt_context_ids[batch_id, token_pos].item()]

        return all_tokens  # If no match, allow all tokens

    generated_ids = model.generate(
        **tokenized_src,
        num_beams=num_beams,
        do_sample=False,
        max_length=max_length,
        # output_scores=True,
        # min_new_tokens=2,
        # prefix_allowed_tokens_fn=target_context_fn,
        logits_processor=LogitsProcessorList([BatchedPrefixConstrainedLogitsProcessor(
            prefix_allowed_tokens=tgt_context_ids,
            num_beams=num_beams,
            padding_token_id=tokenizer.pad_token_id,
        )]),
    )

    de_contextualize_mask = torch.logical_not(torch.concat([
        torch.ones([tgt_context_mask.shape[0], 1],
                   dtype=tgt_context_mask.dtype).to(tgt_context_mask.device),
        tgt_context_mask,
        torch.zeros([tgt_context_mask.shape[0], max(generated_ids.shape[1] - tgt_context_mask.shape[1] - 1, 0)],
                    dtype=tgt_context_mask.dtype).to(tgt_context_mask.device),
    ], dim=1))
    generated_current_ids = (generated_ids * de_contextualize_mask
                             + tokenizer.pad_token_id * torch.logical_not(de_contextualize_mask))
    decoded = tokenizer.batch_decode(generated_current_ids, skip_special_tokens=True)

    decoded_full = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    decoded_context = tokenizer.batch_decode(tgt_context_ids, skip_special_tokens=True)

    return decoded, decoded_context


def translate_dataset(dataset,
                      results_dir,
                      results_file_stem,
                      model,
                      tokenizer,
                      src_lang,
                      tgt_lang,
                      src_ctx_size,
                      tgt_ctx_size,
                      max_length,
                      num_beams,
                      batch_size,
                      full_sequence_decoding,
                      device, ):
    def translate(examples, src_ctx_size, tgt_ctx_size, tokenizer, max_length, full_sequence_decoding):
        sources = [t[src_lang] for t in examples['translation']]
        targets = [t[tgt_lang] for t in examples['translation']]
        sources_context = [t[src_lang] for t in examples['context']]
        targets_context = [t[tgt_lang] for t in examples['context']]

        translate_fn = translate_full_sequence if full_sequence_decoding or tgt_ctx_size < 1 else translate_gold_context

        with torch.no_grad():
            translation, target_context = translate_fn(model, tokenizer, device, src_ctx_size, tgt_ctx_size, sources,
                                                       sources_context, targets_context,
                                                       num_beams=num_beams, max_length=max_length, )

        examples['predicted'] = translation
        examples['target'] = targets
        examples['source'] = sources
        if full_sequence_decoding:
            examples['predicted_context'] = target_context

        # if include_forced_bos_token:
        #     tokenized['forced_bos_token_id'] = [tokenizer.lang_code_to_id[tokenizer_tgt_lang]] * len(targets)
        return examples

    translated_datasets = {}
    infos = {}
    for split, ds in dataset.items():
        print(f'Translating "{split}" split with {len(ds)} examples...')
        translated_dataset = ds.map(
            translate,
            batched=True,
            batch_size=batch_size,
            remove_columns=ds.column_names,
            fn_kwargs={'src_ctx_size': src_ctx_size,
                       'tgt_ctx_size': tgt_ctx_size,
                       'tokenizer': tokenizer,
                       'max_length': max_length,
                       'full_sequence_decoding': full_sequence_decoding, },
            keep_in_memory=True,
        )
        translated_datasets[split] = translated_dataset

        print(f'Saving results to {results_dir}...')
        predicted = translated_dataset['predicted']
        targets = translated_dataset['target']
        sources = translated_dataset['source']

        preds_file_path = os.path.join(results_dir, f'{results_file_stem}.{split}.preds.txt')
        print(f'Saving predictions to {preds_file_path}...')
        with open(preds_file_path, 'w') as f:
            f.write('\n'.join(predicted))

        targets_file_path = os.path.join(results_dir, f'{results_file_stem}.{split}.targets.txt')
        print(f'Saving targets to {targets_file_path}...')
        with open(targets_file_path, 'w') as f:
            f.write('\n'.join(targets))

        sources_file_path = os.path.join(results_dir, f'{results_file_stem}.{split}.sources.txt')
        print(f'Saving sources to {sources_file_path}...')
        with open(sources_file_path, 'w') as f:
            f.write('\n'.join(sources))

        if 'predicted_context' in translated_dataset.column_names:
            context_file_path = os.path.join(results_dir, f'{results_file_stem}.{split}.target_contexts.txt')
            print(f'Saving target contexts to {context_file_path}...')
            with open(context_file_path, 'w') as f:
                f.write('\n'.join(translated_dataset['predicted_context']))

        bleu_metric = sacrebleu.BLEU()
        bleu = bleu_metric.corpus_score(hypotheses=predicted, references=[targets])

        print(f'BLEU: {bleu.score}')
        print(f'{bleu}')
        print(f'{bleu_metric.get_signature()}')

        comet_model = comet.load_from_checkpoint(comet.download_model("Unbabel/wmt22-comet-da"))
        comet_data = [{"src": src, "mt": pred, "ref": tgt}
                      for src, pred, tgt in zip(sources, predicted, targets)]
        comet_preds = comet_model.predict(comet_data, batch_size=64, gpus=1)
        print(f'COMET: {comet_preds["system_score"]}')


        results_file_path = os.path.join(results_dir, f'{results_file_stem}.{split}.results.txt')
        print(f'Saving results to {results_file_path}...')
        with open(results_file_path, 'w') as f:
            f.write(f'{bleu.score}\n')
            f.write(f'{bleu}\n')
            f.write(f'{bleu_metric.get_signature()}\n')
            f.write(f'{comet_preds["system_score"]}')

        infos[split] = {
            'bleu': bleu.score,
            'bleu_info': bleu,
            'bleu_signature': bleu_metric.get_signature(),
            'comet': comet_preds["system_score"],
            'results_file_path': results_file_path,
            'preds_file_path': preds_file_path,
            'targets_file_path': targets_file_path,
        }

    return DatasetDict(translated_datasets), infos
