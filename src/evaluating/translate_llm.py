import os

import comet
import sacrebleu
import torch
from datasets import DatasetDict

from common.common_functions import trim_context


def translate_dataset(dataset,
                      results_dir,
                      results_file_stem,
                      model,
                      tokenizer,
                      prompt_fn,
                      src_lang,
                      tgt_lang,
                      src_ctx_size,
                      tgt_ctx_size,
                      max_prompt_length,
                      max_length,
                      num_beams,
                      do_sample,
                      batch_size,
                      separator='',
                      eos='</s>',
                      unk='<unk>',
                      gold_target_context=False, ):
    device = model.device

    def translate(examples, src_ctx_size, tgt_ctx_size, tokenizer, max_prompt_length, max_length):
        sources = [t[src_lang] for t in examples['translation']]
        targets = [t[tgt_lang] for t in examples['translation']]
        sources_context = [t[src_lang] for t in examples['context']]
        targets_context = [t[tgt_lang] for t in examples['context']]
        sources_context, effective_src_context_size = trim_context(sources_context, src_ctx_size)
        targets_context, effective_tgt_context_size = trim_context(targets_context, tgt_ctx_size)

        prompts = prompt_fn(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            sources=sources,
            source_contexts=sources_context,
            targets=None,
            target_contexts=targets_context if gold_target_context else None,
            sep=separator,
            eos=eos,
        )

        inputs = tokenizer(prompts, return_tensors="pt", max_length=max_prompt_length, padding=True, truncation=True)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length, num_beams=num_beams, do_sample=do_sample)

        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]

        if gold_target_context:
            decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translations = [o.strip() for o in decoded_outputs]
            pass
        else:
            decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            translations = []
            predicted_contexts = []
            for out in decoded_outputs:
                out_split = out.split(separator)
                if len(out_split) > 1:
                    translation = out_split[-1].replace(unk, '').replace(eos, '').strip()
                    translations.append(translation)
                    context = separator.join(out_split[:-1]).replace(unk, '').replace(eos, '').strip()
                    predicted_contexts.append(context)
                else:
                    translations.append(out.replace(unk, '').replace(eos, '').strip())
                    predicted_contexts.append('')
                pass

        examples['predicted'] = translations
        examples['target'] = targets
        examples['source'] = sources
        if not gold_target_context:
            examples['predicted_context'] = predicted_contexts

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
            fn_kwargs={
                'src_ctx_size': src_ctx_size,
                'tgt_ctx_size': tgt_ctx_size,
                'tokenizer': tokenizer,
                'max_prompt_length': max_prompt_length,
                'max_length': max_length,
            },
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
