# import os
# import warnings
#
# import datasets
# import numpy as np
# from datasets import DatasetDict
#
# from common.common_functions import tokenize_all_with_context
# from data.contrapro_dataset import ContraPro
# from data.ctxpro_dataset import CtxPro
#
#
# def load_dataset(dataset_name,
#                  raw_data_dir,
#                  base_data_dir,
#                  src_lang,
#                  tgt_lang,
#                  src_ctx_size,
#                  tgt_ctx_size,
#                  splits=None,
#                  # filter_context_size=False,
#                  ):
#     if dataset_name == 'iwslt2017':
#         dataset = load_iwslt2017_dataset(
#             base_data_dir,
#             src_lang,
#             tgt_lang,
#             src_ctx_size,
#             tgt_ctx_size,
#             splits,
#         )
#     # elif dataset_name == 'wmt2017':
#     #     dataset = load_wmt2017(
#     #         raw_data_dir, base_data_dir,
#     #         src_lang, tgt_lang,
#     #         src_ctx_size, tgt_ctx_size,
#     #         False, True,
#     #     )
#     elif dataset_name == 'contrapro':
#         dataset = load_contrapro_dataset(raw_data_dir,
#                                          base_data_dir,
#                                          src_lang,
#                                          tgt_lang,
#                                          src_ctx_size,
#                                          tgt_ctx_size)
#     else:
#         raise ValueError(f'Dataset {dataset_name} is not supported!')
#
#     return dataset
#
#
# def load_iwslt2017_dataset(base_data_dir,
#                            src_lang,
#                            tgt_lang,
#                            src_ctx_size,
#                            tgt_ctx_size,
#                            splits=None, ):
#     from data.contextual_iwslt2017 import ContextualIWSLT2017, find_language_pairs
#
#     lang1, lang2, lang1_ctx_size, lang2_ctx_size = find_language_pairs(src_lang, tgt_lang, src_ctx_size, tgt_ctx_size)
#
#     data_dir = os.path.join(base_data_dir, f'{lang1}-{lang2}', f'ctx-{lang1_ctx_size}-{lang2_ctx_size}')
#     ds_builder = ContextualIWSLT2017(data_dir, f'iwslt2017-{lang1}-{lang2}-ctx-{lang1_ctx_size}-{lang2_ctx_size}',
#                                      pair=f'{lang1}-{lang2}', is_multilingual=False,
#                                      lang1_ctx_size=lang1_ctx_size,
#                                      lang2_ctx_size=lang2_ctx_size)
#     ds_builder.download_and_prepare(data_dir)
#     dataset = ds_builder.as_dataset()
#
#     # rename 'validation' split to 'valid' for consistency
#     dataset['valid'] = dataset['validation']
#     del dataset['validation']
#
#     if splits is not None:
#         dataset = DatasetDict({k: dataset[k] for k in splits})
#
#     return dataset
#
#
# def load_opensubtitles_dataset(raw_data_dir,
#                                base_data_dir,
#                                src_lang,
#                                tgt_lang,
#                                src_ctx_size,
#                                tgt_ctx_size,
#                                src_phenomena_file_paths=None,
#                                tgt_phenomena_file_paths=None, ):
#     from data.contextual_open_subtitles import ContextualOpenSubtitles, find_language_pairs
#
#     (
#         lang1, lang2,
#         lang1_ctx_size, lang2_ctx_size,
#         lang1_phenomena_file_paths, lang2_phenomena_file_paths
#     ) = find_language_pairs(
#         src_lang, tgt_lang,
#         src_ctx_size, tgt_ctx_size,
#         src_phenomena_file_paths, tgt_phenomena_file_paths
#     )
#
#     data_dir = os.path.join(base_data_dir, f'{lang1}-{lang2}', f'ctx-{lang1_ctx_size}-{lang2_ctx_size}')
#
#     ds_builder = ContextualOpenSubtitles(data_dir, f'os-{lang1}-{lang2}-ctx-{lang1_ctx_size}-{lang2_ctx_size}',
#                                          base_path=raw_data_dir,
#                                          lang1=lang1,
#                                          lang2=lang2,
#                                          lang1_ctx_size=lang1_ctx_size,
#                                          lang2_ctx_size=lang2_ctx_size,
#                                          lang1_phenomena_file_paths=lang1_phenomena_file_paths,
#                                          lang2_phenomena_file_paths=lang2_phenomena_file_paths, )
#     ds_builder.download_and_prepare(data_dir)
#     dataset = ds_builder.as_dataset()
#
#     return dataset
#
#
# def _tokenize(examples,
#               tokenizer,
#               src_lang, tgt_lang,
#               src_ctx_size, tgt_ctx_size,
#               max_length,
#               forced_bos_token_id=None):
#     source = [t[src_lang] for t in examples['translation']]
#     target = [t[tgt_lang] for t in examples['translation']]
#     source_context = [c[src_lang] for c in examples['context']]
#     target_context = [c[tgt_lang] for c in examples['context']]
#     tokenized, _, _ = tokenize_all_with_context(
#         tokenizer,
#         source, target,
#         source_context, target_context,
#         src_ctx_size, tgt_ctx_size,
#         max_length=max_length,
#         return_tensors=None,
#
#     )
#     if forced_bos_token_id is not None:
#         tokenized['forced_bos_token_id'] = [forced_bos_token_id] * len(examples['translation'])
#
#     return tokenized
#
#
# def prepare_dataset(tokenizer,
#                     tokenizer_name,
#                     dataset,
#                     base_data_dir,
#                     train_size,
#                     split_seed,
#                     src_lang,
#                     tgt_lang,
#                     src_ctx_size,
#                     tgt_ctx_size,
#                     max_length):
#     data_dir = os.path.join(base_data_dir, f'{src_lang}-{tgt_lang}', f'ctx-{src_ctx_size}-{tgt_ctx_size}')
#
#     data_file = f'{tokenizer_name}_{src_lang}-{tgt_lang}_{src_ctx_size}-{tgt_ctx_size}'
#     if train_size is not None:
#         data_file += f'_size-{train_size}-{split_seed}'
#     data_file += f'_max-{max_length}.arrow'
#     processed_data_path = os.path.join(data_dir, data_file)
#
#     if os.path.exists(processed_data_path):
#         print(f'Loading tokenized dataset from {processed_data_path}...')
#         tokenized_dataset = datasets.load_from_disk(processed_data_path)
#         return tokenized_dataset
#
#     if train_size is not None:
#         indices = np.random.RandomState(split_seed).permutation(len(dataset))
#         indices = indices[:train_size]
#         dataset = dataset.select(indices)
#
#     tokenized_dataset = dataset.map(
#         _tokenize,
#         batched=True,
#         remove_columns=dataset.column_names,
#         fn_kwargs={
#             'tokenizer': tokenizer,
#             'src_lang': src_lang,
#             'tgt_lang': tgt_lang,
#             'src_ctx_size': src_ctx_size,
#             'tgt_ctx_size': tgt_ctx_size,
#             'max_length': max_length,
#             'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang),
#         })
#
#     tokenized_dataset.save_to_disk(processed_data_path)
#
#     return tokenized_dataset
#
#
# def load_contrapro_dataset(raw_data_dir,
#                            base_data_dir,
#                            src_lang,
#                            tgt_lang,
#                            src_ctx_size,
#                            tgt_ctx_size,
#                            filter_context_size=False, ):
#     print(f'Loading ContraPro dataset from {raw_data_dir}...')
#     context_size = max(max(src_ctx_size, tgt_ctx_size), 1)
#
#     dataset_path = os.path.join(base_data_dir, f'ctx-{context_size}')
#     ds_builder = ContraPro(dataset_path, f'contrapro-ctx-{context_size}',
#                            lang1=src_lang, lang2=tgt_lang,
#                            base_path=raw_data_dir,
#                            ctx_size=context_size,
#                            filter_by_ante_distance=filter_context_size,
#                            files_base_name='contrapro')
#     ds_builder.download_and_prepare(dataset_path)
#     dataset = ds_builder.as_dataset()
#     return dataset
#
#
# def load_ctxpro_dataset(raw_data_dir,
#                         base_data_dir,
#                         src_lang,
#                         tgt_lang,
#                         phenomenon,
#                         files_base_name,
#                         src_ctx_size,
#                         tgt_ctx_size,
#                         splits=('dev', 'devtest', 'test'),
#                         json_dir='evalsets',
#                         inputs_dir='inputs',
#                         tgt_phrase_key=None,
#                         tgt_ctx_phrase_key=None,
#                         src_phrase_key=None,
#                         src_ctx_phrase_key=None,
#                         ctx_sep='<eos>',
#                         ):
#     context_size = max(max(src_ctx_size, tgt_ctx_size), 1)
#
#     tgt_phrase_key = tgt_phrase_key if tgt_phrase_key is not None else 'expected'
#
#     if phenomenon == 'auxiliary':
#         src_phrase_key = src_phrase_key if src_phrase_key is not None else 'src verb'
#         if tgt_ctx_phrase_key is not None:
#             warnings.warn('In auxiliary phenomenon tgt_ctx_phrase_key is not available!')
#         if src_ctx_phrase_key is not None:
#             warnings.warn('In auxiliary phenomenon src_ctx_phrase_key is not available!')
#
#     elif phenomenon == 'formality':
#         src_phrase_key = src_phrase_key if src_phrase_key is not None else 'src pronoun'
#         if tgt_ctx_phrase_key is not None:
#             warnings.warn('In auxiliary phenomenon tgt_ctx_phrase_key is not available!')
#         if src_ctx_phrase_key is not None:
#             warnings.warn('In auxiliary phenomenon src_ctx_phrase_key is not available!')
#
#     elif phenomenon == 'inflection':
#         src_phrase_key = src_phrase_key if src_phrase_key is not None else 'src noun'
#         if tgt_ctx_phrase_key is not None:
#             warnings.warn('In auxiliary phenomenon tgt_ctx_phrase_key is not available!')
#         if src_ctx_phrase_key is not None:
#             warnings.warn('In auxiliary phenomenon src_ctx_phrase_key is not available!')
#
#     else:
#         src_phrase_key = src_phrase_key if src_phrase_key is not None else 'src pronoun'
#         tgt_ctx_phrase_key = tgt_ctx_phrase_key if tgt_ctx_phrase_key is not None else 'ref ante head'
#         src_ctx_phrase_key = src_ctx_phrase_key if src_ctx_phrase_key is not None else 'src ante head'
#
#     dataset_path = os.path.join(base_data_dir, f'{src_lang}-{tgt_lang}', phenomenon, f'ctx-{context_size}')
#
#     ds_builder = CtxPro(dataset_path,
#                         f'ctxpro-{files_base_name}-ctx-{context_size}',
#                         lang1=src_lang,
#                         lang2=tgt_lang,
#                         phenomenon=phenomenon,
#                         files_base_name=files_base_name,
#                         json_dir=json_dir,
#                         inputs_dir=inputs_dir,
#                         ctx_size=context_size,
#                         tgt_phrase_key=tgt_phrase_key,
#                         tgt_ctx_phrase_key=tgt_ctx_phrase_key,
#                         src_phrase_key=src_phrase_key,
#                         src_ctx_phrase_key=src_ctx_phrase_key,
#                         ctx_sep=ctx_sep,
#                         splits=splits,
#                         base_path=raw_data_dir, )
#     ds_builder.download_and_prepare(dataset_path)
#     dataset = ds_builder.as_dataset()
#
#     if splits is not None:
#         dataset = DatasetDict({k: dataset[k] for k in splits})
#
#     return dataset
