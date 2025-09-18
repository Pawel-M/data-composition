import os
import warnings
from copy import deepcopy

import datasets
import numpy as np
from datasets import DatasetDict

from common.common_functions import tokenize_all_with_context
from data.contextual_open_subtitles import ContextualOpenSubtitles, find_language_pairs
from data.contrapro_dataset import ContraPro
from data.ctxpro_dataset import CtxPro, load_phenomena_files_config
from config_utils import expand_path


def map_context_size(context_size, context_size_map):
    if context_size in context_size_map:
        return context_size_map[context_size]
    return context_size


def load_dataset(dataset_name, config, ):
    if dataset_name == 'iwslt2017':
        dataset = load_iwslt2017_dataset(
            base_data_dir=config.dataset_path,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            src_ctx_size=config.src_ctx_size,
            tgt_ctx_size=config.tgt_ctx_size,
            splits=config.dataset_splits,
            annotation_data_dir=config.get('dataset_annotation_path', None),
        )

    elif dataset_name == 'opensubtitles':
        DATASET_CTX_SIZE_MAP = {
            0: 1,
            1: 1,
        }
        dataset = load_opensubtitles_dataset(
            raw_data_dir=config.raw_dataset_path,
            base_data_dir=config.dataset_path,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            src_ctx_size=map_context_size(config.src_ctx_size, DATASET_CTX_SIZE_MAP),
            tgt_ctx_size=map_context_size(config.tgt_ctx_size, DATASET_CTX_SIZE_MAP),
            train_size=config.get('train_size', None),
            valid_size=config.get('valid_size', None),
            test_size=config.get('test_size', None),
            split_seed=config.split_seed,
            max_test_size=config.get('max_test_size', 500_000),
            annotation_data_dir=config.get('dataset_annotation_path', None),
        )

    elif dataset_name == 'contrapro':
        dataset = load_contrapro_dataset(
            raw_data_dir=config.raw_dataset_path,
            base_data_dir=config.dataset_path,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            src_ctx_size=config.src_ctx_size,
            tgt_ctx_size=config.tgt_ctx_size,
            filter_context_size=config.get('filter_context_size', False),
        )

    elif dataset_name == 'wmt19':
        dataset = load_wmt19_dataset(
            base_data_dir=config.dataset_path,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
            splits=config.dataset_splits,
        )

    else:
        raise ValueError(f'Dataset {dataset_name} is not supported!')

    return dataset


def load_iwslt2017_dataset(base_data_dir,
                           src_lang,
                           tgt_lang,
                           src_ctx_size,
                           tgt_ctx_size,
                           splits=None,
                           annotation_data_dir=None):
    from data.contextual_iwslt2017 import ContextualIWSLT2017, find_language_pairs

    base_data_dir = expand_path(base_data_dir)

    if annotation_data_dir is not None:
        annotation_data_dir = expand_path(annotation_data_dir)
        phenomena_files_config = load_phenomena_files_config(annotation_data_dir)
        src_phenomena_file_paths = phenomena_files_config.get((tgt_lang, src_lang), None)
        tgt_phenomena_file_paths = phenomena_files_config.get((src_lang, tgt_lang), None)
    else:
        src_phenomena_file_paths = None
        tgt_phenomena_file_paths = None

    (
        lang1, lang2,
        lang1_ctx_size, lang2_ctx_size,
        lang1_phenomena_file_paths, lang2_phenomena_file_paths
    ) = find_language_pairs(
        src_lang, tgt_lang,
        src_ctx_size, tgt_ctx_size,
        src_phenomena_file_paths, tgt_phenomena_file_paths
    )

    data_dir = os.path.join(base_data_dir, f'{lang1}-{lang2}', f'ctx-{lang1_ctx_size}-{lang2_ctx_size}')
    ds_builder = ContextualIWSLT2017(data_dir, f'iwslt2017-{lang1}-{lang2}-ctx-{lang1_ctx_size}-{lang2_ctx_size}',
                                     pair=f'{lang1}-{lang2}', is_multilingual=False,
                                     lang1_ctx_size=lang1_ctx_size,
                                     lang2_ctx_size=lang2_ctx_size,
                                     lang1_phenomena_file_paths=lang1_phenomena_file_paths,
                                     lang2_phenomena_file_paths=lang2_phenomena_file_paths,
                                     )
    print(f'Loading IWSLT2017 dataset from {data_dir}...')
    ds_builder.download_and_prepare(data_dir)
    dataset = ds_builder.as_dataset()

    # rename 'validation' split to 'valid' for consistency
    dataset['valid'] = dataset['validation']
    del dataset['validation']

    if splits is not None:
        dataset = DatasetDict({k: dataset[k] for k in splits})

    return dataset


def load_opensubtitles_dataset(raw_data_dir,
                               base_data_dir,
                               src_lang,
                               tgt_lang,
                               src_ctx_size,
                               tgt_ctx_size,
                               train_size=None,
                               valid_size=None,
                               test_size=None,
                               split_seed=None,
                               max_test_size=500_000,
                               annotation_data_dir=None):
    raw_data_dir = expand_path(raw_data_dir)
    base_data_dir = expand_path(base_data_dir)

    if annotation_data_dir is not None:
        annotation_data_dir = expand_path(annotation_data_dir)
        phenomena_files_config = load_phenomena_files_config(annotation_data_dir)
        src_phenomena_file_paths = phenomena_files_config.get((tgt_lang, src_lang), None)
        tgt_phenomena_file_paths = phenomena_files_config.get((src_lang, tgt_lang), None)
    else:
        src_phenomena_file_paths = None
        tgt_phenomena_file_paths = None

    (
        lang1, lang2,
        lang1_ctx_size, lang2_ctx_size,
        lang1_phenomena_file_paths, lang2_phenomena_file_paths
    ) = find_language_pairs(
        src_lang, tgt_lang,
        src_ctx_size, tgt_ctx_size,
        src_phenomena_file_paths, tgt_phenomena_file_paths
    )

    data_dir = os.path.join(base_data_dir, f'{lang1}-{lang2}', f'ctx-{lang1_ctx_size}-{lang2_ctx_size}')

    ds_builder = ContextualOpenSubtitles(data_dir, f'os-{lang1}-{lang2}-ctx-{lang1_ctx_size}-{lang2_ctx_size}',
                                         base_path=raw_data_dir,
                                         lang1=lang1,
                                         lang2=lang2,
                                         lang1_ctx_size=lang1_ctx_size,
                                         lang2_ctx_size=lang2_ctx_size,
                                         lang1_phenomena_file_paths=lang1_phenomena_file_paths,
                                         lang2_phenomena_file_paths=lang2_phenomena_file_paths, )
    ds_builder.download_and_prepare(data_dir)
    dataset = ds_builder.as_dataset()

    if train_size is not None or valid_size is not None or test_size is not None:
        print(dataset)
        (
            train_dataset,
            valid_dataset,
            test_dataset,
            train_indices,
            valid_indices,
            test_indices,
        ) = split_dataset(
            dataset['train'],
            train_size,
            valid_size,
            test_size,
            split_seed,
            max_test_size,
            return_indices=True,
        )
        ds_dictionary = {}
        if train_dataset is not None:
            ds_dictionary['train'] = train_dataset
        if valid_dataset is not None:
            ds_dictionary['valid'] = valid_dataset
        if test_dataset is not None:
            ds_dictionary['test'] = test_dataset
        dataset = DatasetDict(ds_dictionary)

    return dataset


def load_ctxpro_opensubtitles_dataset(raw_data_dir,
                                      base_data_dir,
                                      processed_dataset_dir,
                                      annotation_data_dir,
                                      src_lang,
                                      tgt_lang,
                                      src_ctx_size,
                                      tgt_ctx_size,
                                      ):
    raw_data_dir = expand_path(raw_data_dir)
    base_data_dir = expand_path(base_data_dir)
    processed_dataset_dir = expand_path(processed_dataset_dir)
    annotation_data_dir = expand_path(annotation_data_dir)

    data_dir = os.path.join(processed_dataset_dir, f'{src_lang}-{tgt_lang}', f'ctx-{src_ctx_size}-{tgt_ctx_size}')
    data_file = f'os-ctxpro-{src_lang}-{tgt_lang}_{src_ctx_size}-{tgt_ctx_size}'
    processed_data_path = os.path.join(data_dir, data_file)

    try:
        print(f'Loading dataset from {processed_data_path}...')
        ctx_dataset = datasets.load_from_disk(processed_data_path)
        return ctx_dataset
    except Exception:
        pass

    print(f'Dataset not found at {processed_data_path}. Processing...')

    def have_test_subset_in_any(x, lang='en'):
        subsets = x['annotation'][lang]['phenomena']['subset']
        return any([s == 'test' for s in subsets])

    def have_test_subset_in_current(x, lang='en'):
        current_locations = x['annotation'][lang]['phenomena']['current_location']
        subsets = x['annotation'][lang]['phenomena']['subset']
        current_subsets = [s for cl, s in zip(current_locations, subsets) if cl == 0]
        return any([s == 'test' for s in current_subsets])

    def select_only_current_annotations(example, lang='en'):
        annotations = example['annotation'][lang]
        phenomena = annotations['phenomena']
        new_phenomenon = []
        new_rule = []
        new_subset = []
        new_current_location = []
        new_context_location = []
        new_expected = []
        for p, r, s, cl, ctxl, ex in zip(phenomena['phenomenon'], phenomena['rule'], phenomena['subset'],
                                         phenomena['current_location'], phenomena['context_location'],
                                         phenomena['expected']):
            if cl != 0:
                continue
            new_phenomenon.append(p)
            new_rule.append(r)
            new_subset.append(s)
            new_current_location.append(cl)
            new_context_location.append(ctxl)
            new_expected.append(ex)

        annotations['phenomena']['phenomenon'] = new_phenomenon
        annotations['phenomena']['rule'] = new_rule
        annotations['phenomena']['subset'] = new_subset
        annotations['phenomena']['current_location'] = new_current_location
        annotations['phenomena']['context_location'] = new_context_location
        annotations['phenomena']['expected'] = new_expected
        example['annotation'][lang] = annotations
        return example

    dataset = load_opensubtitles_dataset(
        raw_data_dir,
        base_data_dir,
        src_lang,
        tgt_lang,
        src_ctx_size,
        tgt_ctx_size,
        annotation_data_dir=annotation_data_dir,
    )

    tgt_ctx_in_current_ds = dataset.filter(lambda x: x['annotation'][tgt_lang]['current_sentence'])

    tgt_ctx_in_current_non_test_ds = tgt_ctx_in_current_ds.filter(
        lambda x: not have_test_subset_in_any(x, lang=tgt_lang))

    tgt_ctx_in_current_test_in_current_ds = tgt_ctx_in_current_ds.filter(
        have_test_subset_in_current,
        fn_kwargs={'lang': tgt_lang})

    tgt_ctx_in_current_non_test_in_current_test_in_ctx_ds = tgt_ctx_in_current_ds.filter(
        lambda x: have_test_subset_in_any(x, lang=tgt_lang) and not have_test_subset_in_current(x, lang=tgt_lang))

    tgt_ctx_ds = DatasetDict(
        {
            'test': tgt_ctx_in_current_test_in_current_ds['train'],
            'dev': tgt_ctx_in_current_non_test_ds['train'],
            'contaminated': tgt_ctx_in_current_non_test_in_current_test_in_ctx_ds['train'],
        }
    )

    tgt_ctx_stripped_ds = tgt_ctx_ds.map(
        select_only_current_annotations,
        fn_kwargs={'lang': tgt_lang})

    print(f'Saving dataset to {processed_data_path}...')
    tgt_ctx_stripped_ds.save_to_disk(processed_data_path)

    return tgt_ctx_stripped_ds


def load_ctxpro_iwslt2017_dataset(
        base_data_dir,
        processed_dataset_dir,
        annotation_data_dir,
        src_lang,
        tgt_lang,
        src_ctx_size,
        tgt_ctx_size,
        splits=None,
        # src_phenomena_file_paths=None,
        # tgt_phenomena_file_paths=None,
):
    base_data_dir = expand_path(base_data_dir)
    processed_dataset_dir = expand_path(processed_dataset_dir)
    annotation_data_dir = expand_path(annotation_data_dir)

    data_dir = os.path.join(processed_dataset_dir, f'{src_lang}-{tgt_lang}', f'ctx-{src_ctx_size}-{tgt_ctx_size}')
    data_file = f'iwslt-ctxpro-{src_lang}-{tgt_lang}_{src_ctx_size}-{tgt_ctx_size}'
    processed_data_path = os.path.join(data_dir, data_file)

    try:
        print(f'Loading dataset from {processed_data_path}...')
        ctx_dataset = datasets.load_from_disk(processed_data_path)
        return ctx_dataset
    except Exception:
        pass

    print(f'Dataset not found at {processed_data_path}. Processing...')

    def have_test_subset_in_any(x, lang='en'):
        subsets = x['annotation'][lang]['phenomena']['subset']
        return any([s == 'test' for s in subsets])

    def have_test_subset_in_current(x, lang='en'):
        current_locations = x['annotation'][lang]['phenomena']['current_location']
        subsets = x['annotation'][lang]['phenomena']['subset']
        current_subsets = [s for cl, s in zip(current_locations, subsets) if cl == 0]
        return any([s == 'test' for s in current_subsets])

    def select_only_current_annotations(example, lang='en'):
        annotations = example['annotation'][lang]
        phenomena = annotations['phenomena']
        new_phenomenon = []
        new_rule = []
        new_subset = []
        new_current_location = []
        new_context_location = []
        new_expected = []
        for p, r, s, cl, ctxl, ex in zip(phenomena['phenomenon'], phenomena['rule'], phenomena['subset'],
                                         phenomena['current_location'], phenomena['context_location'],
                                         phenomena['expected']):
            if cl != 0:
                continue
            new_phenomenon.append(p)
            new_rule.append(r)
            new_subset.append(s)
            new_current_location.append(cl)
            new_context_location.append(ctxl)
            new_expected.append(ex)

        annotations['phenomena']['phenomenon'] = new_phenomenon
        annotations['phenomena']['rule'] = new_rule
        annotations['phenomena']['subset'] = new_subset
        annotations['phenomena']['current_location'] = new_current_location
        annotations['phenomena']['context_location'] = new_context_location
        annotations['phenomena']['expected'] = new_expected
        example['annotation'][lang] = annotations
        return example

    dataset = load_iwslt2017_dataset(base_data_dir,
                                     src_lang,
                                     tgt_lang,
                                     src_ctx_size,
                                     tgt_ctx_size,
                                     splits=splits,
                                     annotation_data_dir=annotation_data_dir)

    tgt_ctx_in_current_ds = dataset.filter(lambda x: x['annotation'][tgt_lang]['current_sentence'])

    tgt_ctx_in_current_non_test_ds = tgt_ctx_in_current_ds.filter(
        lambda x: not have_test_subset_in_any(x, lang=tgt_lang))

    tgt_ctx_in_current_test_in_current_ds = tgt_ctx_in_current_ds.filter(
        have_test_subset_in_current,
        fn_kwargs={'lang': tgt_lang})

    tgt_ctx_in_current_non_test_in_current_test_in_ctx_ds = tgt_ctx_in_current_ds.filter(
        lambda x: have_test_subset_in_any(x, lang=tgt_lang) and not have_test_subset_in_current(x, lang=tgt_lang))

    tgt_ctx_ds = DatasetDict(
        {
            'test': tgt_ctx_in_current_test_in_current_ds['train'],
            'dev': tgt_ctx_in_current_non_test_ds['train'],
            'contaminated': tgt_ctx_in_current_non_test_in_current_test_in_ctx_ds['train'],
        }
    )

    tgt_ctx_stripped_ds = tgt_ctx_ds.map(
        select_only_current_annotations,
        fn_kwargs={'lang': tgt_lang})

    print(f'Saving dataset to {processed_data_path}...')
    tgt_ctx_stripped_ds.save_to_disk(processed_data_path)

    return tgt_ctx_stripped_ds


def load_wmt19_dataset(base_data_dir,
                       src_lang,
                       tgt_lang,
                       splits=None,):
    base_data_dir = expand_path(base_data_dir)

    WMT19_LANG_PAIRS = (
        'cs-en',
        'de-en',
        'fi-en',
        'fr-de',
        'gu-en',
        'kk-en',
        'lt-en',
        'ru-en',
        'zh-en',
    )

    if f'{src_lang}-{tgt_lang}' in WMT19_LANG_PAIRS:
        lang1, lang2 = src_lang, tgt_lang
    elif f'{tgt_lang}-{src_lang}' in WMT19_LANG_PAIRS:
        lang1, lang2 = tgt_lang, src_lang
    else:
        raise ValueError(f'Language pair {src_lang}-{tgt_lang} is not supported for WMT19 dataset!')


    data_dir = os.path.join(base_data_dir, f'{lang1}-{lang2}', 'ctx-0-0')
    dataset = datasets.load_dataset("wmt/wmt19", f'{lang1}-{lang2}', cache_dir=data_dir)
    if splits is not None:
        dataset = DatasetDict({k: dataset[k] for k in splits})

    return dataset


def _tokenize(examples,
              tokenizer,
              src_lang, tgt_lang,
              src_ctx_size, tgt_ctx_size,
              max_length,
              forced_bos_token_id=None):
    source = [t[src_lang] for t in examples['translation']]
    target = [t[tgt_lang] for t in examples['translation']]
    source_context = [c[src_lang] for c in examples['context']]
    target_context = [c[tgt_lang] for c in examples['context']]
    tokenized, _, _ = tokenize_all_with_context(
        tokenizer,
        source, target,
        source_context, target_context,
        src_ctx_size, tgt_ctx_size,
        max_length=max_length,
        return_tensors=None,

    )
    if forced_bos_token_id is not None:
        tokenized['forced_bos_token_id'] = [forced_bos_token_id] * len(examples['translation'])

    return tokenized


def _load_or_prepare_dataset(raw_dataset, tokenize_fn, tokenize_kwargs, dataset_file, try_load, save_prepared):
    if try_load and os.path.exists(dataset_file):
        print(f'Loading tokenized dataset from {dataset_file}...')
        tokenized_dataset = datasets.load_from_disk(dataset_file)
        return tokenized_dataset

    tokenized_dataset = raw_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_dataset.column_names,
        fn_kwargs=tokenize_kwargs,
        keep_in_memory=True,
    )

    if save_prepared:
        print(f'Saving tokenized dataset to {dataset_file}...')
        tokenized_dataset.save_to_disk(dataset_file)

    return tokenized_dataset


def split_dataset(dataset,
                  train_size,
                  valid_size,
                  test_size,
                  split_seed,
                  max_test_size,
                  return_indices=False, ):
    assert train_size is not None or max_test_size is not None

    dataset_size = len(dataset)
    indices = np.random.RandomState(split_seed).permutation(dataset_size)

    train_indices = None
    train_dataset = None
    valid_indices = None
    valid_dataset = None
    test_indices = None
    test_dataset = None

    if train_size is not None:
        train_indices = indices[:train_size]
        train_dataset = dataset.select(train_indices)

    if valid_size is not None:
        if max_test_size is not None:
            valid_indices = indices[dataset_size - max_test_size - valid_size:dataset_size - max_test_size]
        else:
            assert train_size + valid_size <= dataset_size
            valid_indices = indices[train_size:train_size + valid_size]
        valid_dataset = dataset.select(valid_indices)

    if test_size is not None:
        if max_test_size is not None:
            assert test_size <= max_test_size
        elif valid_size is not None:
            assert train_size + valid_size + test_size <= dataset_size
        else:
            assert train_size + test_size <= dataset_size
        test_indices = indices[-test_size:]
        test_dataset = dataset.select(test_indices)

    if return_indices:
        return train_dataset, valid_dataset, test_dataset, train_indices, valid_indices, test_indices

    return train_dataset, valid_dataset, test_dataset


def prepare_dataset(tokenizer,
                    tokenizer_name,
                    dataset,
                    dataset_name,
                    base_data_dir,
                    train_size,
                    valid_size,
                    test_size,
                    split_seed,
                    src_lang,
                    tgt_lang,
                    src_ctx_size,
                    tgt_ctx_size,
                    max_length,
                    save_load_dataset=True,
                    max_test_size=500_000,
                    tokenize_fn=None,
                    tokenize_kwargs=None, ):
    base_data_dir = expand_path(base_data_dir)

    data_dir = os.path.join(base_data_dir, f'{src_lang}-{tgt_lang}', f'ctx-{src_ctx_size}-{tgt_ctx_size}')
    base_data_file = f'{dataset_name}_{tokenizer_name}_{src_lang}-{tgt_lang}_{src_ctx_size}-{tgt_ctx_size}_max-{max_length}'

    if tokenize_fn is None:
        tokenize_fn = _tokenize

    additional_tokenize_kwargs = {
        'tokenizer': tokenizer,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'src_ctx_size': src_ctx_size,
        'tgt_ctx_size': tgt_ctx_size,
        'max_length': max_length,
        # 'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang),
    }
    if hasattr(tokenizer, 'tgt_lang'):
        additional_tokenize_kwargs['forced_bos_token_id'] = tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)

    if tokenize_kwargs is None:
        tokenize_kwargs = additional_tokenize_kwargs
    else:
        tokenize_kwargs = deepcopy(tokenize_kwargs)
        for k, v in additional_tokenize_kwargs.items():
            if k not in tokenize_kwargs:
                tokenize_kwargs[k] = v

    if train_size is None and valid_size is None and test_size is None:
        prepared_dataset = _load_or_prepare_dataset(
            raw_dataset=dataset,
            tokenize_fn=tokenize_fn,
            tokenize_kwargs=tokenize_kwargs,
            dataset_file=os.path.join(data_dir, base_data_file),
            try_load=save_load_dataset,
            save_prepared=save_load_dataset,
        )
        return prepared_dataset

    (
        train_dataset,
        valid_dataset,
        test_dataset,
        train_indices,
        valid_indices,
        test_indices,
    ) = split_dataset(
        dataset,
        train_size,
        valid_size,
        test_size,
        split_seed,
        max_test_size,
        return_indices=True,
    )

    prepared_train_dataset = None
    if train_size is not None:
        train_data_file = f'{base_data_file}_size-{train_size}-{split_seed}'
        prepared_train_dataset = _load_or_prepare_dataset(
            raw_dataset=train_dataset,
            tokenize_fn=tokenize_fn,
            tokenize_kwargs=tokenize_kwargs,
            dataset_file=os.path.join(data_dir, train_data_file),
            try_load=save_load_dataset,
            save_prepared=save_load_dataset,
        )

        if valid_size is None and test_size is None:
            return prepared_train_dataset

    prepared_valid_dataset = None
    if valid_size is not None:
        if max_test_size is not None:
            valid_data_file = f'{base_data_file}_valid_size-{valid_size}-{split_seed}_test-max-{max_test_size}'
        else:
            valid_data_file = f'{base_data_file}_valid_size-{valid_size}-{split_seed}_train-{train_size}'

        prepared_valid_dataset = _load_or_prepare_dataset(
            raw_dataset=valid_dataset,
            tokenize_fn=tokenize_fn,
            tokenize_kwargs=tokenize_kwargs,
            dataset_file=os.path.join(data_dir, valid_data_file),
            try_load=save_load_dataset,
            save_prepared=save_load_dataset,
        )

    prepared_test_dataset = None
    if test_size is not None:
        test_data_file = f'{base_data_file}_test_size-{test_size}-{split_seed}'
        prepared_test_dataset = _load_or_prepare_dataset(
            raw_dataset=test_dataset,
            tokenize_fn=tokenize_fn,
            tokenize_kwargs=tokenize_kwargs,
            dataset_file=os.path.join(data_dir, test_data_file),
            try_load=save_load_dataset,
            save_prepared=save_load_dataset,
        )

    return prepared_train_dataset, prepared_valid_dataset, prepared_test_dataset


def load_contrapro_dataset(raw_data_dir,
                           base_data_dir,
                           src_lang,
                           tgt_lang,
                           src_ctx_size,
                           tgt_ctx_size,
                           filter_context_size=False,
                           ):
    raw_data_dir = expand_path(raw_data_dir)
    base_data_dir = expand_path(base_data_dir)

    print(f'Loading ContraPro dataset from {raw_data_dir}...')
    context_size = max(max(src_ctx_size, tgt_ctx_size), 1)

    dataset_path = os.path.join(base_data_dir, f'ctx-{context_size}')
    ds_builder = ContraPro(dataset_path, f'contrapro-ctx-{context_size}',
                           lang1=src_lang, lang2=tgt_lang,
                           base_path=raw_data_dir,
                           ctx_size=context_size,
                           filter_by_ante_distance=filter_context_size,
                           files_base_name='contrapro'
                           )
    ds_builder.download_and_prepare(dataset_path)
    dataset = ds_builder.as_dataset()
    return dataset


def load_ctxpro_dataset(raw_data_dir,
                        base_data_dir,
                        src_lang,
                        tgt_lang,
                        phenomenon,
                        files_base_name,
                        src_ctx_size,
                        tgt_ctx_size,
                        splits=('dev', 'devtest', 'test'),
                        json_dir='evalsets',
                        inputs_dir='inputs',
                        tgt_phrase_key=None,
                        tgt_ctx_phrase_key=None,
                        src_phrase_key=None,
                        src_ctx_phrase_key=None,
                        ctx_sep='<eos>',
                        ):
    raw_data_dir = expand_path(raw_data_dir)
    base_data_dir = expand_path(base_data_dir)

    context_size = max(max(src_ctx_size, tgt_ctx_size), 1)

    tgt_phrase_key = tgt_phrase_key if tgt_phrase_key is not None else 'expected'

    if phenomenon == 'auxiliary':
        src_phrase_key = src_phrase_key if src_phrase_key is not None else 'src verb'
        if tgt_ctx_phrase_key is not None:
            warnings.warn('In auxiliary phenomenon tgt_ctx_phrase_key is not available!')
        if src_ctx_phrase_key is not None:
            warnings.warn('In auxiliary phenomenon src_ctx_phrase_key is not available!')

    elif phenomenon == 'formality':
        src_phrase_key = src_phrase_key if src_phrase_key is not None else 'src pronoun'
        if tgt_ctx_phrase_key is not None:
            warnings.warn('In auxiliary phenomenon tgt_ctx_phrase_key is not available!')
        if src_ctx_phrase_key is not None:
            warnings.warn('In auxiliary phenomenon src_ctx_phrase_key is not available!')

    elif phenomenon == 'inflection':
        src_phrase_key = src_phrase_key if src_phrase_key is not None else 'src noun'
        if tgt_ctx_phrase_key is not None:
            warnings.warn('In auxiliary phenomenon tgt_ctx_phrase_key is not available!')
        if src_ctx_phrase_key is not None:
            warnings.warn('In auxiliary phenomenon src_ctx_phrase_key is not available!')

    else:
        src_phrase_key = src_phrase_key if src_phrase_key is not None else 'src pronoun'
        tgt_ctx_phrase_key = tgt_ctx_phrase_key if tgt_ctx_phrase_key is not None else 'ref ante head'
        src_ctx_phrase_key = src_ctx_phrase_key if src_ctx_phrase_key is not None else 'src ante head'

    dataset_path = os.path.join(base_data_dir, f'{src_lang}-{tgt_lang}', phenomenon, f'ctx-{context_size}')

    ds_builder = CtxPro(dataset_path,
                        f'ctxpro-{files_base_name}-ctx-{context_size}',
                        lang1=src_lang,
                        lang2=tgt_lang,
                        phenomenon=phenomenon,
                        files_base_name=files_base_name,
                        json_dir=json_dir,
                        inputs_dir=inputs_dir,
                        ctx_size=context_size,
                        tgt_phrase_key=tgt_phrase_key,
                        tgt_ctx_phrase_key=tgt_ctx_phrase_key,
                        src_phrase_key=src_phrase_key,
                        src_ctx_phrase_key=src_ctx_phrase_key,
                        ctx_sep=ctx_sep,
                        splits=splits,
                        base_path=raw_data_dir, )
    ds_builder.download_and_prepare(dataset_path)
    dataset = ds_builder.as_dataset()

    if splits is not None:
        dataset = DatasetDict({k: dataset[k] for k in splits})

    return dataset
