import datasets

from config_utils import Struct, expand_path
from data.loading import (
    load_opensubtitles_dataset,
    prepare_dataset,
    load_iwslt2017_dataset,
    load_ctxpro_opensubtitles_dataset,
    load_ctxpro_iwslt2017_dataset,
    load_wmt19_dataset,
)


def filter_phenomena(example, filtered_phenomena, tgt_lang,
                     unique=False,
                     only_current_sentence=False,
                     filter_out=False):
    annotations = example['annotation'][tgt_lang]
    phenomena = annotations['phenomena']['phenomenon']
    if only_current_sentence:
        current_sentence_phenomena = []
        if annotations['current_sentence']:
            for i in range(len(annotations['phenomena']['current_location'])):
                if annotations['phenomena']['current_location'][i] == 0:
                    current_sentence_phenomena.append(annotations['phenomena']['phenomenon'][i])
        phenomena = current_sentence_phenomena

    if filter_out:
        return not any([phenomenon in phenomena for phenomenon in filtered_phenomena])

    if unique:
        return any([phenomenon in phenomena for phenomenon in filtered_phenomena]) \
               and all([phenomenon in filtered_phenomena for phenomenon in phenomena])

    return any([phenomenon in phenomena for phenomenon in filtered_phenomena])


DATASET_CTX_SIZE_MAP = {
    0: 1,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}


def load_opensubtitles_datasets(
        tokenizer,
        lang_pairs,
        src_ctx_size,
        tgt_ctx_size,
        sample_ctx_size,
        tokenizer_name,
        raw_data_dir,
        base_data_dir,
        annotation_data_dir,
        # dataset_splits: [valid,test]
        train_size,
        # valid_size,
        split_seed,
        max_length,
        sample_ctx_only=False,
        dataset_name_suffix=None,
        tokenize_fn=None,
        tokenize_kwargs=None,
        tokenizer_lang_code_map=None,
):
    dataset_name = f'os_{dataset_name_suffix}' if dataset_name_suffix is not None else 'os'

    raw_data_dir = expand_path(raw_data_dir)
    base_data_dir = expand_path(base_data_dir)
    annotation_data_dir = expand_path(annotation_data_dir)

    train_datasets = []
    raw_datasets = {}
    for lang_pair in lang_pairs:
        src_lang, tgt_lang = lang_pair.split('-')

        if tokenizer_lang_code_map is not None:
            src_lang_code = tokenizer_lang_code_map[src_lang]
            tgt_lang_code = tokenizer_lang_code_map[tgt_lang]
            tokenizer.src_lang = src_lang_code
            tokenizer.tgt_lang = tgt_lang_code
            print(f'Using languages: {src_lang} ({src_lang_code}) -> {tgt_lang} ({tgt_lang_code})')

        if sample_ctx_size:
            assert src_ctx_size == tgt_ctx_size
            if sample_ctx_only:
                context_sizes = [(s, s) for s in range(1, src_ctx_size + 1)]
            else:
                context_sizes = [(s, s) for s in range(src_ctx_size + 1)]
        else:
            context_sizes = [(src_ctx_size, tgt_ctx_size)]
        for src_cs, tgt_cs in context_sizes:
            dataset_src_ctx_size = DATASET_CTX_SIZE_MAP[src_cs]
            dataset_tgt_ctx_size = DATASET_CTX_SIZE_MAP[tgt_cs]
            dataset = load_opensubtitles_dataset(
                raw_data_dir=raw_data_dir,
                base_data_dir=base_data_dir,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=dataset_src_ctx_size,
                tgt_ctx_size=dataset_tgt_ctx_size,
                annotation_data_dir=annotation_data_dir,
            )

            raw_datasets[(dataset_name, lang_pair)] = dataset['train']

            tokenized_train_dataset = prepare_dataset(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                dataset=dataset['train'],
                dataset_name=dataset_name,
                base_data_dir=base_data_dir,
                train_size=train_size,
                valid_size=None,
                test_size=None,
                split_seed=split_seed,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=src_cs,
                tgt_ctx_size=tgt_cs,
                max_length=max_length,
                save_load_dataset=True,
                tokenize_fn=tokenize_fn,
                tokenize_kwargs=tokenize_kwargs,
            )

            train_datasets.append(tokenized_train_dataset)

    if len(train_datasets) > 1:
        train_dataset = datasets.interleave_datasets(train_datasets)
    else:
        train_dataset = train_datasets[0]

    print(f'Dataset loaded: {len(train_dataset)} examples')

    return train_dataset, raw_datasets


def load_iwslt_datasets(
        tokenizer,
        lang_pairs,
        src_ctx_size,
        tgt_ctx_size,
        sample_ctx_size,
        tokenizer_name,
        base_data_dir,
        annotation_data_dir,
        splits,
        max_length,
        filtered_phenomena=None,
        filter_unique=False,
        filter_only_current_sentence=False,
        filter_out=False,
        sample_ctx_only=False,
        dataset_name_suffix=None,
        tokenize_fn=None,
        tokenize_kwargs=None,
        limit_size=None,
        seed=None,
        tokenizer_lang_code_map=None,
):
    base_data_dir = expand_path(base_data_dir)
    if annotation_data_dir is not None:
        annotation_data_dir = expand_path(annotation_data_dir)

    train_datasets = []
    raw_datasets = {}
    for lang_pair in lang_pairs:
        src_lang, tgt_lang = lang_pair.split('-')

        if tokenizer_lang_code_map is not None:
            src_lang_code = tokenizer_lang_code_map[src_lang]
            tgt_lang_code = tokenizer_lang_code_map[tgt_lang]
            tokenizer.src_lang = src_lang_code
            tokenizer.tgt_lang = tgt_lang_code
            print(f'Using languages: {src_lang} ({src_lang_code}) -> {tgt_lang} ({tgt_lang_code})')

        if sample_ctx_size:
            assert src_ctx_size == tgt_ctx_size
            if sample_ctx_only:
                context_sizes = [(s, s) for s in range(1, src_ctx_size + 1)]
            else:
                context_sizes = [(s, s) for s in range(src_ctx_size + 1)]
        else:
            context_sizes = [(src_ctx_size, tgt_ctx_size)]
        for src_cs, tgt_cs in context_sizes:
            dataset = load_iwslt2017_dataset(
                base_data_dir=base_data_dir,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=src_cs,
                tgt_ctx_size=tgt_cs,
                splits=splits,
                annotation_data_dir=annotation_data_dir,
            )

            if filtered_phenomena is not None:
                print(f'Filtering examples with phenomena: {filtered_phenomena}')
                dataset = dataset.filter(
                    filter_phenomena,
                    fn_kwargs={
                        'filtered_phenomena': filtered_phenomena,
                        'tgt_lang': tgt_lang,
                        'unique': filter_unique,
                        'only_current_sentence': filter_only_current_sentence,
                        'filter_out': filter_out,
                    },
                    keep_in_memory=True,
                )
                print('Filtered dataset:', dataset)

            for split in splits:
                dataset_name = f'iwslt2017.{split}'
                if dataset_name_suffix is not None:
                    dataset_name += f'_{dataset_name_suffix}'

                raw_datasets[(dataset_name + f'_ctx-{src_cs}-{tgt_cs}', lang_pair)] = dataset[split]

                tokenized_train_dataset = prepare_dataset(
                    tokenizer=tokenizer,
                    tokenizer_name=tokenizer_name,
                    dataset=dataset[split],
                    dataset_name=dataset_name,
                    base_data_dir=base_data_dir,
                    train_size=limit_size,
                    valid_size=None,
                    test_size=None,
                    split_seed=seed,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    src_ctx_size=src_cs,
                    tgt_ctx_size=tgt_cs,
                    max_length=max_length,
                    save_load_dataset=filtered_phenomena is None,
                    tokenize_fn=tokenize_fn,
                    tokenize_kwargs=tokenize_kwargs,
                )

                train_datasets.append(tokenized_train_dataset)

    if len(train_datasets) > 1:
        train_dataset = datasets.interleave_datasets(train_datasets)
    else:
        train_dataset = train_datasets[0]

    print(f'Dataset loaded: {len(train_dataset)} examples')

    return train_dataset, raw_datasets


def load_ctxpro_datasets(
        lang_pairs,
        src_ctx_size,
        tgt_ctx_size,
        sample_ctx_size,
        raw_data_dir,
        base_data_dir,
        processed_dataset_dir,
        annotation_data_dir,
        tokenizer,
        tokenizer_name,
        max_length,
        filtered_phenomena,
        filter_unique=False,
        filter_only_current_sentence=False,
        filter_out=False,
        limit_size=None,
        seed=None,
        sample_ctx_only=False,
        dataset_name_suffix=None,
        tokenize_fn=None,
        tokenize_kwargs=None,
        tokenizer_lang_code_map=None,
):
    dataset_name = f'os-ctxpro-train_{dataset_name_suffix}' if dataset_name_suffix is not None else 'os-ctxpro-train'

    raw_data_dir = expand_path(raw_data_dir)
    base_data_dir = expand_path(base_data_dir)
    annotation_data_dir = expand_path(annotation_data_dir)

    train_datasets = []
    raw_datasets = {}
    for lang_pair in lang_pairs:
        src_lang, tgt_lang = lang_pair.split('-')

        if tokenizer_lang_code_map is not None:
            src_lang_code = tokenizer_lang_code_map[src_lang]
            tgt_lang_code = tokenizer_lang_code_map[tgt_lang]
            tokenizer.src_lang = src_lang_code
            tokenizer.tgt_lang = tgt_lang_code
            print(f'Using languages: {src_lang} ({src_lang_code}) -> {tgt_lang} ({tgt_lang_code})')

        if sample_ctx_size:
            assert src_ctx_size == tgt_ctx_size
            if sample_ctx_only:
                context_sizes = [(s, s) for s in range(1, src_ctx_size + 1)]
            else:
                context_sizes = [(s, s) for s in range(src_ctx_size + 1)]
        else:
            context_sizes = [(src_ctx_size, tgt_ctx_size)]
        for src_cs, tgt_cs in context_sizes:
            dataset_src_ctx_size = DATASET_CTX_SIZE_MAP[src_cs]
            dataset_tgt_ctx_size = DATASET_CTX_SIZE_MAP[tgt_cs]
            dataset = load_ctxpro_opensubtitles_dataset(
                raw_data_dir=raw_data_dir,
                base_data_dir=base_data_dir,
                processed_dataset_dir=processed_dataset_dir,
                annotation_data_dir=annotation_data_dir,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=dataset_src_ctx_size,
                tgt_ctx_size=dataset_tgt_ctx_size,
            )

            if filtered_phenomena is not None:
                print(f'Filtering examples with phenomena: {filtered_phenomena}')
                dataset = dataset.filter(
                    filter_phenomena,
                    fn_kwargs={
                        'filtered_phenomena': filtered_phenomena,
                        'tgt_lang': tgt_lang,
                        'unique': filter_unique,
                        'only_current_sentence': filter_only_current_sentence,
                        'filter_out': filter_out,
                    },
                    keep_in_memory=True,
                )
                print('Filtered dataset:', dataset)

            raw_datasets[(dataset_name + '_dev', lang_pair)] = dataset['dev']
            raw_datasets[(dataset_name + '_test', lang_pair)] = dataset['test']
            raw_datasets[(dataset_name + '_contaminated', lang_pair)] = dataset['contaminated']

            tokenized_train_dataset = prepare_dataset(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                dataset=dataset['dev'],
                dataset_name=dataset_name,
                base_data_dir=processed_dataset_dir,
                train_size=limit_size,
                valid_size=None,
                test_size=None,
                split_seed=seed,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=src_cs,
                tgt_ctx_size=tgt_cs,
                max_length=max_length,
                save_load_dataset=filtered_phenomena is None,
                tokenize_fn=tokenize_fn,
                tokenize_kwargs=tokenize_kwargs,
            )

            train_datasets.append(tokenized_train_dataset)

    if len(train_datasets) > 1:
        train_dataset = datasets.interleave_datasets(train_datasets)
    else:
        train_dataset = train_datasets[0]

    print(f'Dataset loaded: {len(train_dataset)} examples')

    return train_dataset, raw_datasets


def load_iwslt_ctxpro_datasets(
        lang_pairs,
        src_ctx_size,
        tgt_ctx_size,
        sample_ctx_size,
        base_data_dir,
        processed_dataset_dir,
        annotation_data_dir,
        tokenizer,
        tokenizer_name,
        max_length,
        filtered_phenomena,
        filter_unique=False,
        filter_only_current_sentence=False,
        filter_out=False,
        limit_size=None,
        splits=None,
        sample_ctx_only=False,
        dataset_name_suffix=None,
        tokenize_fn=None,
        tokenize_kwargs=None,
        seed=None,
        tokenizer_lang_code_map=None,
):
    dataset_name = 'iwslt-ctxpro-train'
    if dataset_name_suffix is not None:
        dataset_name += f'_{dataset_name_suffix}'

    base_data_dir = expand_path(base_data_dir)
    annotation_data_dir = expand_path(annotation_data_dir)

    train_datasets = []
    raw_datasets = {}
    for lang_pair in lang_pairs:
        src_lang, tgt_lang = lang_pair.split('-')

        if tokenizer_lang_code_map is not None:
            src_lang_code = tokenizer_lang_code_map[src_lang]
            tgt_lang_code = tokenizer_lang_code_map[tgt_lang]
            tokenizer.src_lang = src_lang_code
            tokenizer.tgt_lang = tgt_lang_code
            print(f'Using languages: {src_lang} ({src_lang_code}) -> {tgt_lang} ({tgt_lang_code})')

        if sample_ctx_size:
            assert src_ctx_size == tgt_ctx_size
            if sample_ctx_only:
                context_sizes = [(s, s) for s in range(1, src_ctx_size + 1)]
            else:
                context_sizes = [(s, s) for s in range(src_ctx_size + 1)]
        else:
            context_sizes = [(src_ctx_size, tgt_ctx_size)]
        for src_cs, tgt_cs in context_sizes:
            dataset_src_ctx_size = DATASET_CTX_SIZE_MAP[src_cs]
            dataset_tgt_ctx_size = DATASET_CTX_SIZE_MAP[tgt_cs]
            dataset = load_ctxpro_iwslt2017_dataset(
                base_data_dir=base_data_dir,
                processed_dataset_dir=processed_dataset_dir,
                annotation_data_dir=annotation_data_dir,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=dataset_src_ctx_size,
                tgt_ctx_size=dataset_tgt_ctx_size,
                splits=splits,
            )

            if filtered_phenomena is not None:
                print(f'Filtering examples with phenomena: {filtered_phenomena}')
                dataset = dataset.filter(
                    filter_phenomena,
                    fn_kwargs={
                        'filtered_phenomena': filtered_phenomena,
                        'tgt_lang': tgt_lang,
                        'unique': filter_unique,
                        'only_current_sentence': filter_only_current_sentence,
                        'filter_out': filter_out,
                    },
                    keep_in_memory=True,
                )
                print('Filtered dataset:', dataset)

            raw_datasets[(dataset_name, lang_pair)] = dataset['test']

            tokenized_train_dataset = prepare_dataset(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                dataset=dataset['test'],
                dataset_name=dataset_name,
                base_data_dir=base_data_dir,
                train_size=limit_size,
                valid_size=None,
                test_size=None,
                split_seed=seed,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=src_cs,
                tgt_ctx_size=tgt_cs,
                max_length=max_length,
                save_load_dataset=False,
                tokenize_fn=tokenize_fn,
                tokenize_kwargs=tokenize_kwargs,
            )

            train_datasets.append(tokenized_train_dataset)

    if len(train_datasets) > 1:
        train_dataset = datasets.interleave_datasets(train_datasets)
    else:
        train_dataset = train_datasets[0]

    print(f'Dataset loaded: {len(train_dataset)} examples')

    return train_dataset, raw_datasets


def load_wmt19_datasets(
        tokenizer,
        lang_pairs,
        tokenizer_name,
        base_data_dir,
        splits,
        max_length,
        dataset_name_suffix=None,
        tokenize_fn=None,
        tokenize_kwargs=None,
        limit_size=None,
        seed=None,
        tokenizer_lang_code_map=None,
):
    base_data_dir = expand_path(base_data_dir)

    train_datasets = []
    raw_datasets = {}
    for lang_pair in lang_pairs:
        src_lang, tgt_lang = lang_pair.split('-')

        if tokenizer_lang_code_map is not None:
            src_lang_code = tokenizer_lang_code_map[src_lang]
            tgt_lang_code = tokenizer_lang_code_map[tgt_lang]
            tokenizer.src_lang = src_lang_code
            tokenizer.tgt_lang = tgt_lang_code
            print(f'Using languages: {src_lang} ({src_lang_code}) -> {tgt_lang} ({tgt_lang_code})')

        dataset = load_wmt19_dataset(
            base_data_dir=base_data_dir,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            splits=splits,
        )

        for split in splits:
            dataset_name = f'wmt19.{split}'
            if dataset_name_suffix is not None:
                dataset_name += f'_{dataset_name_suffix}'

            raw_datasets[(dataset_name, lang_pair)] = dataset[split]

            tokenized_train_dataset = prepare_dataset(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                dataset=dataset[split],
                dataset_name=dataset_name,
                base_data_dir=base_data_dir,
                train_size=limit_size,
                valid_size=None,
                test_size=None,
                split_seed=seed,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_ctx_size=0,
                tgt_ctx_size=0,
                max_length=max_length,
                save_load_dataset=True,
                tokenize_fn=tokenize_fn,
                tokenize_kwargs=tokenize_kwargs,
            )

            train_datasets.append(tokenized_train_dataset)

    if len(train_datasets) > 1:
        train_dataset = datasets.interleave_datasets(train_datasets)
    else:
        train_dataset = train_datasets[0]

    print(f'Dataset loaded: {len(train_dataset)} examples')

    return train_dataset, raw_datasets


def load_train_dataset(tokenizer, config, tokenize_fn=None, tokenize_kwargs=None,
                       dataset_name_suffix=None, os_dataset_name_suffix=None, tokenizer_lang_code_map=None,
                       return_partial_datasets=False):
    os_dataset_name_suffix = os_dataset_name_suffix if os_dataset_name_suffix is not None else dataset_name_suffix

    if 'dataset_name' in config:
        if config.dataset_name == 'opensubtitles':
            train_dataset, all_raw_datasets = load_opensubtitles_datasets(
                tokenizer=tokenizer,
                lang_pairs=config.lang_pairs,
                src_ctx_size=config.src_ctx_size,
                tgt_ctx_size=config.tgt_ctx_size,
                sample_ctx_size=config.sample_ctx_size,
                sample_ctx_only=config.get('sample_ctx_only', False),
                tokenizer_name=config.tokenizer_name,
                raw_data_dir=config.raw_dataset_path,
                base_data_dir=config.dataset_path,
                annotation_data_dir=config.dataset_annotation_path,
                train_size=config.train_size,
                split_seed=config.split_seed,
                max_length=config.max_length,
                dataset_name_suffix=os_dataset_name_suffix,
                tokenize_fn=tokenize_fn,
                tokenize_kwargs=tokenize_kwargs,
                tokenizer_lang_code_map=tokenizer_lang_code_map,
            )
            print(f'Loaded OpenSubtitles dataset: {len(train_dataset)} examples')

        elif config.dataset_name == 'iwslt2017':
            train_dataset, all_raw_datasets = load_iwslt_datasets(
                tokenizer=tokenizer,
                lang_pairs=config.lang_pairs,
                src_ctx_size=config.src_ctx_size,
                tgt_ctx_size=config.tgt_ctx_size,
                sample_ctx_size=config.sample_ctx_size,
                sample_ctx_only=config.get('sample_ctx_only', False),
                tokenizer_name=config.tokenizer_name,
                base_data_dir=config.dataset_path,
                splits=config.dataset_splits,
                max_length=config.max_length,
                dataset_name_suffix=dataset_name_suffix,
                tokenize_fn=tokenize_fn,
                tokenize_kwargs=tokenize_kwargs,
                annotation_data_dir=config.get('dataset_annotation_path'),
                filtered_phenomena=config.get('filtered_phenomena'),
                filter_unique=config.get('filter_unique', False),
                filter_only_current_sentence=config.get('filter_only_current_sentence', False),
                filter_out=config.get('filter_out', False),
                limit_size=config.get('limit_size'),
                seed=config.get('seed'),
                tokenizer_lang_code_map=tokenizer_lang_code_map,
            )
            print(f'Loaded IWSLT2017 dataset: {len(train_dataset)} examples')

        elif config.dataset_name == 'ctxpro':
            train_dataset, all_raw_datasets = load_ctxpro_datasets(
                lang_pairs=config.lang_pairs,
                src_ctx_size=config.src_ctx_size,
                tgt_ctx_size=config.tgt_ctx_size,
                sample_ctx_size=config.sample_ctx_size,
                sample_ctx_only=config.get('sample_ctx_only', False),
                raw_data_dir=config.raw_dataset_path,
                base_data_dir=config.base_dataset_path,
                processed_dataset_dir=config.processed_dataset_path,
                annotation_data_dir=config.dataset_annotation_path,
                max_length=config.max_length,
                tokenizer=tokenizer,
                tokenizer_name=config.tokenizer_name,
                filtered_phenomena=config.get('filtered_phenomena'),
                filter_unique=config.get('filter_unique', False),
                filter_only_current_sentence=config.get('filter_only_current_sentence', False),
                filter_out=config.get('filter_out', False),
                limit_size=config.get('limit_size'),
                seed=config.get('seed'),
                dataset_name_suffix=dataset_name_suffix,
                tokenize_fn=tokenize_fn,
                tokenize_kwargs=tokenize_kwargs,
                tokenizer_lang_code_map=tokenizer_lang_code_map,
            )
            print(f'Loaded ctxpro dataset: {len(train_dataset)} examples')

        elif config.dataset_name == 'iwslt2017_ctxpro':
            train_dataset, all_raw_datasets = load_iwslt_ctxpro_datasets(
                tokenizer=tokenizer,
                lang_pairs=config.lang_pairs,
                src_ctx_size=config.src_ctx_size,
                tgt_ctx_size=config.tgt_ctx_size,
                sample_ctx_size=config.sample_ctx_size,
                sample_ctx_only=config.get('sample_ctx_only', False),
                tokenizer_name=config.tokenizer_name,
                base_data_dir=config.dataset_path,
                splits=config.dataset_splits,
                max_length=config.max_length,
                dataset_name_suffix=dataset_name_suffix,
                tokenize_fn=tokenize_fn,
                tokenize_kwargs=tokenize_kwargs,
                annotation_data_dir=config.get('dataset_annotation_path', None),
                processed_dataset_dir=config.processed_dataset_path,
                filtered_phenomena=config.get('filtered_phenomena'),
                filter_unique=config.get('filter_unique', False),
                filter_only_current_sentence=config.get('filter_only_current_sentence', False),
                filter_out=config.get('filter_out', False),
                limit_size=config.get('limit_size'),
                seed=config.get('seed'),
                tokenizer_lang_code_map=tokenizer_lang_code_map,
            )
            print(f'Loaded IWSLT2017-ctxpro dataset: {len(train_dataset)} examples')

        elif config.dataset_name == 'wmt19':
            train_dataset, all_raw_datasets = load_wmt19_datasets(
                tokenizer=tokenizer,
                lang_pairs=config.lang_pairs,
                tokenizer_name=config.tokenizer_name,
                base_data_dir=config.dataset_path,
                splits=config.dataset_splits,
                max_length=config.max_length,
                dataset_name_suffix=dataset_name_suffix,
                tokenize_fn=tokenize_fn,
                tokenize_kwargs=tokenize_kwargs,
                limit_size=config.get('limit_size'),
                seed=config.get('seed'),
                tokenizer_lang_code_map=tokenizer_lang_code_map,
            )
            print(f'Loaded WMT19 dataset: {len(train_dataset)} examples')

        train_datasets = [train_dataset]
    elif 'datasets' in config:
        train_datasets = []
        all_raw_datasets = {}
        for dataset_config in config.datasets:
            dataset_config = Struct(**dataset_config)
            if dataset_config.dataset_name == 'opensubtitles':
                train_dataset, raw_datasets = load_opensubtitles_datasets(
                    tokenizer=tokenizer,
                    lang_pairs=dataset_config.lang_pairs,
                    src_ctx_size=config.src_ctx_size,
                    tgt_ctx_size=config.tgt_ctx_size,
                    sample_ctx_size=dataset_config.sample_ctx_size,
                    sample_ctx_only=dataset_config.get('sample_ctx_only', False),
                    tokenizer_name=config.tokenizer_name,
                    raw_data_dir=dataset_config.raw_dataset_path,
                    base_data_dir=dataset_config.dataset_path,
                    annotation_data_dir=dataset_config.dataset_annotation_path,
                    train_size=dataset_config.train_size,
                    split_seed=dataset_config.split_seed,
                    max_length=config.max_length,
                    dataset_name_suffix=os_dataset_name_suffix,
                    tokenize_fn=tokenize_fn,
                    tokenize_kwargs=tokenize_kwargs,
                    tokenizer_lang_code_map=tokenizer_lang_code_map,
                )
                print(f'Loaded OpenSubtitles dataset: {len(train_dataset)} examples')

            elif dataset_config.dataset_name == 'iwslt2017':
                train_dataset, raw_datasets = load_iwslt_datasets(
                    tokenizer=tokenizer,
                    lang_pairs=dataset_config.lang_pairs,
                    src_ctx_size=config.src_ctx_size,
                    tgt_ctx_size=config.tgt_ctx_size,
                    sample_ctx_size=dataset_config.sample_ctx_size,
                    sample_ctx_only=dataset_config.get('sample_ctx_only', False),
                    tokenizer_name=config.tokenizer_name,
                    base_data_dir=dataset_config.dataset_path,
                    splits=dataset_config.dataset_splits,
                    max_length=config.max_length,
                    dataset_name_suffix=dataset_name_suffix,
                    tokenize_fn=tokenize_fn,
                    tokenize_kwargs=tokenize_kwargs,
                    annotation_data_dir=dataset_config.get('dataset_annotation_path', None),
                    filtered_phenomena=dataset_config.get('filtered_phenomena'),
                    filter_unique=dataset_config.get('filter_unique', False),
                    filter_only_current_sentence=dataset_config.get('filter_only_current_sentence', False),
                    filter_out=dataset_config.get('filter_out', False),
                    limit_size=dataset_config.get('limit_size'),
                    seed=config.get('seed'),
                    tokenizer_lang_code_map=tokenizer_lang_code_map,
                )
                print(f'Loaded IWSLT2017 dataset: {len(train_dataset)} examples')

            elif dataset_config.dataset_name == 'ctxpro':
                train_dataset, raw_datasets = load_ctxpro_datasets(
                    lang_pairs=dataset_config.lang_pairs,
                    src_ctx_size=config.src_ctx_size,
                    tgt_ctx_size=config.tgt_ctx_size,
                    sample_ctx_size=dataset_config.sample_ctx_size,
                    sample_ctx_only=dataset_config.get('sample_ctx_only', False),
                    raw_data_dir=dataset_config.raw_dataset_path,
                    base_data_dir=dataset_config.base_dataset_path,
                    processed_dataset_dir=dataset_config.processed_dataset_path,
                    annotation_data_dir=dataset_config.dataset_annotation_path,
                    max_length=config.max_length,
                    tokenizer=tokenizer,
                    tokenizer_name=config.tokenizer_name,
                    filtered_phenomena=dataset_config.get('filtered_phenomena'),
                    filter_unique=dataset_config.get('filter_unique', False),
                    filter_only_current_sentence=dataset_config.get('filter_only_current_sentence', False),
                    filter_out=dataset_config.get('filter_out', False),
                    limit_size=dataset_config.get('limit_size'),
                    seed=config.get('seed'),
                    dataset_name_suffix=dataset_name_suffix,
                    tokenize_fn=tokenize_fn,
                    tokenize_kwargs=tokenize_kwargs,
                    tokenizer_lang_code_map=tokenizer_lang_code_map,
                )
                print(f'Loaded ctxpro dataset: {len(train_dataset)} examples')

            elif dataset_config.dataset_name == 'iwslt2017_ctxpro':
                train_dataset, raw_datasets = load_iwslt_ctxpro_datasets(
                    tokenizer=tokenizer,
                    lang_pairs=dataset_config.lang_pairs,
                    src_ctx_size=config.src_ctx_size,
                    tgt_ctx_size=config.tgt_ctx_size,
                    sample_ctx_size=dataset_config.sample_ctx_size,
                    sample_ctx_only=dataset_config.get('sample_ctx_only', False),
                    tokenizer_name=config.tokenizer_name,
                    base_data_dir=dataset_config.dataset_path,
                    splits=dataset_config.dataset_splits,
                    max_length=config.max_length,
                    dataset_name_suffix=dataset_name_suffix,
                    tokenize_fn=tokenize_fn,
                    tokenize_kwargs=tokenize_kwargs,
                    annotation_data_dir=dataset_config.get('dataset_annotation_path', None),
                    processed_dataset_dir=dataset_config.processed_dataset_path,
                    filtered_phenomena=dataset_config.get('filtered_phenomena'),
                    filter_unique=dataset_config.get('filter_unique', False),
                    filter_only_current_sentence=dataset_config.get('filter_only_current_sentence', False),
                    filter_out=dataset_config.get('filter_out', False),
                    limit_size=dataset_config.get('limit_size'),
                    seed=config.get('seed'),
                    tokenizer_lang_code_map=tokenizer_lang_code_map,
                )
                print(f'Loaded IWSLT2017-ctxpro dataset: {len(train_dataset)} examples')

            elif dataset_config.dataset_name == 'wmt19':
                train_dataset, raw_datasets = load_wmt19_datasets(
                    tokenizer=tokenizer,
                    lang_pairs=dataset_config.lang_pairs,
                    tokenizer_name=config.tokenizer_name,
                    base_data_dir=dataset_config.dataset_path,
                    splits=dataset_config.dataset_splits,
                    max_length=config.max_length,
                    dataset_name_suffix=dataset_name_suffix,
                    tokenize_fn=tokenize_fn,
                    tokenize_kwargs=tokenize_kwargs,
                    limit_size=dataset_config.get('limit_size'),
                    seed=config.get('seed'),
                    tokenizer_lang_code_map=tokenizer_lang_code_map,
                )
                print(f'Loaded WMT19 dataset: {len(train_dataset)} examples')

            train_datasets.append(train_dataset)
            all_raw_datasets.update(raw_datasets)

        print(f'Loaded datasets: {len(train_datasets)}')
        seed = config.get('seed', None)
        train_dataset = datasets.concatenate_datasets(train_datasets).shuffle(seed)
        print(f'Combined datasets: {len(train_dataset)} examples')

    else:
        raise ValueError('Unknown dataset name')

    if return_partial_datasets:
        return train_dataset, all_raw_datasets, train_datasets

    return train_dataset, all_raw_datasets
