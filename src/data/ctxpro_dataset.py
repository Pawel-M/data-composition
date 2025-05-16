# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
import json
import os
import warnings

import datasets

_DESCRIPTION = """"""
_HOMEPAGE_URL = ""
_CITATION = """"""

_VERSION = "0.0.1"
_BASE_URL = ""

_PHENOMENA = [
    'auxiliary',
    'formality',
    'gender',
    'inflection',
    'animacy',
]

# Please note that only few pairs are shown here. You can use config to generate data for all language pairs
_LANGUAGE_PAIRS = [
    ("en", "de"),
    ("en", "es"),
    ("en", "fr"),
    ("en", "it"),
    ("en", "pl"),
    ("en", "pt"),
    ("en", "ru"),

    ("de", "en"),
    ("es", "en"),
    ("fr", "en"),
    ("it", "en"),
    ("pl", "en"),
    ("pt", "en"),
    ("ru", "en"),
]


def load_phenomena_files_config(phenomena_dir, phenomena_config_file='lang_pair_sets.yaml', return_raw_config=False):
    import yaml
    phenomena_list_file = os.path.join(phenomena_dir, phenomena_config_file)
    with open(phenomena_list_file, 'r') as f:
        raw_config = yaml.safe_load(f)

    configs = {}
    for pair, config in raw_config.items():
        src_lang, tgt_lang = pair.split('-')
        configs[(src_lang, tgt_lang)] = [(phenomenon, subset, os.path.join(phenomena_dir, path))
                                         for phenomenon, subsets in config['rules'].items()
                                         for subset, path in subsets.items()]

    if return_raw_config:
        return configs, raw_config

    return configs


class CtxProConfig(datasets.BuilderConfig):
    def __init__(self, *args,
                 lang1,
                 lang2,
                 phenomenon,
                 files_base_name,  # animacy.opensubtitles
                 json_dir='evalsets',
                 inputs_dir='inputs',
                 ctx_size=0,
                 tgt_phrase_key='expected',
                 tgt_ctx_phrase_key='ref ante head',
                 src_phrase_key='src pronoun',
                 src_ctx_phrase_key='src ante head',
                 ctx_sep='<eos>',
                 splits=('dev', 'devtest', 'test'),
                 # filter_by_ante_distance=False,
                 **kwargs):
        super().__init__(
            *args,
            name=f"{ctx_size}",
            **kwargs,
        )
        self.lang1 = lang1
        self.lang2 = lang2
        self.phenomenon = phenomenon
        self.ctx_size = ctx_size
        self.files_base_name = files_base_name
        self.json_dir = json_dir
        self.inputs_dir = inputs_dir
        self.tgt_phrase_key = tgt_phrase_key
        self.tgt_ctx_phrase_key = tgt_ctx_phrase_key
        self.src_phrase_key = src_phrase_key
        self.src_ctx_phrase_key = src_ctx_phrase_key
        self.ctx_sep = ctx_sep
        self.splits = splits
        # self.filter_by_ante_distance = filter_by_ante_distance


class CtxPro(datasets.GeneratorBasedBuilder):
    # BUILDER_CONFIGS = [
    #     CtxProConfig(
    #         lang1='en', lang2='de',
    #
    #     )
    #     for lang1, lang2, phenomena in _LANGUAGE_PAIRS
    #     for phenomenon in phenomena
    # ]
    BUILDER_CONFIG_CLASS = CtxProConfig

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.config.ctx_size = kwargs.get('ctx_size', 0)
    #     self.config.files_base_name = kwargs.get('files_base_name', 'contrapro')
    #     self.config.tgt_phrase_key = kwargs.get('tgt_phrase_key', 'ref pronoun')

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "segment_id": datasets.Value("int32"),
                    "translation": datasets.Translation(languages=[self.config.lang1, self.config.lang2]),
                    "context_phrase": {
                        self.config.lang1: datasets.Value("string"),
                        self.config.lang2: datasets.Value("string"),
                    },
                    "phrase": {
                        self.config.lang1: datasets.Value("string"),
                        self.config.lang2: datasets.Value("string"),
                    },
                    "context": {
                        self.config.lang1: datasets.Sequence(datasets.Value("string")),
                        self.config.lang2: datasets.Sequence(datasets.Value("string")),
                    },
                    "context_distance": datasets.Value("int32"),
                    "rule": datasets.Value("string"),
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # download_url = _BASE_URL.format(self.config.lang1, self.config.lang2)
        # # path = dl_manager.download_and_extract(download_url)
        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={
                    "datapath": self.base_path,
                    "split_name": split_name,
                },
            )
            for split_name in self.config.splits]

    def load_ctxpro(self,
                    datapath,
                    files_base_name,
                    src_lang,
                    tgt_lang,
                    split_name,
                    json_dir,
                    inputs_dir,
                    ctx_size,
                    ctx_sep, ):

        datapath = os.path.expanduser(datapath)
        file_name_stem = f'{files_base_name}.{src_lang}-{tgt_lang}.{split_name}'
        json_file_path = os.path.join(datapath, json_dir, f'{file_name_stem}.json')
        inputs_file_path = os.path.join(datapath, inputs_dir, file_name_stem)

        with open(json_file_path, 'r') as f:
            data_json = json.load(f)

        with open(inputs_file_path, 'r') as f:
            src_inputs = f.readlines()

        assert len(data_json) == len(src_inputs)

        data = []
        for d, src in zip(data_json, src_inputs):
            src_sentences = [s.strip() for s in src.split(ctx_sep)]
            src = src_sentences[-1]
            tgt = d['ref segment']
            src_context = src_sentences[:-1]
            if ctx_size == 0:
                src_context = []
            elif ctx_size < len(src_context):
                src_context = src_context[-ctx_size:]

            tgt_context = []

            # if self.config.filter_by_ante_distance and self.config.ctx_size is not None and d['ante distance'] > self.config.ctx_size:
            #     continue
            data.append((src, tgt, src_context, tgt_context, d))
        return data

    def _generate_examples(self, datapath, split_name):
        l1, l2 = self.config.lang1, self.config.lang2
        ctx_size = self.config.ctx_size
        ctx_sep = self.config.ctx_sep
        files_base_name = self.config.files_base_name
        json_dir = self.config.json_dir
        inputs_dir = self.config.inputs_dir
        tgt_phrase_key = self.config.tgt_phrase_key
        tgt_ctx_phrase_key = self.config.tgt_ctx_phrase_key
        src_phrase_key = self.config.src_phrase_key
        src_ctx_phrase_key = self.config.src_ctx_phrase_key
        data = self.load_ctxpro(datapath, files_base_name, l1, l2, split_name, json_dir, inputs_dir, ctx_size, ctx_sep)

        for sentence_counter, data_point in enumerate(data):
            d = data_point[-1]
            src = data_point[0]
            tgt = data_point[1]
            src_context = data_point[2]
            tgt_context = data_point[3]

            document_id = d['document id']
            segment_id = d['segment id']
            source = d['src segment'].strip()
            target = d['ref segment'].strip()

            if src != source:
                warnings.warn(f'Error: {src} != {source}')
            if tgt != target:
                warnings.warn(f'Error: {tgt} != {target}')

            src_phrase = d[src_phrase_key].strip() if src_phrase_key in d else None
            tgt_phrase = d[tgt_phrase_key].strip() if tgt_phrase_key in d else None
            src_ctx_phrase = d[src_ctx_phrase_key].strip() if src_ctx_phrase_key in d else None
            tgt_ctx_phrase = d[tgt_ctx_phrase_key].strip() if tgt_ctx_phrase_key in d else None

            rule = d['rule'].strip()

            # source_antecedent = d['src ante head'].strip()
            # source_phrase = d['src pronoun'].strip()
            # target_antecedent = d['ref ante head'].strip() \
            #     if 'ref ante head' in d and d['ref ante head'] is not None \
            #     else None
            # target_pronoun = d[tgt_phrase_key].strip()
            # # wrong_targets = [error['contrastive'].strip() for error in d['errors']]
            # # wrong_targets_pronouns = [error['replacement'].strip() for error in d['errors']]

            ante_distance = d['ante distance'] if 'ante distance' in d else None

            result = (
                sentence_counter,
                {
                    "id": str(sentence_counter),
                    "document_id": document_id,
                    "segment_id": segment_id,
                    "translation": {
                        l1: source,
                        l2: target
                    },
                    "context": {
                        l1: src_context,
                        l2: tgt_context,
                    },
                    "context_phrase": {
                        l1: src_ctx_phrase,
                        l2: tgt_ctx_phrase,
                    },
                    "phrase": {
                        l1: src_phrase,
                        l2: tgt_phrase,
                    },
                    "context_distance": ante_distance,
                    "rule": rule,
                },
            )
            yield result


if __name__ == '__main__':
    from datasets import load_dataset_builder

    ds_builder = ContraPro('../data/ContraPro/ctx0', base_path='../../Datasets/ContraPro', ctx_size=0)
    ds_builder.download_and_prepare('../data/ContraPro/ctx0')
    dataset = ds_builder.as_dataset('train')
    for i in range(10):
        print(dataset[i])
