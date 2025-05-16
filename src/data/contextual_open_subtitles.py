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

import json
# Lint as: python3
import os
from collections import defaultdict
from typing import Tuple

import datasets

_DESCRIPTION = """\
This is a new collection of translated movie subtitles from http://www.opensubtitles.org/.

IMPORTANT: If you use the OpenSubtitle corpus: Please, add a link to http://www.opensubtitles.org/ to your website and to your reports and publications produced with the data!

This is a slightly cleaner version of the subtitle collection using improved sentence alignment and better language checking.

62 languages, 1,782 bitexts
total number of files: 3,735,070
total number of tokens: 22.10G
total number of sentence fragments: 3.35G
"""
_HOMEPAGE_URL = "http://opus.nlpl.eu/OpenSubtitles.php"
_CITATION = """\
P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)
"""

_VERSION = "2018.0.0"
_BASE_NAME = "OpenSubtitles.{}.{}"
_BASE_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/{}-{}.txt.zip"

# Please note that only few pairs are shown here. You can use config to generate data for all language pairs
_LANGUAGE_PAIRS = [
    ("de", "el"),
    ("de", "en"),
    ("de", "es"),
    ("de", "it"),
    ("de", "nl"),

    ("el", "en"),
    ("el", "es"),
    ("el", "it"),
    ("el", "nl"),

    ("en", "es"),
    ("en", "it"),
    ("en", "nl"),

    ("es", "it"),
    ("es", "nl"),

    ("it", "nl"),

    ("en", "fr"),
    ("en", "pl"),
    ("en", "pt"),
    ("en", "ru"),
]


def find_language_pairs(src_lang, tgt_lang,
                        src_ctx_size, tgt_ctx_size,
                        src_phenomena_file_paths=None, tgt_phenomena_file_paths=None):
    if (src_lang, tgt_lang) in _LANGUAGE_PAIRS:
        lang1, lang2 = src_lang, tgt_lang
        lang1_ctx_size, lang2_ctx_size = src_ctx_size, tgt_ctx_size
        lang1_phenomena_file_paths, lang2_phenomena_file_paths = src_phenomena_file_paths, tgt_phenomena_file_paths
    elif (tgt_lang, src_lang) in _LANGUAGE_PAIRS:
        lang1, lang2 = tgt_lang, src_lang
        lang1_ctx_size, lang2_ctx_size = tgt_ctx_size, src_ctx_size
        lang1_phenomena_file_paths, lang2_phenomena_file_paths = tgt_phenomena_file_paths, src_phenomena_file_paths
    else:
        raise AttributeError(
            f'Language pair {src_lang}-{tgt_lang} not available. Choose from {_LANGUAGE_PAIRS}.')

    return lang1, lang2, lang1_ctx_size, lang2_ctx_size, lang1_phenomena_file_paths, lang2_phenomena_file_paths


def load_phenomena_files(phenomena_file_paths):
    phenomena = defaultdict(list)
    if not phenomena_file_paths:
        return phenomena

    print('Loading phenomena files...')
    print(phenomena_file_paths)

    for phenomenon, subset, file_path in phenomena_file_paths:
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        for data in json_data:
            key = (data['document id'], data['segment id'])
            phenomena[key].append((phenomenon, data['rule'], subset, data))

    return phenomena


class ContextualOpenSubtitlesConfig(datasets.BuilderConfig):
    def __init__(self,
                 *args,
                 lang1=None,
                 lang2=None,
                 lang1_ctx_size=1,
                 lang2_ctx_size=1,
                 lang1_phenomena_file_paths: Tuple[str] = None,
                 lang2_phenomena_file_paths: Tuple[str] = None,
                 **kwargs):
        super().__init__(
            *args,
            name=f"{lang1}-{lang2}",
            **kwargs,
        )
        self.lang1 = lang1
        self.lang2 = lang2
        self.lang1_ctx_size = lang1_ctx_size
        self.lang2_ctx_size = lang2_ctx_size
        self.lang1_phenomena_file_paths = lang1_phenomena_file_paths
        self.lang2_phenomena_file_paths = lang2_phenomena_file_paths


class ContextualOpenSubtitles(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ContextualOpenSubtitlesConfig(
            lang1=lang1,
            lang2=lang2,
            description=f"Translating {lang1} to {lang2} or vice versa",
            version=datasets.Version(_VERSION),
        )
        for lang1, lang2 in _LANGUAGE_PAIRS
    ]
    BUILDER_CONFIG_CLASS = ContextualOpenSubtitlesConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "meta": {
                        "year": datasets.Value("uint32"),
                        "document id": datasets.Value("string"),
                        "segment_id": datasets.Value("uint32"),
                        "context_segment_ids": datasets.Sequence(datasets.Sequence(datasets.Value("uint32"))),
                    },
                    "translation": datasets.Translation(languages=(self.config.lang1, self.config.lang2)),
                    "context": {
                        self.config.lang1: datasets.Sequence(datasets.Value("string")),
                        self.config.lang2: datasets.Sequence(datasets.Value("string")),
                    },
                    "annotation": {
                        self.config.lang1: {
                            "current_sentence": datasets.Value("bool"),
                            "any_sentence": datasets.Value("bool"),
                            "phenomena": datasets.Sequence({
                                "phenomenon": datasets.Value("string"),
                                "rule": datasets.Value("string"),
                                "subset": datasets.Value("string"),
                                "current_location": datasets.Value("uint32"),
                                "context_location": datasets.Value("uint32"),
                                "expected": datasets.Value("string"),
                            })
                        },
                        self.config.lang2: {
                            "current_sentence": datasets.Value("bool"),
                            "any_sentence": datasets.Value("bool"),
                            "phenomena": datasets.Sequence({
                                "phenomenon": datasets.Value("string"),
                                "rule": datasets.Value("string"),
                                "subset": datasets.Value("string"),
                                "current_location": datasets.Value("uint32"),
                                "context_location": datasets.Value("uint32"),
                                "expected": datasets.Value("string"),
                            })
                        },
                    }
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # def _base_url(lang1, lang2):
        #     return _BASE_URL.format(lang1, lang2)
        #
        # download_url = _base_url(self.config.lang1, self.config.lang2)
        # path = dl_manager.download_and_extract(download_url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapath": self.base_path},
            )
        ]

    @classmethod
    def _extract_info(cls, sentence_id):
        # see https://github.com/huggingface/datasets/issues/1844
        # sentence ids have the following format: en/2017/7006210/7050201.xml.gz
        # lang/year/imdb_id/opensubtitles_id.xml.gz
        parts = sentence_id[: -len(".xml.gz")].split("/")
        parts.pop(0)  # remove lang, we do not need it

        # returns year, imdb_id, opensubtitles_id
        return tuple(map(int, parts))

    # @classmethod
    # def _reset_contexts(cls, lang1_ctx_size, lang2_ctx_size):
    #     lang1_ctx = [''] * lang1_ctx_size
    #     lang2_ctx = [''] * lang2_ctx_size
    #     lang1_ctx_ids = [[]] * lang1_ctx_size
    #     lang2_ctx_ids = [[]] * lang2_ctx_size
    #     return lang1_ctx, lang2_ctx, lang1_ctx_ids, lang2_ctx_ids

    def update_phenomena(self, phenomena, ctx_phenomena, doc_id, segment_id, ctx_size):

        for phenomenon in ctx_phenomena:
            phenomenon['current_location'] += 1
            if phenomenon['context_location'] is not None:
                phenomenon['context_location'] += 1
        ctx_phenomena = [ph.copy() for ph in ctx_phenomena if ph['current_location'] <= ctx_size]

        current_sentence_phenomenon = False
        for phenomenon, rule, subset, data in phenomena.get((doc_id, segment_id), []):
            current_sentence_phenomenon = True
            context_location = data['ante distance'] if 'ante distance' in data else None
            ctx_phenomena.append({
                "phenomenon": phenomenon,
                "rule": rule,
                "subset": subset,
                "current_location": 0,
                "context_location": context_location,
                "expected": data['expected'],
            })

        return ctx_phenomena, current_sentence_phenomenon

    def _generate_examples(self, datapath):
        l1, l2 = self.config.lang1, self.config.lang2
        lang1_ctx_size, lang2_ctx_size = self.config.lang1_ctx_size, self.config.lang2_ctx_size
        lang_pair = f'{l1}-{l2}'

        src_phenomena = load_phenomena_files(self.config.lang1_phenomena_file_paths)
        tgt_phenomena = load_phenomena_files(self.config.lang2_phenomena_file_paths)

        tsv_path = os.path.join(datapath, lang_pair, f'{lang_pair}.tsv')
        with open(tsv_path, 'r') as f:
            src_ctx, tgt_ctx = [], []
            ctx_ids = []
            ctx_src_phenomena, ctx_tgt_phenomena = [], []
            segment_id = 1
            prev_doc_id = None
            for sentence_counter, line in enumerate(f):
                doc_id, src, tgt = [line_part.strip() for line_part in line.split('\t')]
                _, year, _ = [part.strip() for part in doc_id.split('/')]

                if prev_doc_id != doc_id:
                    prev_doc_id = doc_id
                    segment_id = 1
                    src_ctx, tgt_ctx = [], []
                    ctx_ids = []
                    ctx_src_phenomena, ctx_tgt_phenomena = [], []
                else:
                    segment_id += 1

                (
                    ctx_src_phenomena,
                    current_src_sentence_phenomenon
                ) = self.update_phenomena(
                    src_phenomena,
                    ctx_src_phenomena,
                    doc_id,
                    segment_id,
                    lang1_ctx_size
                )

                (
                    ctx_tgt_phenomena,
                    current_tgt_sentence_phenomenon
                ) = self.update_phenomena(
                    tgt_phenomena,
                    ctx_tgt_phenomena,
                    doc_id,
                    segment_id,
                    lang2_ctx_size
                )

                result = (
                    sentence_counter,
                    {
                        "id": str(sentence_counter),
                        "meta": {
                            "year": int(year),
                            "document id": doc_id,
                            "segment_id": segment_id,
                            "context_segment_ids": ctx_ids,
                        },
                        "translation": {l1: src, l2: tgt},
                        "context": {
                            l1: src_ctx.copy(),
                            l2: tgt_ctx.copy()
                        },
                        "annotation": {
                            l1: {
                                "current_sentence": current_src_sentence_phenomenon,
                                "any_sentence": len(ctx_src_phenomena) > 0,
                                "phenomena": ctx_src_phenomena.copy(),
                            },
                            l2: {
                                "current_sentence": current_tgt_sentence_phenomenon,
                                "any_sentence": len(ctx_tgt_phenomena) > 0,
                                "phenomena": ctx_tgt_phenomena.copy(),
                            },
                        }
                    },
                )

                src_ctx.append(src)
                tgt_ctx.append(tgt)
                src_ctx = src_ctx[-lang1_ctx_size:]
                tgt_ctx = tgt_ctx[-lang2_ctx_size:]
                yield result


if __name__ == '__main__':
    ds_builder = ContextualOpenSubtitles('../data/OpenSubtitles_hf/de-en', 'de-en', lang1_ctx_size=5, lang2_ctx_size=5)
    ds_builder.download_and_prepare('../data/OpenSubtitles_hf/de-en')
