# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""IWSLT 2017 dataset """

import os
from collections import defaultdict
from typing import Tuple

import datasets
import json

_HOMEPAGE = "https://sites.google.com/site/iwsltevaluation2017/TED-tasks"

_DESCRIPTION = """\
The IWSLT 2017 Multilingual Task addresses text translation, including zero-shot translation, with a single MT system across all directions including English, German, Dutch, Italian and Romanian. As unofficial task, conventional bilingual text translation is offered between English and Arabic, French, Japanese, Chinese, German and Korean.
"""

_CITATION = """\
@inproceedings{cettolo-etal-2017-overview,
    title = "Overview of the {IWSLT} 2017 Evaluation Campaign",
    author = {Cettolo, Mauro  and
      Federico, Marcello  and
      Bentivogli, Luisa  and
      Niehues, Jan  and
      St{\\"u}ker, Sebastian  and
      Sudoh, Katsuhito  and
      Yoshino, Koichiro  and
      Federmann, Christian},
    booktitle = "Proceedings of the 14th International Conference on Spoken Language Translation",
    month = dec # " 14-15",
    year = "2017",
    address = "Tokyo, Japan",
    publisher = "International Workshop on Spoken Language Translation",
    url = "https://aclanthology.org/2017.iwslt-1.1",
    pages = "2--14",
}
"""

REPO_URL = "https://huggingface.co/datasets/iwslt2017/resolve/main/"
MULTI_URL = REPO_URL + "data/2017-01-trnmted/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.zip"
BI_URL = REPO_URL + "data/2017-01-trnted/texts/{source}/{target}/{source}-{target}.zip"

# XXX: Artificially removed DE from here, as it also exists within bilingual data
MULTI_LANGUAGES = ["en", "it", "nl", "ro"]
BI_LANGUAGES = ["ar", "de", "en", "fr", "ja", "ko", "zh"]
MULTI_PAIRS = [f"{source}-{target}" for source in MULTI_LANGUAGES for target in MULTI_LANGUAGES if source != target]
BI_PAIRS = [
    f"{source}-{target}"
    for source in BI_LANGUAGES
    for target in BI_LANGUAGES
    if source != target and (source == "en" or target == "en")
]

PAIRS = MULTI_PAIRS + BI_PAIRS


def find_language_pairs(src_lang, tgt_lang, src_ctx_size, tgt_ctx_size,
                        src_phenomena_file_paths=None, tgt_phenomena_file_paths=None):
    if f'{src_lang}-{tgt_lang}' in PAIRS:
        lang1, lang2 = src_lang, tgt_lang
        lang1_ctx_size, lang2_ctx_size = src_ctx_size, tgt_ctx_size
        lang1_phenomena_file_paths, lang2_phenomena_file_paths = src_phenomena_file_paths, tgt_phenomena_file_paths
    elif f'{tgt_lang}-{src_lang}' in PAIRS:
        lang1, lang2 = tgt_lang, src_lang
        lang1_ctx_size, lang2_ctx_size = tgt_ctx_size, src_ctx_size
        lang1_phenomena_file_paths, lang2_phenomena_file_paths = tgt_phenomena_file_paths, src_phenomena_file_paths
    else:
        raise AttributeError(
            f'Language pair {src_lang}-{tgt_lang} not available. Choose from {PAIRS}.')

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
            if 'expected' not in data:
                data['expected'] = data['ref pronoun']
            phenomena[key].append((phenomenon, data['rule'], subset, data))

    return phenomena


class ContextualIWSLT2017Config(datasets.BuilderConfig):
    """BuilderConfig for NewDataset"""

    def __init__(self, pair, is_multilingual, lang1_ctx_size=1, lang2_ctx_size=1,
                 lang1_phenomena_file_paths: Tuple[str] = None,
                 lang2_phenomena_file_paths: Tuple[str] = None,
                 **kwargs):
        """

        Args:
            pair: the language pair to consider
            is_multilingual: Is this pair in the multilingual dataset (download source is different)
            **kwargs: keyword arguments forwarded to super.
        """
        self.pair = pair
        self.is_multilingual = is_multilingual
        self.lang1_ctx_size = lang1_ctx_size
        self.lang2_ctx_size = lang2_ctx_size
        self.lang1_phenomena_file_paths = lang1_phenomena_file_paths
        self.lang2_phenomena_file_paths = lang2_phenomena_file_paths

        super().__init__(**kwargs)


class ContextualIWSLT2017(datasets.GeneratorBasedBuilder):
    """The IWSLT 2017 Evaluation Campaign includes a multilingual TED Talks MT task."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    BUILDER_CONFIG_CLASS = ContextualIWSLT2017Config
    BUILDER_CONFIGS = [
        ContextualIWSLT2017Config(
            name="iwslt2017-" + pair,
            description="A small dataset",
            version=datasets.Version("1.0.0"),
            pair=pair,
            is_multilingual=pair in MULTI_PAIRS,
            lang1_ctx_size=1,
            lang2_ctx_size=1,
        )
        for pair in PAIRS
    ]

    def _info(self):
        lang1, lang2 = self.config.pair.split("-")
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "translation": datasets.features.Translation(languages=self.config.pair.split("-")),
                    "context": {
                        lang1: datasets.Sequence(datasets.Value("string")),
                        lang2: datasets.Sequence(datasets.Value("string")),
                    },
                    "annotation": {
                        lang1: {
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
                        lang2: {
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
                    },
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        source, target = self.config.pair.split("-")
        if self.config.is_multilingual:
            dl_dir = dl_manager.download_and_extract(MULTI_URL)
            data_dir = os.path.join(dl_dir, "DeEnItNlRo-DeEnItNlRo")
            years = [2010]
        else:
            bi_url = BI_URL.format(source=source, target=target)
            dl_dir = dl_manager.download_and_extract(bi_url)
            data_dir = os.path.join(dl_dir, f"{source}-{target}")
            years = [2010, 2011, 2012, 2013, 2014, 2015]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            f"train.tags.{self.config.pair}.{source}",
                        )
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            f"train.tags.{self.config.pair}.{target}",
                        )
                    ],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            f"IWSLT17.TED.tst{year}.{self.config.pair}.{source}.xml",
                        )
                        for year in years
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            f"IWSLT17.TED.tst{year}.{self.config.pair}.{target}.xml",
                        )
                        for year in years
                    ],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "source_files": [
                        os.path.join(
                            data_dir,
                            f"IWSLT17.TED.dev2010.{self.config.pair}.{source}.xml",
                        )
                    ],
                    "target_files": [
                        os.path.join(
                            data_dir,
                            f"IWSLT17.TED.dev2010.{self.config.pair}.{target}.xml",
                        )
                    ],
                },
            ),
        ]

    @classmethod
    def _reset_contexts(cls, lang1_ctx_size, lang2_ctx_size):
        lang1_ctx = [''] * lang1_ctx_size
        lang2_ctx = [''] * lang2_ctx_size
        return lang1_ctx, lang2_ctx

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

    def _generate_examples(self, source_files, target_files):
        """Yields examples."""
        id_ = 0
        source, target = self.config.pair.split("-")
        lang1_ctx_size, lang2_ctx_size = self.config.lang1_ctx_size, self.config.lang2_ctx_size

        src_phenomena = load_phenomena_files(self.config.lang1_phenomena_file_paths)
        tgt_phenomena = load_phenomena_files(self.config.lang2_phenomena_file_paths)

        for source_file, target_file in zip(source_files, target_files):
            with open(source_file, "r", encoding="utf-8") as sf:
                with open(target_file, "r", encoding="utf-8") as tf:
                    previous_l1_id, previous_l2_id = None, None
                    for source_row, target_row in zip(sf, tf):
                        source_row = source_row.strip()
                        target_row = target_row.strip()

                        if source_row.startswith("<"):
                            if source_row.startswith("<seg"):
                                # Remove <seg id="1">.....</seg>
                                # Very simple code instead of regex or xml parsing
                                part1 = source_row.split(">")[1]
                                source_row = part1.split("<")[0]
                                part1 = target_row.split(">")[1]
                                target_row = part1.split("<")[0]

                                source_row = source_row.strip()
                                target_row = target_row.strip()

                            elif source_row.startswith("<talkid"):
                                part1 = source_row.split(">")[1]
                                l1_id = part1.split("<")[0]
                                part1 = target_row.split(">")[1]
                                l2_id = part1.split("<")[0]

                                l1_id = l1_id.strip()
                                l2_id = l2_id.strip()
                                continue

                            else:
                                continue

                        if l1_id != previous_l1_id or l2_id != previous_l2_id:
                            lang1_ctx, lang2_ctx = self._reset_contexts(lang1_ctx_size, lang2_ctx_size)
                            ctx_src_phenomena, ctx_tgt_phenomena = [], []
                            segment_id = 1
                        else:
                            segment_id += 1

                        (
                            ctx_src_phenomena,
                            current_src_sentence_phenomenon
                        ) = self.update_phenomena(
                            src_phenomena,
                            ctx_src_phenomena,
                            l1_id,
                            segment_id,
                            lang1_ctx_size
                        )

                        (
                            ctx_tgt_phenomena,
                            current_tgt_sentence_phenomenon
                        ) = self.update_phenomena(
                            tgt_phenomena,
                            ctx_tgt_phenomena,
                            l2_id,
                            segment_id,
                            lang2_ctx_size
                        )

                        result = {
                            "translation": {source: source_row, target: target_row},
                            "context": {
                                source: lang1_ctx.copy(),
                                target: lang2_ctx.copy()
                            },
                            "annotation": {
                                source: {
                                    "current_sentence": current_src_sentence_phenomenon,
                                    "any_sentence": len(ctx_src_phenomena) > 0,
                                    "phenomena": ctx_src_phenomena.copy(),
                                },
                                target: {
                                    "current_sentence": current_tgt_sentence_phenomenon,
                                    "any_sentence": len(ctx_tgt_phenomena) > 0,
                                    "phenomena": ctx_tgt_phenomena.copy(),
                                },
                            }
                        }

                        yield id_, result

                        previous_l1_id, previous_l2_id = l1_id, l2_id

                        lang1_ctx.append(source_row)
                        lang2_ctx.append(target_row)
                        lang1_ctx = lang1_ctx[1:]
                        lang2_ctx = lang2_ctx[1:]

                        id_ += 1
