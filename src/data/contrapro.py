import dataclasses
import json
import os
import re
import string
from typing import List, Optional

from config_utils import expand_path


@dataclasses.dataclass
class ContrastiveDataPoint:
    source: str
    target: str
    targets: Optional[List[str]]
    source_context: List[str]
    target_context: List[str]
    source_phrase: str
    target_phrase: str
    targets_phrases: Optional[List[str]]
    source_context_phrase: str
    source_context_phrase_id: Optional[int]
    target_context_phrase: Optional[str]
    context_distance: int
    src_phrase_indices: Optional[List[int]]
    src_context_phrase_indices: Optional[List[int]]
    tgt_phrases_indices: Optional[List[int]]
    tgt_context_phrases_indices: Optional[List[int]]
    data: dict


def load_contrapro(dir, context_size, filter_context_size=False, use_json_lines=True, limit_size=None):
    dir = expand_path(dir)

    print(f'Loading data from "{dir}"...')
    print('use_json_lines:', use_json_lines)

    working_context_size = context_size if context_size is not None else 1
    working_context_size = max(working_context_size, 1)


    contrapro_file = os.path.join(dir, 'contrapro.json')
    context_dir = os.path.join(dir, f'ctx{working_context_size}')
    src_file = os.path.join(context_dir, 'contrapro.text.en')
    tgt_file = os.path.join(context_dir, 'contrapro.text.de')
    src_context_file = os.path.join(context_dir, 'contrapro.context.en')
    tgt_context_file = os.path.join(context_dir, 'contrapro.context.de')

    with open(contrapro_file, 'r') as f:
        data_json = json.load(f)

    data = []
    context_id = 0

    with open(src_file, 'r') as f:
        src_lines = f.readlines()

    with open(tgt_file, 'r') as f:
        tgt_lines = f.readlines()

    with open(src_context_file, 'r') as f:
        src_context_lines = f.readlines()

    with open(tgt_context_file, 'r') as f:
        tgt_context_lines = f.readlines()

    sources_mismatched = 0
    targets_mismatched = 0
    contrastives_mismatched = 0
    for i, d in enumerate(data_json):

        context_start_line = context_id * working_context_size
        if context_size is not None and context_size > 0:
            src_context = src_context_lines[context_start_line:context_start_line + working_context_size]
            tgt_context = tgt_context_lines[context_start_line:context_start_line + working_context_size]
        else:
            src_context = []
            tgt_context = []

        # remove newline symbols
        src_context = [c.strip() for c in src_context]
        tgt_context = [c.strip() for c in tgt_context]
        if filter_context_size and context_size is not None and d['ante distance'] > context_size:
            context_id += len(d['errors']) + 1
            continue

        if use_json_lines:
            src_line = d['src segment'].strip()
            tgt_line = d['ref segment'].strip()
            contrastive = [e['contrastive'].strip() for e in d['errors']]
            tgts = [tgt_line] + contrastive
        else:
            src_line = src_lines[context_id].strip()
            tgt_line = tgt_lines[context_id].strip()
            contrastive = [e['contrastive'].strip() for e in d['errors']]
            tgts = [tgt_line] + [tgt_lines[context_id + i + 1].strip() for i in range(len(contrastive))]

        if d['src segment'].strip() != src_line:
            sources_mismatched += 1

        if d['ref segment'].strip() != tgt_line:
            targets_mismatched += 1

        if d['errors'][0]['contrastive'].strip() != tgts[1]:
            contrastives_mismatched += 1

        data.append(ContrastiveDataPoint(
            source=src_line,
            target=tgt_line,
            targets=tgts,
            source_context=src_context,
            target_context=tgt_context,
            source_phrase=d['src pronoun'],
            target_phrase=d['ref pronoun'],
            targets_phrases=[d['ref pronoun']] + [error['replacement'] for error in d['errors']],
            source_context_phrase=d['src ante head'],
            source_context_phrase_id=d['src ante head id'],
            target_context_phrase=d['ref ante head'],
            context_distance=d['ante distance'],
            data=d,
            src_phrase_indices=None,
            src_context_phrase_indices=None,
            tgt_phrases_indices=None,
            tgt_context_phrases_indices=None,
        ))

        context_id += len(tgts)

    print(f'Sources mismatched: {sources_mismatched}')
    print(f'Targets mismatched: {targets_mismatched}')
    print(f'Contrastives mismatched: {contrastives_mismatched}')

    if limit_size is not None and limit_size > 0:
        data = data[:min(limit_size, len(data))]

    return data

