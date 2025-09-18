LANG_MAP = {
    'en': 'English',
    'de': 'German',
    'fr': 'French',
    'es': 'Spanish',
    'it': 'Italian',
    'nl': 'Dutch',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ko': 'Korean',
    'zh': 'Chinese',
    # 'pl': 'pol_Latn',
    # 'el': 'ell_Grek',
}


# def generate_single_prompt(src_lang, tgt_lang, source, target=None):
#     # text = "English: My name is TowerBase.\nPortuguese:"
#     prompt = f'{LANG_MAP[src_lang]}: {source}\n{LANG_MAP[tgt_lang]}: '
#     if target is not None:
#         prompt += target
#     return prompt
#
#
# def generate_prompt(sources, sources_context, targets_context, src_lang, tgt_lang, sep='\n\n'):
#     assert len(sources) == len(sources_context) == len(targets_context)
#     # sep = '\n\n'
#     prompts = []
#     for source, source_context, target_context in zip(sources, sources_context, targets_context):
#         assert len(source_context) == len(target_context)
#
#         prompt = ''
#         for src_ctx, tgt_ctx in zip(source_context, target_context):
#             prompt += generate_single_prompt(src_lang, tgt_lang, src_ctx, tgt_ctx) + sep
#
#         prompt += generate_single_prompt(src_lang, tgt_lang, source)
#         prompts.append(prompt)
#
#     return prompts


def combine_context_and_current(current, contexts, sep):
    all = contexts + [current]
    all = [s.strip() for s in all if s is not None and len(s) > 0]
    full = sep.join(all)
    return full


def generate_full_sequence_prompt(
        src_lang, tgt_lang,
        sources, source_contexts,
        targets=None, target_contexts=None,
        sep='<sep>',
        eos='</s>',
        include_pure_prompt=False,
):
    if sep is None:
        sep = ' '

    # add spaces before and after separator
    if sep not in [' ', '\n']:
        if sep[0] != ' ':
            sep = ' ' + sep
        if sep[-1] != ' ':
            sep += ' '

    prompts = []
    pure_prompts = []
    for i in range(len(sources)):
        source = sources[i]
        source_context = source_contexts[i]
        full_source = combine_context_and_current(source, source_context, sep)

        prompt = f'{LANG_MAP[src_lang]}: {full_source}\n{LANG_MAP[tgt_lang]}:'
        pure_prompts.append(prompt)

        if target_contexts is not None:
            targets_context = target_contexts[i]
            if targets is not None:
                target = targets[i]
                full_target = combine_context_and_current(target, targets_context, sep)
                prompt += ' ' + full_target + eos
            else:
                full_target = combine_context_and_current(None, targets_context, sep)
                prompt += ' ' + full_target
                if len(targets_context) > 0:
                    prompt += sep

        prompts.append(prompt)

    if include_pure_prompt:
        return prompts, pure_prompts

    return prompts

# def generate_prompt(src_lang, tgt_lang, source, target=None, eos='</s>'):
#     prompt = f'{src_lang}: {source}\n{tgt_lang}: '
#
#     if target is not None:
#         prompt += target + ' ' + eos
#
#     return prompt
#
#
# src_lang_code = 'en'
# tgt_lang_code = 'de'
# src_lang = 'English'
# tgt_lang = 'German'
#
# print(generate_prompt(src_lang, tgt_lang, data_train[0]["translation"][src_lang_code],
#                       data_train[0]["translation"][tgt_lang_code]))
# print(generate_prompt(src_lang, tgt_lang, data_train[0]["translation"][src_lang_code]))
