from typing import List, Optional, Union

import math
import torch
from transformers.utils import logging

from data.contrapro import ContrastiveDataPoint


def get_sentence_score(encoded_ids, logits, attention_mask=None, return_token_logprobs=False, to_cpu=True):
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected_logprobs = logprobs.gather(2, encoded_ids.unsqueeze(2)).squeeze(-1)
    if attention_mask is not None:
        selected_logprobs = selected_logprobs * attention_mask
    total_logprob = selected_logprobs.sum(1)

    if to_cpu:
        total_logprob = total_logprob.cpu()
        selected_logprobs = selected_logprobs.cpu()

    if return_token_logprobs:
        return total_logprob.detach(), torch.exp(total_logprob).detach(), selected_logprobs.detach()

    return total_logprob.detach(), torch.exp(total_logprob)


def trim_context(context: Union[List[str], List[List[str]]], context_size):
    if context is None or len(context) == 0:
        return [], 0

    batched = isinstance(context, list) and isinstance(context[0], list)

    if not batched:
        context = [context]

    effective_context = []
    for c in context:
        c = [s for s in c if s is not None and len(s) > 0]
        if c is None or len(c) == 0 or context_size is None or context_size == 0:
            c = []
        else:
            max_context = min(len(c), context_size)
            c = c[-max_context:]
        effective_context.append(c)

    if not batched:
        return effective_context[0], len(effective_context[0])

    return effective_context, [len(c) for c in effective_context]


def concatenate_current_and_context(current: Union[str, List[str]],
                                    context: Union[List[str], List[List[str]]],
                                    sep_token):
    batched = isinstance(current, list)
    if not batched:
        assert isinstance(current, str) and isinstance(context, list), \
            f'current: {isinstance(current, str)}, context: {isinstance(context, list)}'
        current = [current]
        context = [context]
    else:
        assert isinstance(context, list)
        assert len(current) == len(context) and all(isinstance(c, list) for c in context), \
            print(f'Lengths of the current and context lists do not match: {len(current)} vs {len(context)}')

    concatenated = []
    for c, cc in zip(current, context):
        if cc is None or len(cc) == 0:
            concatenated.append(c)
        else:
            if sep_token is not None:
                concatenated.append(f'{sep_token} '.join(cc + [c]))
            else:
                concatenated.append(' '.join(cc + [c]))

    if batched:
        return concatenated
    else:
        return concatenated[0]


def tokenize_with_context(tokenizer,
                          current: List[str],
                          context: List[List[str]],
                          context_size,
                          is_target: bool,
                          max_length=None):
    context, effective_context_size = trim_context(context, context_size)
    all = concatenate_current_and_context(current, context, tokenizer.sep_token)
    if is_target:
        tokenized = tokenizer(text_target=all,
                              max_length=max_length,
                              truncation=True,
                              padding=True,
                              return_token_type_ids=False,
                              return_tensors='pt', )
    else:
        tokenized = tokenizer(text=all,
                              max_length=max_length,
                              truncation=True,
                              padding=True,
                              return_token_type_ids=False,
                              return_tensors='pt', )

    return tokenized, effective_context_size


def tokenize_all_with_context(tokenizer,
                              sources, targets,
                              sources_context, targets_context,
                              source_context_size, target_context_size,
                              max_length=None,
                              return_tensors=None, ):
    src_context, sources_context_sizes = trim_context(sources_context, source_context_size)
    tgt_context, targets_context_sizes = trim_context(targets_context, target_context_size)

    sep_token = tokenizer.sep_token
    all_sources = concatenate_current_and_context(sources, src_context, sep_token)
    all_targets = concatenate_current_and_context(targets, tgt_context, sep_token)
    all_tokenized = tokenizer(text=all_sources,
                              text_target=all_targets,
                              max_length=max_length,
                              padding=False,
                              truncation=True,
                              # return_tensors='pt',
                              return_tensors=return_tensors,
                              return_token_type_ids=False, )
    return all_tokenized, sources_context_sizes, targets_context_sizes


def score_contrastive(model,
                      tokenizer,
                      device,
                      source_context_size,
                      target_context_size,
                      max_length,
                      data: List[ContrastiveDataPoint],
                      ):
    (
        tokenized_src,
        src_effective_context_size
    ) = tokenize_with_context(tokenizer,
                              current=[d.source for d in data],
                              context=[d.source_context for d in data],
                              context_size=source_context_size,
                              is_target=False,
                              max_length=max_length)
    tokenized_src = tokenized_src.to(device)

    encoder = model.get_encoder()

    encoder_out = encoder(
        input_ids=tokenized_src['input_ids'],
        attention_mask=tokenized_src['attention_mask'],
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

    decoder = model.get_decoder()
    head = model.lm_head
    head_bias = model.final_logits_bias if hasattr(model, 'final_logits_bias') else None

    num_targets = torch.tensor([len(d.targets) for d in data], device=device)
    targets_cumulated = torch.concat([torch.tensor([0], device=device), num_targets[:-1]]).cumsum(dim=0)
    (
        tokenized_tgt,
        tgt_effective_context_size
    ) = tokenize_with_context(tokenizer,
                              current=[target for d in data for target in d.targets],
                              context=[d.target_context for d in data for _ in d.targets],
                              context_size=target_context_size,
                              is_target=True,
                              max_length=max_length)
    tokenized_tgt = tokenized_tgt.to(device)

    encoded_tgt_ids = tokenized_tgt['input_ids']
    tgt_attention_mask = tokenized_tgt['attention_mask']
    encoder_last_hidden_state = encoder_out['last_hidden_state']
    encoder_attention_mask = tokenized_src['attention_mask']
    encoder_last_hidden_state = encoder_last_hidden_state.repeat_interleave(num_targets, dim=0)
    encoder_attention_mask = encoder_attention_mask.repeat_interleave(num_targets, dim=0)

    pad_token_id = torch.tensor([[model.config.decoder_start_token_id]], device=device)
    input_tgt_ids = torch.concat((pad_token_id.repeat(encoded_tgt_ids.size(0), 1), encoded_tgt_ids[:, :-1]), dim=-1)
    decoder_outputs = decoder(
        input_ids=input_tgt_ids,
        attention_mask=tgt_attention_mask,
        encoder_hidden_states=encoder_last_hidden_state,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )
    decoder_hidden_state = decoder_outputs['last_hidden_state']

    logits = head(decoder_hidden_state)
    if head_bias is not None:
        logits += head_bias

    total_logprob, total_prob = get_sentence_score(encoded_tgt_ids, logits, tgt_attention_mask)

    tokens = []
    total_logprobs = []
    for i, d in enumerate(data):
        tokens.append([
            tokenizer.convert_ids_to_tokens(tokenized_src['input_ids'][i], skip_special_tokens=True),
            [tokenizer.convert_ids_to_tokens(encoded_tgt_ids[targets_cumulated[i] + k], skip_special_tokens=True)
             for k in range(num_targets[i])]
        ])
        total_logprobs.append([total_logprob[targets_cumulated[i] + k] for k in range(num_targets[i])])

    additional_data = None
    return_values = (
        total_logprobs,
        tokens,
        additional_data,
    )
    return return_values


def score_contextual(
        model,
        tokenizer,
        device,
        source_context_size,
        target_context_size,
        max_length,
        sources,
        targets,
        source_contexts,
        target_contexts,
        output_attentions=False,
):
    (
        tokenized_src,
        src_effective_context_size
    ) = tokenize_with_context(tokenizer,
                              current=sources,
                              context=source_contexts,
                              context_size=source_context_size,
                              is_target=False,
                              max_length=max_length)
    tokenized_src = tokenized_src.to(device)

    encoder = model.get_encoder()

    encoder_out = encoder(
        input_ids=tokenized_src['input_ids'],
        attention_mask=tokenized_src['attention_mask'],
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=True,
    )

    decoder = model.get_decoder()
    head = model.lm_head
    head_bias = model.final_logits_bias if hasattr(model, 'final_logits_bias') else None

    # num_targets = torch.tensor([len(d.targets) for d in data], device=device)
    # targets_cumulated = torch.concat([torch.tensor([0], device=device), num_targets[:-1]]).cumsum(dim=0)
    (
        tokenized_tgt,
        tgt_effective_context_size
    ) = tokenize_with_context(tokenizer,
                              current=targets,
                              context=target_contexts,
                              context_size=target_context_size,
                              is_target=True,
                              max_length=max_length)
    tokenized_tgt = tokenized_tgt.to(device)

    encoded_tgt_ids = tokenized_tgt['input_ids']
    tgt_attention_mask = tokenized_tgt['attention_mask']
    encoder_last_hidden_state = encoder_out['last_hidden_state']
    encoder_attention_mask = tokenized_src['attention_mask']

    pad_token_id = torch.tensor([[model.config.decoder_start_token_id]], device=device)
    input_tgt_ids = torch.concat((pad_token_id.repeat(encoded_tgt_ids.size(0), 1), encoded_tgt_ids[:, :-1]), dim=-1)
    decoder_outputs = decoder(
        input_ids=input_tgt_ids,
        attention_mask=tgt_attention_mask,
        encoder_hidden_states=encoder_last_hidden_state,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=True,
    )
    decoder_hidden_state = decoder_outputs['last_hidden_state']

    logits = head(decoder_hidden_state)
    if head_bias is not None:
        logits += head_bias

    total_logprob, total_prob, token_logprob = get_sentence_score(
        encoded_ids=encoded_tgt_ids,
        logits=logits,
        attention_mask=tgt_attention_mask,
        return_token_logprobs=True,
        to_cpu=False,
    )

    # tokens = []
    # total_logprobs = []
    # for i, d in enumerate(data):
    #     tokens.append([
    #         tokenizer.convert_ids_to_tokens(tokenized_src['input_ids'][i], skip_special_tokens=True),
    #         [tokenizer.convert_ids_to_tokens(encoded_tgt_ids[targets_cumulated[i] + k], skip_special_tokens=True)
    #          for k in range(num_targets[i])]
    #     ])
    #     total_logprobs.append([total_logprob[targets_cumulated[i] + k] for k in range(num_targets[i])])

    # additional_data = None
    # return_values = (
    #     total_logprobs,
    #     tokens,
    #     additional_data,
    # )

    if output_attentions:
        attentions = {
            'encoder': encoder_out['attentions'],
            'decoder': decoder_outputs['attentions'],
            'cross': decoder_outputs['cross_attentions'],
        }
        return token_logprob, encoded_tgt_ids, tgt_attention_mask, attentions
    return token_logprob, encoded_tgt_ids, tgt_attention_mask
