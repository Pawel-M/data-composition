from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM, KwargsForCausalLM
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple


def fixed_cross_entropy(
        source: torch.Tensor,
        target: torch.Tensor,
        prompt_mask: torch.Tensor,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        **kwargs,
) -> torch.Tensor:
    # reduction = "sum" if num_items_in_batch is not None else "mean"
    reduction = "none" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    loss = loss * (1 - prompt_mask.float())
    if reduction == "none":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss.sum()
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
        logits,
        labels,
        prompt_mask,
        vocab_size: int,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        prompt_mask = nn.functional.pad(prompt_mask, (0, 1), value=0)
        shift_labels = labels[..., 1:].contiguous()
        prompt_mask = prompt_mask[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    prompt_mask = prompt_mask.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    prompt_mask = prompt_mask.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, prompt_mask, num_items_in_batch, ignore_index, **kwargs)
    return loss



class CompletionLossLlamaForCausalLM(LlamaForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(
                logits=logits,
                labels=labels,
                prompt_mask=token_type_ids,
                vocab_size=self.config.vocab_size,
                **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
