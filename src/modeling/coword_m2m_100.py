import re
from collections import defaultdict
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import M2M100ForConditionalGeneration, M2M100Model, M2M100PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.m2m_100.configuration_m2m_100 import M2M100Config
from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)
from torch.nn import CrossEntropyLoss

from modeling.freezing_utils import HeadDisabler

if is_flash_attn_2_available():
    pass

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "M2M100Config"
_CHECKPOINT_FOR_DOC = "facebook/m2m100_418M"


class CoWordM2M100Config(M2M100Config):
    model_type = "m2m_100"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
            self,
            vocab_size=128112,
            max_position_embeddings=1024,
            encoder_layers=12,
            encoder_ffn_dim=4096,
            encoder_attention_heads=16,
            decoder_layers=12,
            decoder_ffn_dim=4096,
            decoder_attention_heads=16,
            encoder_layerdrop=0.05,
            decoder_layerdrop=0.05,
            use_cache=True,
            is_encoder_decoder=True,
            activation_function="relu",
            d_model=1024,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            init_std=0.02,
            decoder_start_token_id=2,
            scale_embedding=True,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            mask_probability=0.0,
            mask_token_id=58101,
            **kwargs,
    ):
        self.mask_probability = mask_probability
        self.mask_token_id = mask_token_id

        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            encoder_layerdrop=encoder_layerdrop,
            decoder_layerdrop=decoder_layerdrop,
            use_cache=use_cache,
            is_encoder_decoder=is_encoder_decoder,
            activation_function=activation_function,
            d_model=d_model,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            init_std=init_std,
            decoder_start_token_id=decoder_start_token_id,
            scale_embedding=scale_embedding,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            mask_probability=mask_probability,
            mask_token_id=mask_token_id,
            **kwargs,
        )


class CoWordM2M100ForConditionalGeneration(M2M100ForConditionalGeneration):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: CoWordM2M100Config):
        super().__init__(config)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            exp_mask: Optional[torch.LongTensor] = None,
            source_sentences: Optional[torch.LongTensor] = None,
            target_sentences: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if self.training and self.config.mask_probability > 0.0:
            # Apply mask to input_ids based on mask_probability only to elements where source_sentences are 1
            mask = torch.rand(input_ids.shape, device=source_sentences.device) < self.config.mask_probability
            # apply only to the current sentence which is marked by the maximum value in source_sentences (for each row)
            mask = mask * (source_sentences == source_sentences.max(dim=1, keepdim=True)[0])
            # mask = mask * (source_sentences == 1)
            # put config.mask_token_id in the masked positions
            input_ids[mask] = self.config.mask_token_id

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            # move labels to the correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
