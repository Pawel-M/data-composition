from typing import Optional, Union, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import MarianMTModel, MarianConfig, MarianPreTrainedModel, MarianModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import logging
from transformers.models.marian.modeling_marian import shift_tokens_right

logger = logging.get_logger(__name__)

class CoWordMarianConfig(MarianConfig):
    model_type = "marian"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
            self,
            vocab_size=58101,
            decoder_vocab_size=None,
            max_position_embeddings=1024,
            encoder_layers=12,
            encoder_ffn_dim=4096,
            encoder_attention_heads=16,
            decoder_layers=12,
            decoder_ffn_dim=4096,
            decoder_attention_heads=16,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            use_cache=True,
            is_encoder_decoder=True,
            activation_function="gelu",
            d_model=1024,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            decoder_start_token_id=58100,
            scale_embedding=False,
            pad_token_id=58100,
            eos_token_id=0,
            forced_eos_token_id=0,
            share_encoder_decoder_embeddings=True,
            mask_probability=0.0,
            mask_token_id=58101,
            **kwargs,
    ):
        self.mask_probability = mask_probability
        self.mask_token_id = mask_token_id

        super().__init__(
            vocab_size=vocab_size,
            decoder_vocab_size=decoder_vocab_size,
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
            eos_token_id=eos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            share_encoder_decoder_embeddings=share_encoder_decoder_embeddings,
            **kwargs,
        )


class CoWordMarianMTModel(MarianMTModel):
    def __init__(self, config: CoWordMarianConfig):
        super(MarianPreTrainedModel, self).__init__(config)
        self.model = MarianModel(config)

        target_vocab_size = config.vocab_size if config.share_encoder_decoder_embeddings else config.decoder_vocab_size
        self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
        self.lm_head = torch.nn.Linear(config.d_model, target_vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
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
    ) -> Seq2SeqLMOutput:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))

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

