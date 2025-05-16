import re
from collections import defaultdict

import torch
from torch import nn
from transformers import M2M100ForConditionalGeneration, M2M100Model, M2M100PreTrainedModel
from transformers.models.m2m_100.configuration_m2m_100 import M2M100Config
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)

from modeling.freezing_utils import HeadDisabler

if is_flash_attn_2_available():
    pass

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "M2M100Config"
_CHECKPOINT_FOR_DOC = "facebook/m2m100_418M"

_ATTN_TYPE_TO_MODULES = {
    'encoder': ('encoder', 'self_attn'),
    'decoder': ('decoder', 'self_attn'),
    'cross': ('decoder', 'encoder_attn'),
}

_PROJECTION_MAP = {
    'q': 'q_proj',
    'k': 'k_proj',
    'v': 'v_proj',
    'out': 'out_proj',
}


# class GradientMasker:
#     def __init__(self, mask):
#         self.mask = mask
#
#     def __call__(self, grad):
#         grad = grad.clone()
#         grad *= self.mask
#         return grad


class FreezingM2M100Model(M2M100Model):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: M2M100Config):
        super().__init__(config)
        self._unfreeze_patterns = None
        self._head_disablers_handles = []
        # self._heads_to_unfreeze = None

    def freeze_model(self, config):
        self._reset_head_disablers()
        unfreeze_patterns = []

        if hasattr(config, 'unfreeze_layers'):
            for part, layers in config.unfreeze_layers.items():
                for layer in layers:
                    unfreeze_patterns.append(rf'{part}.layers.{layer}.*')

        if hasattr(config, 'unfreeze_heads'):
            for attn_type, layers in config.unfreeze_heads.items():
                part, module = _ATTN_TYPE_TO_MODULES[attn_type]
                part_module = self.encoder if attn_type == 'encoder' else self.decoder
                for layer_config in layers:
                    layer = layer_config['layer']
                    layer_module = part_module.layers[layer]
                    attn_module = layer_module.encoder_attn if attn_type == 'cross' else layer_module.self_attn
                    if 'projections' in layer_config:
                        for projection in layer_config['projections']:
                            proj = _PROJECTION_MAP[projection]

                            unfreeze_patterns.append(rf'{part}.layers.{layer}.{module}.{proj}.*')

                            if 'heads' in layer_config \
                                    and len(layer_config['heads']) > 0 \
                                    and projection in ('q', 'k', 'v'):
                                if projection == 'q':
                                    proj_module = attn_module.q_proj
                                elif projection == 'k':
                                    proj_module = attn_module.k_proj
                                elif projection == 'v':
                                    proj_module = attn_module.v_proj

                                self._apply_head_disabler(proj_module, layer_config['heads'], attn_module.num_heads)
                                # heads_to_unfreeze[(attn_type, layer, projection)].extend(layer_config['heads'])

                    else:
                        unfreeze_patterns.append(rf'{part}.layers.{layer}.{module}.*')
                        if 'heads' in layer_config:
                            raise ValueError('Heads can not be unfrozen without specifying projections.')

        if hasattr(config, 'unfreeze_mlps'):
            for part, layers in config.unfreeze_mlps.items():
                for layer in layers:
                    unfreeze_patterns.append(rf'{part}.layers.{layer}.fc*')

        self._unfreeze_patterns = unfreeze_patterns
        # self._heads_to_unfreeze = heads_to_unfreeze
        self._freeze_params(unfreeze_patterns)

    def _reset_head_disablers(self):
        if self._head_disablers_handles is None:
            self._head_disablers_handles = []
            return

        for head_disabler_handler in self._head_disablers_handles:
            head_disabler_handler.remove()

        self._head_disablers_handles = []

    def _apply_head_disabler(self, linear, unfrozen_heads, num_heads):
        weight_disabler_handler = linear.weight.register_hook(
            HeadDisabler(
                linear.weight,
                unfrozen_heads,
                num_heads
            )
        )

        bias_disabler_handler = linear.bias.register_hook(
            HeadDisabler(
                linear.bias,
                unfrozen_heads,
                num_heads
            )
        )

        self._head_disablers_handles.append(weight_disabler_handler)
        self._head_disablers_handles.append(bias_disabler_handler)

    def _freeze_params(self, unfreeze_patterns):
        print('Freezing model...')
        print('Keeping unfrozen:')
        # for pattern in unfreeze_patterns:
        #     print(pattern)

        for name, param in self.named_parameters():
            unfreeze = False
            for patter in unfreeze_patterns:
                match_q = re.match(patter, name)
                if match_q:
                    unfreeze = True
                    break

            param.requires_grad = unfreeze
            if unfreeze:
                print(name)


class FreezingM2M100ForConditionalGeneration(M2M100ForConditionalGeneration):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: M2M100Config):
        super(M2M100PreTrainedModel, self).__init__(config)
        self.model = FreezingM2M100Model(config)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_model(self, config):
        self.model.freeze_model(config)

    def post_backward_pass(self):
        self.model.post_backward_pass()
