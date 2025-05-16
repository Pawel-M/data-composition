from datasets import Dataset
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import RandomSampler
from transformers import LogitsProcessor, Seq2SeqTrainer, LogitsProcessorList
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.trainer import _is_peft_model
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.training_args import OptimizerNames
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_mp_enabled,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
)

# Integrations must be imported before ML frameworks:
# isort: off
# isort: on

from dataclasses import dataclass
from typing import Optional, Any, Union

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        pass
else:
    IS_XLA_FSDPV2_POST_2_2 = False

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_safetensors_available():
    pass

if is_peft_available():
    pass

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        pass

if is_accelerate_available("0.28.0"):
    pass

if is_datasets_available():
    import datasets

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import torch
from packaging import version
from torch import nn

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.utils import (
    is_datasets_available,
    is_in_notebook,
    logging,
)

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets

logger = logging.get_logger(__name__)


class BatchedForcedBOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: torch.LongTensor):
        self.bos_token_id: torch.LongTensor = bos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            scores_size = scores.shape[0]
            batch_size = self.bos_token_id.shape[0]
            beam_size = scores_size // batch_size
            assert batch_size * beam_size == scores_size

            scores[:, :] = -float("inf")
            bos_ids = self.bos_token_id.repeat_interleave(beam_size)
            scores.index_put_((torch.arange(scores_size, dtype=torch.long), bos_ids),
                              torch.tensor(0, dtype=scores.dtype))
        return scores


class WeightingM2MSeq2SeqTrainer(Seq2SeqTrainer):
    KEEP_DATASET_COLUMNS = [
        'forced_bos_token_id',
        'exp_mask',
        'source_sentences',
        'target_sentences',
        # 'phrase_src_mask',
        # 'ctx_cue_src_mask',
        # 'ctx_cue_tgt_mask',
    ]

    # def __init__(
    #         self,
    #         model: Union["PreTrainedModel", nn.Module] = None,
    #         args: "TrainingArguments" = None,
    #         data_collator: Optional["DataCollator"] = None,
    #         train_dataset: Optional[Union[Dataset, "IterableDataset", "datasets.Dataset"]] = None,
    #         eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
    #         processing_class: Optional[
    #             Union["PreTrainedTokenizerBase", "BaseImageProcessor", "FeatureExtractionMixin", "ProcessorMixin"]
    #         ] = None,
    #         model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
    #         compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
    #         callbacks: Optional[List["TrainerCallback"]] = None,
    #         optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    #         preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    #         save_attention_plots: bool = False,
    #         accumulate_gradients_fns: Optional[List[Callable]] = None,
    # ):
    #     super().__init__(
    #         model=model,
    #         args=args,
    #         data_collator=data_collator,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         processing_class=processing_class,
    #         model_init=model_init,
    #         compute_metrics=compute_metrics,
    #         callbacks=callbacks,
    #         optimizers=optimizers,
    #         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    #     )
    #
    #     self.save_attention_plots = save_attention_plots
    #     self.accumulate_gradients_fns = accumulate_gradients_fns
    #
    #     self.accumulated_mean_head_gradients = {}
    #     self.accumulated_max_head_gradients = {}
    #     self.accumulated_masked_mean_head_gradients = {}
    #     self.accumulated_masked_max_head_gradients = {}
    #     self.accumulated_scaled_mean_head_gradients = {}
    #     self.accumulated_scaled_max_head_gradients = {}
    #     self.accumulated_scaled_positive_head_gradients = {}
    #     self.accumulated_masked_scaled_mean_head_gradients = {}
    #     self.accumulated_masked_scaled_max_head_gradients = {}
    #     self.accumulated_masked_scaled_positive_head_gradients = {}
    #     self.accumulated_decoder_head_gradients = {}
    #     self.accumulated_scaled_decoder_head_gradients = {}

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # exp_mask = None
        # if "exp_mask" in inputs:
        #     exp_mask = inputs.pop("exp_mask")

        # phrase_src_mask = inputs.pop('phrase_src_mask') if 'phrase_src_mask' in inputs else None
        # ctx_cue_src_mask = inputs.pop('ctx_cue_src_mask') if 'ctx_cue_src_mask' in inputs else None
        # ctx_cue_tgt_mask = inputs.pop('ctx_cue_tgt_mask') if 'ctx_cue_tgt_mask' in inputs else None

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=True)

        for attn_type in ['cross_attentions', 'decoder_attentions', 'encoder_attentions']:
            for attn in outputs[attn_type]:
                # attn = attn.detach()
                # attn = attn.to(self.args.device)
                attn.retain_grad()

        # input_mask = inputs['attention_mask']
        # output_mask = inputs['labels'] != model.tokenizer.pad_token_id
        # TODO: Remove when finished with plotting
        # input_ids = inputs['input_ids']
        # labels = inputs['labels']
        # input_tokens = [model.tokenizer.convert_ids_to_tokens(input_ids[i]) for i in range(input_ids.size(0))]
        # output_tokens = [model.tokenizer.convert_ids_to_tokens(labels[i]) for i in range(labels.size(0))]

        del inputs
        if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

        else:
            self.accelerator.backward(loss, **kwargs)
            # Finally we need to normalize the loss for reporting
            if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps
            return loss.detach()

    # copy and pasted from Seq2SeqTrainer in my downloaded source code
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        if "forced_bos_token_id" in inputs:
            forced_bos_token_ids = inputs.pop("forced_bos_token_id")
            logits_processor = LogitsProcessorList([BatchedForcedBOSTokenLogitsProcessor(forced_bos_token_ids)])
        else:
            forced_bos_token_ids = None
            logits_processor = None

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
                "labels" in generation_inputs
                and "decoder_input_ids" in generation_inputs
                and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        with summon_full_params_context:
            generated_tokens = self.model.generate(
                **generation_inputs,
                logits_processor=logits_processor,
                **gen_kwargs
            )

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns) - set(self.KEEP_DATASET_COLUMNS))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                "No columns in the dataset match the model's forward method signature. "
                f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "
                "Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    # Remove the forced_bos_token_id as well as the inputs for the forward step
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if "forced_bos_token_id" in inputs:
            inputs.pop("forced_bos_token_id")

        # if "exp_mask" in inputs:
        #     inputs.pop("exp_mask")

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs, output_attentions=True)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


@dataclass
class WeightingDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        # 'exp_mask',
        # 'source_sentences',
        # 'target_sentences',
        # 'phrase_src_mask',
        # 'ctx_cue_src_mask',
        # 'ctx_cue_tgt_mask',
        exp_mask = [feature['exp_mask'] for feature in features] if 'exp_mask' in features[0].keys() else None
        source_sentences = [feature['source_sentences'] for feature in features] if 'source_sentences' in features[
            0].keys() else None
        target_sentences = [feature['target_sentences'] for feature in features] if 'target_sentences' in features[
            0].keys() else None
        # phrase_src_mask = [feature['phrase_src_mask'] for feature in features] if 'phrase_src_mask' in features[0].keys() else None
        # ctx_cue_src_mask = [feature['ctx_cue_src_mask'] for feature in features] if 'ctx_cue_src_mask' in features[0].keys() else None
        # ctx_cue_tgt_mask = [feature['ctx_cue_tgt_mask'] for feature in features] if 'ctx_cue_tgt_mask' in features[0].keys() else None

        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None

        exclude_additional_keys = [
            'exp_mask',
            'source_sentences',
            'target_sentences',
            'phrase_src_mask',
            'ctx_cue_src_mask',
            'ctx_cue_tgt_mask'
        ]
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        non_labels_features = [{k: v for k, v in feature.items() if k not in exclude_additional_keys} for feature in
                               non_labels_features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            batch['labels'] = self.pad_labels(labels, features, label_name, no_padding, self.label_pad_token_id)
            # if no_padding:
            #     if isinstance(features[0][label_name], list):
            #         batch["labels"] = list(labels)
            #     else:
            #         batch["labels"] = [np.concatenate([label, []]) for label in labels]
            # else:
            #     max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
            #     max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
            #     if self.pad_to_multiple_of is not None:
            #         max_label_length = (
            #             (max_label_length + self.pad_to_multiple_of - 1)
            #             // self.pad_to_multiple_of
            #             * self.pad_to_multiple_of
            #         )
            #
            #     padding_side = self.tokenizer.padding_side
            #     if isinstance(features[0][label_name], list):
            #         batch["labels"] = [
            #             label + [self.label_pad_token_id] * (max_label_length - len(label))
            #             if padding_side == "right"
            #             else [self.label_pad_token_id] * (max_label_length - len(label)) + label
            #             for label in labels
            #         ]
            #     else:
            #         batch["labels"] = [
            #             np.concatenate(
            #                 [
            #                     label,
            #                     np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
            #                 ]
            #             )
            #             if padding_side == "right"
            #             else np.concatenate(
            #                 [
            #                     np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
            #                     label,
            #                 ]
            #             )
            #             for label in labels
            #         ]

        if exp_mask is not None:
            batch['exp_mask'] = self.pad_labels(exp_mask, features, 'exp_mask', no_padding, 0)

        if source_sentences is not None:
            batch['source_sentences'] = self.pad_labels(source_sentences, features, 'source_sentences', no_padding, 0)

        if target_sentences is not None:
            batch['target_sentences'] = self.pad_labels(target_sentences, features, 'target_sentences', no_padding, 0)

        # if phrase_src_mask is not None:
        #     batch['phrase_src_mask'] = self.pad_labels(phrase_src_mask, features, 'phrase_src_mask', no_padding, 0)
        #
        # if ctx_cue_src_mask is not None:
        #     batch['ctx_cue_src_mask'] = self.pad_labels(ctx_cue_src_mask, features, 'ctx_cue_src_mask', no_padding, 0)
        #
        # if ctx_cue_tgt_mask is not None:
        #     batch['ctx_cue_tgt_mask'] = self.pad_labels(ctx_cue_tgt_mask, features, 'ctx_cue_tgt_mask', no_padding, 0)

        self.tensorize(batch, label_name, return_tensors)
        self.tensorize(batch, 'exp_mask', return_tensors)
        self.tensorize(batch, 'source_sentences', return_tensors)
        self.tensorize(batch, 'target_sentences', return_tensors)
        # self.tensorize(batch, 'phrase_src_mask', return_tensors)
        # self.tensorize(batch, 'ctx_cue_src_mask', return_tensors)
        # self.tensorize(batch, 'ctx_cue_tgt_mask', return_tensors)

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        # if batch.get("labels", None) is not None:
        #     if return_tensors == "pt":
        #         import torch
        #
        #         batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        #     elif return_tensors == "tf":
        #         import tensorflow as tf
        #
        #         batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
        #     else:
        #         batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        # else:
        #     batch["labels"] = None

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch

    def pad_labels(self, labels, features, label_name, no_padding, pad_token_id):
        if no_padding:
            if isinstance(features[0][label_name], list):
                batch_labels = list(labels)
            else:
                batch_labels = [np.concatenate([label, []]) for label in labels]
        else:
            max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
            max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            if isinstance(features[0][label_name], list):
                batch_labels = [
                    label + [pad_token_id] * (max_label_length - len(label))
                    if padding_side == "right"
                    else [pad_token_id] * (max_label_length - len(label)) + label
                    for label in labels
                ]
            else:
                batch_labels = [
                    np.concatenate(
                        [
                            label,
                            np.array([pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                        ]
                    )
                    if padding_side == "right"
                    else np.concatenate(
                        [
                            np.array([pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            label,
                        ]
                    )
                    for label in labels
                ]

        return batch_labels

    def tensorize(self, batch, label_name, return_tensors):
        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get(label_name, None) is not None:
            if return_tensors == "pt":
                import torch

                batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch[label_name] = tf.constant(batch[label_name], dtype=tf.int64)
            else:
                batch[label_name] = np.array(batch[label_name], dtype=np.int64)
        else:
            batch[label_name] = None
