from typing import Iterator, Optional

import torch
from transformers import Trainer


class CompletionLossTrainer(Trainer):
    def get_batch_samples(
        self, epoch_iterator: Iterator, num_batches: int, device: torch.device
    ) -> tuple[list, Optional[torch.Tensor]]:
        """
        Collects a specified number of batches from the epoch iterator and optionally counts the number of items in the batches to properly scale the loss.
        """
        batch_samples = []
        num_items_in_batch = None

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "labels" in batch_samples[0]
            and (
                # num_items_in_batch is passed to model forward
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                self.model_accepts_loss_kwargs
                # num_items_in_batch is passed to compute_loss_func
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                or self.compute_loss_func is not None
                # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
            )
        )

        if count_num_items_in_batch:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
                num_prompt_tokens = sum(
                    [(batch["token_type_ids"].ne(0)).sum() for batch in batch_samples]
                )
                num_items_in_batch = num_items_in_batch - num_prompt_tokens
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    num_items_in_batch = num_items_in_batch.unsqueeze(0)

        return batch_samples, num_items_in_batch