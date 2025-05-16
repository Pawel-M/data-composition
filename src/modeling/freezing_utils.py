import torch


class HeadDisabler:
    def __init__(self, params, unfrozen_heads, num_heads):
        self.unfrozen_heads = unfrozen_heads
        self.num_heads = num_heads
        self.head_dim = params.shape[0] // self.num_heads

        self.mask = self.create_mask(params)

    def create_mask(self, params):
        mask = torch.zeros_like(params)
        for head in self.unfrozen_heads:
            mask[head * self.head_dim:(head + 1) * self.head_dim, ...] = 1

        return mask

    def __call__(self, grad):
        self.mask = self.mask.to(grad.device)
        grad = grad.clone()
        grad *= self.mask
        return grad