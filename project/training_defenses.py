"""Masked PGD examples for the fixed adversarial-training objective."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def pgd_training_examples(model, clean, labels, *, epsilon=8 / 255,
                          steps=7, step_size=2 / 255):
    """Return the last iterate of one random-start masked PGD trajectory.

    The inner objective is unweighted cross-entropy. Its per-example gradient
    sign equals that of positive class-weighted cross-entropy for this CNN,
    which has no batch normalization or other interaction between examples.
    Only input gradients are requested; parameter .grad fields are untouched.
    The caller applies the weighted clean/adversarial outer objective.
    """
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not math.isfinite(step_size) or step_size <= 0 or steps < 1:
        raise ValueError("steps and step_size must be positive and finite")
    clean = clean.detach()
    if epsilon == 0:
        return clean.clone()
    mask = clean > 0
    with torch.no_grad():
        noise = torch.empty_like(clean).uniform_(-epsilon, epsilon)
        current = (clean + noise * mask).clamp(0, 1)
    for _ in range(steps):
        current = current.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(current), labels, reduction="sum")
            gradient, = torch.autograd.grad(loss, current)
        with torch.no_grad():
            delta = (current + step_size * gradient.sign() - clean).clamp(-epsilon, epsilon)
            current = (clean + delta * mask).clamp(0, 1)
    return current.detach()
