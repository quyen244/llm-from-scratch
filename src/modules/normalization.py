import torch 
import torch.nn as nn


class RMSNorm(nn.Module):
    """Tự viết thay vì phụ thuộc nn.RMSNorm (chỉ có từ torch bản khá mới)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


    