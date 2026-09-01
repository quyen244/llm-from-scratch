import torch 
import torch.nn as nn
from src.modules.normalization import RMSNorm
from src.modules.attention import Attention

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, block_size: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)  # sketch gốc chỉ có 1 norm dùng chung 2 lần -> sai
        self.norm2 = RMSNorm(d_model)
        self.attention = Attention(d_model=d_model, num_heads=num_heads, block_size=block_size)

        hidden = d_model * 4  # sketch gốc không mở rộng chiều ẩn của FFN
        self.fnn1 = nn.Linear(in_features=d_model, out_features=hidden)
        self.fnn2 = nn.Linear(in_features=hidden, out_features=d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual connection: sketch gốc THIẾU hoàn toàn cả 2 residual này
        x = x + self.attention(self.norm1(x))
        x = x + self.fnn2(self.act(self.fnn1(self.norm2(x))))
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, block_size: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)  # sketch gốc chỉ có 1 norm dùng chung 2 lần -> sai
        self.norm2 = RMSNorm(d_model)
        self.attention = Attention(d_model=d_model, num_heads=num_heads, block_size=block_size)

        hidden = d_model * 4  # sketch gốc không mở rộng chiều ẩn của FFN
        self.fnn1 = nn.Linear(in_features=d_model, out_features=hidden)
        self.fnn2 = nn.Linear(in_features=hidden, out_features=d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual connection: sketch gốc THIẾU hoàn toàn cả 2 residual này
        x = x + self.attention(self.norm1(x))
        x = x + self.fnn2(self.act(self.fnn1(self.norm2(x))))
        return x

