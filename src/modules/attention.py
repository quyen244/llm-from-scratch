import torch 
import torch.nn as nn 



class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, block_size: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(in_features=d_model, out_features=d_model * 3)
        self.out_proj = nn.Linear(in_features=d_model, out_features=d_model)  # sketch gốc THIẾU lớp này

        # causal mask cố định, không học -> đăng ký buffer để tự chuyển device theo model
        mask = torch.tril(torch.ones(block_size, block_size)).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # mỗi cái (B, T, d_model)

        # tách head ĐÚNG cách: reshape rồi transpose, không reshape thẳng
        # (B,T,d_model) -> (B,T,num_heads,head_dim) -> (B,num_heads,T,head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B,num_heads,T,T)
        # causal mask: vị trí t không được nhìn thấy vị trí > t -> ĐÂY LÀ PHẦN SKETCH GỐC THIẾU HOÀN TOÀN
        attn_scores = attn_scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = attn_weights @ v  # (B,num_heads,T,head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)  # gộp head lại đúng thứ tự
        return self.out_proj(out)
