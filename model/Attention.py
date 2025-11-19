import sys
import torch.nn as nn
import torch
import torch.nn.functional as F
class SelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):

        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )

        out = self.out_proj(out)

        return out

class InLineAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

        self.residual = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim * 3, kernel_size=1, groups=num_heads)
        )

        head_dim = model_dim // num_heads
        self.scale = head_dim ** -0.5

    def forward(self, x):

        B, T, N, C = x.shape
        x = x.reshape(B * T, N, C)

        num_heads = self.num_heads
        head_dim = C // num_heads

        q = self.FC_Q(x)  # 16, 12, 170, 64
        k = self.FC_K(x)
        v = self.FC_V(x)

        q = q.reshape(B * T, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * T, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B * T, N, num_heads, head_dim).permute(0, 2, 1, 3)
        res_weight = self.residual(x.mean(dim=1).unsqueeze(dim=-1)).reshape(B * T * C, 1, 3)
        kv = (k.transpose(-2, -1) * (self.scale / N) ** 0.5) @ (v * (self.scale / N) ** 0.5)
        x = q @ kv + (1 - q @ k.mean(dim=2, keepdim=True).transpose(-2, -1) * self.scale) * v.mean(dim=2, keepdim=True)
        x = x.transpose(1, 2).reshape(B * T, N, C)
        v = v.transpose(1, 2).reshape(B * T, N, C).permute(0, 2, 1).reshape(1, B * T * C, N)
        residual = F.conv1d(v, res_weight, None, padding=1, groups=B * T * C)
        residual = residual.reshape(B * T, C, N).permute(0, 2, 1)
        out = x + residual
        out = out.reshape(B, T, N, C)
        out = self.out_proj(out)

        return out

if __name__ == '__main__':

    x = torch.randn(16, 12, 170, 64)
    att = InLineAttention(64, num_heads=4)
    out = att(x)
    print(out.shape)
