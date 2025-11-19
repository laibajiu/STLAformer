import sys
import torch.nn as nn
import torch
from torchinfo import summary
from CGAFusion import CGAFusion
from NodePosition import NodePositionalEncoding, TimePositionalEncoding
from Attention import SelfAttention, InLineAttention
class STLAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()
        self.attns = InLineAttention(model_dim, num_heads)
        self.attnt = InLineAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.lnt = nn.LayerNorm(model_dim)
        self.lns = nn.LayerNorm(model_dim)
        self.ln = nn.LayerNorm(model_dim)
        self.dropoutt = nn.Dropout(dropout)
        self.dropouts = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.cgf = CGAFusion(model_dim)

    def forward(self, x, node_emb, time_emb):
        xs = (x + node_emb).transpose(2, -2)
        xt = (x + time_emb).transpose(1, -2)

        residuals = xs
        residualt = xt

        outt = self.attnt(xt)
        outs = self.attns(xs)

        outs = self.dropouts(outs)
        outt = self.dropoutt(outt)
        outs = self.lns(residuals + outs)
        outt = self.lnt(residualt + outt)

        outs = outs.transpose(2, -2)
        outt = outt.transpose(1, -2)

        out = self.cgf(outs, outt)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = self.ln(residual + out)

        return out

class STLformer(nn.Module):
    def __init__(
            self, num_nodes, in_steps=12, out_steps=12, steps_per_week=2016, input_dim=2, output_dim=1,
            input_embedding_dim=32, tow_embedding_dim=32, feed_forward_dim=256,
            num_heads=4, num_layers=3, dropout=0.3
    ):  # dropout=0.1
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_week = steps_per_week
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tow_embedding_dim = tow_embedding_dim
        self.model_dim = (
                input_embedding_dim +
                tow_embedding_dim
        )
        self.node_position_dims = self.model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_proj = nn.Linear(self.input_dim, input_embedding_dim)
        if tow_embedding_dim > 0:
            self.tow_embedding = nn.Embedding(steps_per_week, tow_embedding_dim)
        self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)

        self.attn_layers_st = nn.ModuleList(
            [
                STLAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout, False)
                for _ in range(num_layers)
            ]
        )

        self.node_position_encoding = NodePositionalEncoding(self.node_position_dims)
        self.time_position_encoding = TimePositionalEncoding(self.node_position_dims)

        self.MLP = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_dim, self.model_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        if self.tow_embedding_dim > 0:  # 24
            tow = x[..., 1]
        x = x[..., :self.input_dim]
        x = self.input_proj(x)
        features = [x]

        if self.tow_embedding_dim > 0:
            tow_emb = self.tow_embedding((tow * self.steps_per_week).long())
            features.append(tow_emb)
        x = torch.cat(features, dim=-1)

        node_emb = self.node_position_encoding(x)
        time_emb = self.time_position_encoding(x)

        bx = self.MLP(x)
        x = x - bx
        for attn in self.attn_layers_st:
            x = attn(x, node_emb, time_emb)
        x = x + bx

        out = x.transpose(1, 2)
        out = out.reshape(
            batch_size, self.num_nodes, self.in_steps * self.model_dim
        )
        out = self.output_proj(out).view(
            batch_size, self.num_nodes, self.out_steps, self.output_dim
        )
        out = out.transpose(1, 2)

        return out
