import torch
from torch import nn


class PositionalEncodingsFixed(nn.Module):

    def __init__(self, emb_dim, temperature=10000):

        super(PositionalEncodingsFixed, self).__init__()

        self.emb_dim = emb_dim
        self.temperature = temperature

    def _1d_pos_enc(self, mask, dim):
        temp = torch.arange(self.emb_dim // 2).float().to(mask.device)
        temp = self.temperature ** (2 * (temp.div(2, rounding_mode='floor')) / self.emb_dim)

        enc = (~mask).cumsum(dim).float().unsqueeze(-1) / temp
        enc = torch.stack([
            enc[..., 0::2].sin(), enc[..., 1::2].cos()
        ], dim=-1).flatten(-2)

        return enc

    def forward(self, bs, h, w, device):
        mask = torch.zeros(bs, h, w, dtype=torch.bool, requires_grad=False, device=device)
        x = self._1d_pos_enc(mask, dim=2)
        y = self._1d_pos_enc(mask, dim=1)

        return torch.cat([y, x], dim=3).permute(0, 3, 1, 2)


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: nn.Module
    ):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x):
        return (
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
    ):

        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(self, src, pos_emb, src_mask, src_key_padding_mask):
        output = src
        for layer in self.layers:
            output = layer(output, pos_emb, src_mask, src_key_padding_mask)
        return self.norm(output)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout
        )
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(self, src, pos_emb, src_mask, src_key_padding_mask):
        if self.norm_first:
            src_norm = self.norm1(src)
            q = k = src_norm + pos_emb
            src = src + self.dropout1(self.self_attn(
                query=q,
                key=k,
                value=src_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )[0])

            src_norm = self.norm2(src)
            src = src + self.dropout2(self.mlp(src_norm))
        else:
            q = k = src + pos_emb
            src = self.norm1(src + self.dropout1(self.self_attn(
                query=q,
                key=k,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )[0]))

            src = self.norm2(src + self.dropout2(self.mlp(src)))

        return src
    
    
class UpsamplingLayer(nn.Module):

    def __init__(self, in_channels, out_channels, leaky=True):

        super(UpsamplingLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU() if leaky else nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x):
        return self.layer(x)


class DensityMapRegressor(nn.Module):

    def __init__(self, in_channels, reduction):

        super(DensityMapRegressor, self).__init__()

        if reduction == 8:
            self.regressor = nn.Sequential(
                UpsamplingLayer(in_channels, 128),
                UpsamplingLayer(128, 64),
                UpsamplingLayer(64, 32),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.LeakyReLU()
            )
        elif reduction == 16:
            self.regressor = nn.Sequential(
                UpsamplingLayer(in_channels, 128),
                UpsamplingLayer(128, 64),
                UpsamplingLayer(64, 32),
                UpsamplingLayer(32, 16),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.LeakyReLU()
            )

        self.reset_parameters()

    def forward(self, x):
        return self.regressor(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)