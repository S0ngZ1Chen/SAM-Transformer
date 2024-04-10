import torch
from torch import nn

from model_utils import PositionalEncodingsFixed, TransformerEncoder, DensityMapRegressor

class Sam_Head(nn.Module):
    def __init__(self, emb_dim=256, num_encoder_layers=3, num_heads=8, dropout=0.1, layer_norm_eps=1e-5, mlp_factor=8, norm_first=False, activation=nn.GELU, norm=True, reduction=8):
        super(Sam_Head, self).__init__()
        self.emb_dim = emb_dim
        self.num_encoder_layers = num_encoder_layers

        self.pos_emb = PositionalEncodingsFixed(self.emb_dim)

        if self.num_encoder_layers > 0:
            self.encoder = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
    
    def forward(self, x):
        bs, c, h, w = x.size()
        pos_emb = self.pos_emb(bs, h, w, x.device).flatten(2).permute(2, 0, 1)
        x = x.flatten(2).permute(2, 0, 1)

        if self.num_encoder_layers > 0:
            x = self.encoder(x, pos_emb, src_key_padding_mask=None, src_mask=None)
        else:
            x = x

        x = x.permute(1, 2, 0).reshape(-1, self.emb_dim, h, w)
        x = self.regression_head(x)
        return x
    
if __name__=='__main__':
    model = Sam_Head().cuda()
    input = torch.Tensor(1,256,64,64).cuda()
    output = model(input)
    print(output.shape)

    #print(model)
    num = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (num / 1e6))