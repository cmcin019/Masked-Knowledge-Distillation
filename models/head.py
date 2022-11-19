
import torch
import torch.nn as nn
import utils

from utils import trunc_normal_

class DINOHead(nn.Module):
    def __init__(self, 
                in_dim, 
                out_dim, 
                use_bn=False, 
                norm_last_layer=True, 
                nlayers=3, 
                hidden_dim=2048, 
                bottleneck_dim=256, 
                **kwargs):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        self.last_norm = nn.BatchNorm1d(out_dim, affine=False, **kwargs)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class iBOTHead(DINOHead):

    def __init__(self, *args, patch_out_dim=8192, norm=None, act='gelu', last_norm=None, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, 
                 shared_head=False, **kwargs):
        
        super(iBOTHead, self).__init__(*args, **kwargs)

        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
                self.last_layer2.weight_g.data.fill_(1)
                if norm_last_layer:
                    self.last_layer2.weight_g.requires_grad = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None

            self.last_norm2 = nn.BatchNorm1d(patch_out_dim, affine=False, **kwargs)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None

            self.last_norm2 = self.last_norm

    def forward(self, x):
        if len(x.shape) == 2:
            return super(iBOTHead, self).forward(x)

        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])
        
        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)
        
        return x1, x2


class MRKDHead(DINOHead):
    def __init__(self, *args, patch_out_dim=8192, norm=None, act='gelu', last_norm=None, 
                nlayers=3, hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True, 
                shared_head=False, **kwargs):
        
        super(MRKDHead, self).__init__(*args,
                                    norm=norm,
                                    act=act,
                                    last_norm=last_norm,
                                    nlayers=nlayers,
                                    hidden_dim=hidden_dim,
                                    bottleneck_dim=bottleneck_dim,
                                    norm_last_layer=norm_last_layer, 
                                    **kwargs)
        pass

    def forward(self, x):
        return 0