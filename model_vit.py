# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License

import torch
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import SqueezeExcite

specification = {
    'ExtremelyFastViT_M0': { # 3M
        'C': '64_128_192', 'D': 16, 'N': '4_4_4', 'X': '1_2_3', 'A': '1_2_3','drop_path': 0,
        'weights': 'None'},
    'ExtremelyFastViT_M1': { # S2
        'C': '64_144_192', 'D': 16, 'N': '4_3_4', 'X': '1_2_3', 'A': '1_3_3','drop_path': 0,
        'weights': 'None'},
    'ExtremelyFastViT_M2': { # 4M
        'C': '128_192_224', 'D': 16, 'N': '4_3_2', 'X': '1_2_3', 'A': '2_4_7','drop_path': 0,
        'weights': 'None'},
    # 'ExtremelyFastViT_M2': { # 4M
    #     'C': '128_144_192', 'D': 16, 'N': '4_3_4', 'X': '1_2_3', 'A': '2_3_3','drop_path': 0,
    #     'weights': 'None'},
    'ExtremelyFastViT_M3': { # 7M
        'C': '128_240_320', 'D': 16, 'N': '4_3_4', 'X': '1_2_3', 'A': '2_5_5','drop_path': 0,
        'weights': 'None'},
    'ExtremelyFastViT_M4': { # 9M
        'C': '128_256_384', 'D': 16, 'N': '4_4_4', 'X': '1_2_3', 'A': '2_4_6','drop_path': 0,
        'weights': 'None'},
    'ExtremelyFastViT_M5': { # 13M
        'C': '192_288_384', 'D': 16, 'N': '3_3_4', 'X': '1_3_4', 'A': '4_6_6','drop_path': 0,
        'weights': 'None'},
#     'ExtremelyFastViT_L': {
#         'C': '192_288_384', 'D': 32, 'N': '3_3_4', 'X': '1_3_4', 'E': '4_2_2_3','drop_path': 0,
#         'weights': 'None'},
} # v * (nh) = dim


@register_model
def ExtremelyFastViT_M0(num_classes=1000, distillation=False, # wd: 0.025; clip: 0.02
                pretrained=False, fuse=False):
    return model_factory(**specification['ExtremelyFastViT_M0'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)

@register_model
def ExtremelyFastViT_M1(num_classes=1000, distillation=False,
                pretrained=False, fuse=False):
    return model_factory(**specification['ExtremelyFastViT_M1'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)

@register_model
def ExtremelyFastViT_M2(num_classes=1000, distillation=False,
                pretrained=False, fuse=False):
    return model_factory(**specification['ExtremelyFastViT_M2'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)

@register_model
def ExtremelyFastViT_M3(num_classes=1000, distillation=False, # wd: 0.2; clip: 0.002
                pretrained=False, fuse=False):
    return model_factory(**specification['ExtremelyFastViT_M3'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)

@register_model
def ExtremelyFastViT_M4(num_classes=1000, distillation=False, # wd: 0.01; clipgrad: 0.02
                pretrained=False, fuse=False):
    return model_factory(**specification['ExtremelyFastViT_M4'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)

@register_model
def ExtremelyFastViT_M5(num_classes=1000, distillation=False, # wd: 0.025; clip: 0.01 # 0.1 0.02
                pretrained=False, fuse=False): # 55 onnx, 13.0M, 569M
    return model_factory(**specification['ExtremelyFastViT_M5'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)

@register_model
def ExtremelyFastViT_M5_384(num_classes=1000, distillation=False, # wd: 0.025; clip: 0.01 # 0.1 0.02
                pretrained=False, fuse=False): # 55 onnx, 13.0M, 569M
    return model_factory(**specification['ExtremelyFastViT_M5'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         image_size=384, window_size=[12, 12, 12])

FLOPS_COUNTER = 0

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1)**2
        FLOPS_COUNTER += a * b * output_points * (ks**2) // groups

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution, activation=torch.nn.Hardswish):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
            
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.se(x)
        x = self.conv3(x)
        return x


def b16(n, activation, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution**2
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)
       

class MLP(torch.nn.Module):
    def __init__(self, ed, h, resolution,
                 mlp_activation=torch.nn.ReLU):
        super().__init__()
        
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = mlp_activation()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class CascadeAttention_2d(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        # self.qkv = Conv2d_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        qkvs = []
        dws = []
        ks = [7, 5, 3, 3, 3, 3, 3, 3]
        dw_dim = self.key_dim
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(torch.nn.Sequential(Conv2d_BN(dw_dim, dw_dim, ks[i], 1, ks[i]//2, groups=dw_dim, resolution=resolution)))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)


        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * (resolution**4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**4)
        #attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution**4)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        spx = x.chunk(len(self.qkvs), dim=1)
        spo = []
        sp = spx[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:
                sp = sp + spx[i]
            sp = qkv(sp)
            q, k, v = sp.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
            # B, C/h, H, W
            q = self.dws[i](q)
            q = q.flatten(2)
            k = k.flatten(2)
            v = v.flatten(2)
            # import ipdb; ipdb.set_trace()
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            # BNN
            attn = attn.softmax(dim=-1)
            # BCHW
            sp = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)
            spo.append(sp)
        x = torch.cat(spo, 1)
        x = self.proj(x)
        return x


class BasicBlock(torch.nn.Module):
    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 activation=None,
                 resolution=14,
                 window_resolution=7,
                 drop_path=0.):
        super().__init__()
            
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution), drop_path)
        self.ffn0 = Residual(MLP(ed, int(ed * 2), resolution, activation), drop_path)

        self.type = type
        if type == 's':
            self.mixer = Residual(SwinAttention(ed, kd, nh, attn_ratio=ar, \
                    activation=activation, resolution=resolution, window_resolution=window_resolution), drop_path)
        else:
            raise NotImplementedError
                
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution), drop_path)
        self.ffn1 = Residual(MLP(ed, int(ed * 2), resolution, activation), drop_path)

    def forward(self, x):
        x = self.dw0(x)
        x = self.ffn0(x)
        x = self.mixer(x)
        x = self.dw1(x)
        x = self.ffn1(x)
        return x


class SwinAttention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=torch.nn.Hardswish,
                 resolution=14,
                 window_resolution=7,
                 subsample=False,
                 out_dim=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        # assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.subsample = subsample
        self.out_dim = out_dim
        
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadeAttention_2d(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio, activation=activation, resolution=window_resolution)

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))
               
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.permute(0, 3, 1, 2)

        return x


class FastViT(torch.nn.Module):
    """ Fast Vision Transformer
    """

    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 window_size=[7],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 distillation=True,
                 drop_path=0,):
        super().__init__()
        global FLOPS_COUNTER

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation

        self.patch_embed = hybrid_backbone

        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        
        resolution = img_size // patch_size
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                if i == 0:
                    self.blocks1.append(
                        BasicBlock(stg, ed, kd, nh, ar, \
                            activation=attention_activation, resolution=resolution, window_resolution=wd, drop_path=drop_path))
                if i == 1:
                    self.blocks2.append(
                        BasicBlock(stg, ed, kd, nh, ar, \
                            activation=attention_activation, resolution=resolution, window_resolution=wd, drop_path=drop_path))
                if i == 2:
                    self.blocks3.append(
                        BasicBlock(stg, ed, kd, nh, ar, \
                            activation=attention_activation, resolution=resolution, window_resolution=wd, drop_path=drop_path))
            if do[0] == 'Subsample':
                #('Subsample' stride)
                if i == 0:
                    blk = self.blocks2
                elif i == 1:
                    blk = self.blocks3
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(
                                    Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution), drop_path),
                                    Residual(MLP(embed_dim[i], int(embed_dim[i] * 2), resolution, mlp_activation), drop_path),))
                blk.append(
                    PatchMerging(*embed_dim[i:i + 2], resolution, activation=do[2]))
                resolution = resolution_
                blk.append(torch.nn.Sequential(
                                    Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution), drop_path),
                                    Residual(MLP(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution, mlp_activation), drop_path),))

        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        
        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


def model_factory(C, D, X, N, A, drop_path, weights,
                  num_classes, distillation, pretrained, fuse, 
                  act=None, 
                  image_size=224,
                  window_size=[7,7,7]):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    attn_ratio = [int(x) for x in A.split('_')]
    act = torch.nn.ReLU if act == None else act
    model = FastViT(
        img_size=image_size,
        patch_size=16,
        stages=['s', 's', 's'],
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=attn_ratio,
        window_size=window_size,
        down_ops=[
            #('Subsample', stride)
            ['Subsample', 2, act], ['Subsample', 2, act], ['']
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            weights, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    # if fuse:
    #     utils.replace_batchnorm(model)

    return model


def profile(model, dummy_input):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/fast_vitm5'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        import time
        # torch.cuda.synchronize()
        _start = time.time()
        runtimes = 15
        with torch.no_grad():
            for runtime in range(runtimes):
                out=model(dummy_input)
                prof.step()
        # torch.cuda.synchronize()
        _end = time.time()
        avg_time = (_end - _start) * 1000 / runtimes
        print('Runtime: {} ms'.format(avg_time))

def measure(model, dummy_input):
    import time
    # torch.cuda.synchronize()
    _start = time.time()
    runtimes = 15
    with torch.no_grad():
        for runtime in range(runtimes):
            out=model(dummy_input)
    # torch.cuda.synchronize()
    _end = time.time()
    avg_time = (_end - _start) * 1000 / runtimes
    print('Runtime: {} ms'.format(avg_time))
    
def compute_throughput_cpu(model, dummy_input):
    # inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    import time
    T0 = 10
    T1 = 60
    start = time.time()
    batch_size = dummy_input.size(0)
    while time.time() - start < T0:
        model(dummy_input)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(dummy_input)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print( batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

if __name__ == '__main__':
    # M0, M1, M2, M3, M4, M5
    batchsize = 1024
    device = torch.device('cpu')
    net = ExtremelyFastViT_M5(fuse=False, pretrained=False)
    net.eval()
    dummy_input = torch.randn(batchsize, 3, 224, 224).to(device)
    net = net.to(device)
    torch.onnx.export(net, dummy_input, 'vit_bs16.onnx')
    # profile(net, dummy_input)
    compute_throughput_cpu(net, dummy_input)
    # net(torch.randn(1, 3, 224, 224))
    # print(sum(p.numel() for p in net.parameters() if p.requires_grad), 'parameters')