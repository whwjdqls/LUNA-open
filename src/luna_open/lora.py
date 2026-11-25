"""Independent low-rank linear adapters; the original weight remains frozen.

Uses W x + (alpha/r) B A x, random A and zero B. Applied to the released LHM
reconstruction transformer's linear layers only; no pretrained file is edited.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__()
        if not isinstance(base,nn.Linear) or type(rank) is not int or rank<1:
            raise ValueError('LoRA requires a linear layer and positive integer rank')
        if not math.isfinite(alpha) or alpha<=0:
            raise ValueError('LoRA alpha must be positive and finite')
        self.base=base.requires_grad_(False)
        self.in_features=base.in_features;self.out_features=base.out_features
        self.rank=rank;self.alpha=float(alpha);self.scale=self.alpha/rank;self.enabled=True
        self.lora_A=nn.Parameter(base.weight.new_empty(rank,base.in_features))
        self.lora_B=nn.Parameter(base.weight.new_zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    @property
    def weight(self):return self.base.weight

    @property
    def bias(self):return self.base.bias

    def forward(self,x):
        out=self.base(x)
        if self.enabled:
            out=out+F.linear(F.linear(x,self.lora_A),self.lora_B)*self.scale
        return out


def inject_lora(transformer,rank=16,alpha=16.):
    if any(isinstance(m,LoRALinear) for m in transformer.modules()):
        raise ValueError('LoRA already installed')
    targets=[(name,m) for name,m in transformer.named_modules() if isinstance(m,nn.Linear)]
    if not targets:raise ValueError('No linear LoRA targets')
    transformer.requires_grad_(False)
    for name,base in targets:
        parent,_,key=name.rpartition('.')
        setattr(transformer.get_submodule(parent) if parent else transformer,key,LoRALinear(base,rank,alpha))
    return [name for name,_ in targets]


def adapter_parameters(model):
    return [(name,p) for name,p in model.named_parameters() if name.endswith(('.lora_A','.lora_B'))]


def adapter_state(model):
    return {name:p for name,p in adapter_parameters(model)}


def load_adapter_state(model,state):
    expected=dict(adapter_parameters(model))
    if state.keys()!=expected.keys():raise ValueError('LoRA parameter membership differs')
    for name,p in expected.items():
        if p.shape!=state[name].shape or not torch.isfinite(state[name]).all():
            raise ValueError(f'Invalid LoRA parameter: {name}')
    with torch.no_grad():
        for name,p in expected.items():p.copy_(state[name])


@contextmanager
def adapters_disabled(model):
    modules=[m for m in model.modules() if isinstance(m,LoRALinear)]
    previous=[m.enabled for m in modules]
    try:
        for m in modules:m.enabled=False
        yield
    finally:
        for m,enabled in zip(modules,previous):m.enabled=enabled
