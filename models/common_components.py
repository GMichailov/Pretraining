import torch
import torch.nn as nn

# ===================================== Norms =====================================

## RMSNorms

class RMSNorm(nn.Module):
    def __init__(
            self, d_model: int, bias: bool, device: str,
            eps: float, weight_dtype: torch.dtype,
            calculation_dtype: torch.dtype
        ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=weight_dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=weight_dtype)) if bias else None
        self.calculation_dtype = calculation_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation_dtype = x.dtype
        x = x.to(self.calculation_dtype)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight.to(self.calculation_dtype) * x
        if self.bias is not None:
            x = x + self.bias
        return x.to(activation_dtype)


class UnweightedRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, calculation_dtype: str):
        super().__init__()
        self.eps = eps
        self.calculation_dtype = calculation_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation_dtype = x.dtype
        x = x.to(self.calculation_dtype)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(activation_dtype)


## LayerNorms

class LayerNorm(nn.Module):
    def __init__(
            self, d_model: int, bias: bool, device: str,
            eps: float, weight_dtype: torch.dtype,
            calculation_dtype: torch.dtype
        ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=weight_dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=weight_dtype)) if bias else None
        self.calculation_dtype = calculation_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation_dtype = x.dtype
        x = x.to(self.calculation_dtype)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        x = self.weight.to(self.calculation_dtype) * x
        if self.bias is not None:
            x = x + self.bias
        return x.to(activation_dtype)


class UnweightedLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, calculation_dtype: str):
        super().__init__()
        self.eps = eps
        self.calculation_dtype = calculation_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation_dtype = x.dtype
        x = x.to(self.calculation_dtype)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x.to(activation_dtype)


## AdaLNs

class AdaLN(nn.Module):
    def __init__(self, d_model: int, c_dim: int, bias: bool, device: str,
                 eps: float, weight_dtype: torch.dtype, calculation_dtype: torch.dtype):
        super().__init__()
        self.norm = UnweightedLayerNorm(d_model, eps, calculation_dtype)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(c_dim, 2 * d_model, bias=bias, device=device, dtype=weight_dtype)
        self.calculation_dtype = calculation_dtype

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        activation_dtype = x.dtype
        x = self.norm(x)
        c = c.to(self.calculation_dtype)
        scale, shift = self.linear(self.silu(c)).chunk(2, dim=-1)
        x = x * (1 + scale.to(self.calculation_dtype)) + shift.to(self.calculation_dtype)
        return x.to(activation_dtype)


class AdaLNZero(nn.Module):
    def __init__(self, d_model: int, c_dim: int, bias: bool, device: str,
                 eps: float, weight_dtype: torch.dtype, calculation_dtype: torch.dtype):
        super().__init__()
        self.norm = UnweightedLayerNorm(d_model, eps, calculation_dtype)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(c_dim, 3 * d_model, bias=bias, device=device, dtype=weight_dtype)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.calculation_dtype = calculation_dtype

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activation_dtype = x.dtype
        x = self.norm(x)
        c = c.to(self.calculation_dtype)
        scale, shift, gate = self.linear(self.silu(c)).chunk(3, dim=-1)
        x = x * (1 + scale.to(self.calculation_dtype)) + shift.to(self.calculation_dtype)
        return x.to(activation_dtype), gate.to(activation_dtype)


# ===================================== Attention =====================================

# These all take a latent hidden dim and rope/nope dimensions as params

## MHA

class MHA(nn.Module):
    def __init__(self, 
                 d_model: int, d_attention: int, num_heads: int, bias: bool, 
                 device: str, linear_dtype: torch.dtype, norm_dtype: torch.dtype,

    ):
        super().__init__()
        self.qkv = nn.Linear(3 * d_model, d_attention, bias, device, linear_dtype)
        self.


## MQA

## GQA

 