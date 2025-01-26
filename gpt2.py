### Modelo GPT2

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# Configuración del modelo
# Hiperparámetros del modelo GPT: Se utilizan valores más pequeños que en el modelo original
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weight token emb: vocabulario x vector embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # weight pos encoding: tamaño de bloque x vector embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # heads
            ln_f = nn.LayerNorm(config.n_embd) # layer normalization
        ))
        self.lm_head = nn.linear(config.n_embd, config.vocab_size, bias=False) # linear head
