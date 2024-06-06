import torch, torch.nn as nn
import torch.functional as F


__all__ = ["BigramLanguageModel", ]

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super(BigramLanguageModel, self).__init__()

        self.emb_layer = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:

        idx = idx.clone().type(torch.long)
        logits = self.emb_layer(idx)  # (B, T, C)

        return logits.permute(0, 2, 1) 