#!/usr/bin/env python3
"""Two closed-form toy transformer circuits. No training or external answer computation."""
import argparse
import json
import math

import torch
from torch import nn


def causal_attention(q, k, v, mask):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return scores.masked_fill(~mask, -torch.inf).softmax(dim=-1) @ v


def check_tokens(tokens, length, vocab_size):
    if tokens.ndim != 2 or tokens.shape[1] != length or tokens.shape[0] == 0:
        raise ValueError(f'expected a nonempty batch of {length} token IDs')
    if tokens.dtype != torch.long or bool(((tokens < 0) | (tokens >= vocab_size)).any()):
        raise ValueError(f'expected int64 token IDs in 0..{vocab_size - 1}')


def inventory(model):
    """Registered state, including frozen weights; source constants need separate review."""
    params = list(model.parameters())
    buffers = list(model.buffers())
    return {
        'parameter_elements': sum(p.numel() for p in params),
        'nonzero_parameter_elements': sum(int(p.count_nonzero()) for p in params),
        'buffer_elements': sum(b.numel() for b in buffers),
        'registered_tensor_bytes': sum(t.numel() * t.element_size() for t in params + buffers),
        'scope': 'registered tensors only; architecture and source constants are documented separately',
    }


class ParityCircuit(nn.Module):
    """Parity of 1..10 bits; IDs 0,1 are bits, 2 is left padding, 3 is delimiter.

    Residual channels: 0 = input bit, 1 = count, 2 = parity.
    One head, width 3, ReLU MLP width 10, identity normalization, float32 by default.
    Only the delimiter-position readout implements the specified function.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 3)
        self.q = nn.Linear(3, 3, bias=False)
        self.k = nn.Linear(3, 3, bias=False)
        self.v = nn.Linear(3, 3, bias=False)
        self.o = nn.Linear(3, 3, bias=False)
        self.up = nn.Linear(3, 10)
        self.down = nn.Linear(10, 3)
        self.readout = nn.Linear(3, 4)
        self.register_buffer('causal_mask', torch.ones(11, 11, dtype=torch.bool).tril())
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()
            self.embedding.weight[1, 0] = 1
            self.v.weight[1, 0] = 11  # Uniform average over 11 slots becomes a count.
            self.o.weight[1, 1] = 1
            for k in range(10):
                self.up.weight[k, 1] = 1
                self.up.bias[k] = -k
                self.down.weight[2, k] = 1 if k == 0 else 2 * (-1) ** k
            self.readout.weight[0, 2] = -1
            self.readout.bias[0] = 1
            self.readout.weight[1, 2] = 1
            self.readout.bias[2:] = -2  # Padding and delimiter are never answer tokens.
        self.requires_grad_(False)
        self.eval()

    def forward(self, tokens):
        check_tokens(tokens, 11, 4)
        if not bool((tokens[:, -1] == 3).all()) or bool((tokens[:, :-1] == 3).any()):
            raise ValueError('use ten bit/padding slots followed by one delimiter')
        x = self.embedding(tokens)
        x = x + self.o(causal_attention(self.q(x), self.k(x), self.v(x), self.causal_mask))
        x = x + self.down(self.up(x).relu())
        return self.readout(x[:, -1])


class PositionRouter(nn.Module):
    """Copy one of four input tokens (IDs 0..3) to the next-token readout.

    Residual channels: 0..3 = token, 4..7 = position, 8..11 = routed token.
    One head of dimension 4, score gap 16, identity normalization, no MLP.
    """
    def __init__(self, position):
        super().__init__()
        if type(position) is not int or not 0 <= position < 4:
            raise ValueError('position must be an integer in 0..3')
        self.embedding = nn.Embedding(4, 12)
        self.position_embedding = nn.Embedding(4, 12)
        self.q = nn.Linear(12, 4)
        self.k = nn.Linear(12, 4, bias=False)
        self.v = nn.Linear(12, 4, bias=False)
        self.o = nn.Linear(4, 12, bias=False)
        self.readout = nn.Linear(12, 4, bias=False)
        self.register_buffer('causal_mask', torch.ones(4, 4, dtype=torch.bool).tril())
        self.register_buffer('positions', torch.arange(4))
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()
            for i in range(4):
                self.embedding.weight[i, i] = 1
                self.position_embedding.weight[i, 4 + i] = 1
                self.k.weight[i, 4 + i] = 1
                self.v.weight[i, i] = 1
                self.o.weight[8 + i, i] = 1
                self.readout.weight[i, 8 + i] = 1
            self.q.bias[position] = 16 * math.sqrt(4)
        self.requires_grad_(False)
        self.eval()

    def forward(self, tokens):
        check_tokens(tokens, 4, 4)
        x = self.embedding(tokens) + self.position_embedding(self.positions)
        x = x + self.o(causal_attention(self.q(x), self.k(x), self.v(x), self.causal_mask))
        return self.readout(x[:, -1])


def encode_bits(bits):
    """Input formatting only; does not compute a parity label."""
    if not 1 <= len(bits) <= 10 or any(c not in '01' for c in bits):
        raise ValueError('bits must be a string of 1..10 binary digits')
    return [2] * (10 - len(bits)) + [int(c) for c in bits] + [3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task', required=True)
    p = sub.add_parser('parity')
    p.add_argument('--bits', required=True)
    p = sub.add_parser('route')
    p.add_argument('--tokens', required=True, help='four digits, each in 0..3')
    p.add_argument('--position', required=True, type=int, help='zero-based source position')
    args = parser.parse_args()
    try:
        if args.task == 'parity':
            model = ParityCircuit()
            encoded = encode_bits(args.bits)
            architecture = 'one causal head, width 3, ReLU MLP width 10, identity normalization'
        else:
            if len(args.tokens) != 4 or any(c not in '0123' for c in args.tokens):
                raise ValueError('tokens must contain four digits in 0..3')
            model = PositionRouter(args.position)
            encoded = [int(c) for c in args.tokens]
            architecture = 'one causal head, width 12, head dimension 4, no MLP, identity normalization'
        with torch.inference_mode():
            logits = model(torch.tensor([encoded], dtype=torch.long))[0]
        print(json.dumps({'task': args.task, 'prediction': int(logits.argmax()),
                          'logits': logits.tolist(), 'architecture': architecture,
                          **inventory(model)}, indent=2))
    except ValueError as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
