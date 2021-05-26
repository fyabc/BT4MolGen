#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Architectures for molecule Transformer training.

Transformer Base: L6 / E512 / F2048 / H8
"""

from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture


def _make_architectures():
    for l in [2, 4, 6, 8, 10]:
        for e in [16, 32, 64, 128, 256, 512, 1024]:
            for f in [4 * e, 8 * e]:
                for h in [4, 8]:
                    arch_name = f'transformer_mol_{l}.{e}.{f}.{h}'

                    def func(args, l=l, e=e, f=f, h=h):
                        args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', e)
                        args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', f)
                        args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', h)
                        args.encoder_layers = getattr(args, 'encoder_layers', l)
                        args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', e)
                        args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', f)
                        args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', h)
                        args.decoder_layers = getattr(args, 'decoder_layers', l)
                        base_architecture(args)

                    register_model_architecture('transformer', arch_name)(func)


_make_architectures()
