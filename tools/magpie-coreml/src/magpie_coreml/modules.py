"""Traceable re-implementations of Magpie's stages, sharing parameters with the NeMo model.

Shapes are fixed so every stage converts to a static ML Program. Masks are additive and finite
(`NEG`) so fp16 never produces inf or NaN. Decoder rows are [conditional, unconditional]: the
second row is the classifier-free-guidance branch, whose context embedding and text are zeroed,
so its cross-attention contributes exactly zero and is skipped.
"""

import torch
import torch.nn.functional as F
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.modules.transformer_2501 import TransformerLayer
from torch import Tensor, nn

from magpie_coreml.checkpoint import AUDIO_EOS, CODES, FORBIDDEN, VOCAB

NEG = -1e4
TEXT = 256  # text bucket, as in the speech-swift graphs
CONTEXT = 217  # baked speaker context frames
CACHE = 480  # self-attention cache: 217 context + 1 BOS + 250 steps, rounded up to a multiple of 16
HEADS = 12
HEAD_DIM = 64
CROSS_DIM = 128  # one cross-attention head
D = 768
TOP_K = 80
PRIOR_LAYERS = range(2, 11)
ALIGNMENT_LAYERS = (4, 5, 8, 9)


def _self_qkv(layer: TransformerLayer, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(B, T, D) -> q, k, v as (B, H, T, Dh)."""
    qkv = layer.self_attention.qkv_net(layer.norm_self(x)).unflatten(-1, (3, HEADS, HEAD_DIM))
    q, k, v = qkv.unbind(2)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def _attend(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
    """softmax(q k^T / sqrt(d) + mask) v, merged back to (B, T, H * Dh)."""
    scores = torch.matmul(q, k.transpose(-1, -2)) * HEAD_DIM**-0.5 + mask
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out.transpose(1, 2).flatten(2)


def _ffn(layer: TransformerLayer, x: Tensor) -> Tensor:
    """Kernel-1 position-wise FFN as two matmuls."""
    w1 = layer.pos_ff.proj.conv.weight[:, :, 0]
    w2 = layer.pos_ff.o_net.conv.weight[:, :, 0]
    return F.linear(F.gelu(F.linear(layer.norm_pos_ff(x), w1), approximate="tanh"), w2)


def _cross_kv(layer: TransformerLayer, memory: Tensor) -> tuple[Tensor, Tensor]:
    """Cross-attention keys and values of the text, (1, 1, T, 128) each."""
    kv = layer.cross_attention.kv_net(layer.norm_xattn_memory(memory))
    k, v = kv.unflatten(-1, (2, 1, CROSS_DIM)).unbind(2)
    return k.transpose(1, 2), v.transpose(1, 2)


def _cross_probs(layer: TransformerLayer, x: Tensor, k: Tensor, mask: Tensor) -> Tensor:
    """Cross-attention probabilities of the conditional row x: (1, T, D) -> (1, 1, T, T_text)."""
    q = layer.cross_attention.q_net(layer.norm_xattn_query(x)).unsqueeze(1)
    return torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * CROSS_DIM**-0.5 + mask, dim=-1)


def _cross_out(layer: TransformerLayer, probs: Tensor, v: Tensor) -> Tensor:
    out = torch.matmul(probs, v)  # (1, 1, T, 128)
    return layer.cross_attention.o_net(out.squeeze(1))


class TextEncoder(nn.Module):
    """tokens (1, 256) int32, mask (1, 256) -> encoder_output (1, 256, 768).

    The encoder is causal (attention and kernel-3 convolutions), so padding after the text never
    reaches a real position; the mask only zeroes the padded rows.
    """

    def __init__(self, model: MagpieTTSModel):
        super().__init__()
        self.embedding = model.text_embedding
        self.encoder = model.encoder
        self.register_buffer("causal", torch.triu(torch.full((TEXT, TEXT), NEG), diagonal=1))

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        x = self.embedding(tokens) + self.encoder.position_embeddings.weight[:TEXT]
        for layer in self.encoder.layers:
            q, k, v = _self_qkv(layer, x)
            x = x + layer.self_attention.o_net(_attend(q, k, v, self.causal))
            h = layer.norm_pos_ff(x).transpose(1, 2)
            h = F.gelu(F.conv1d(F.pad(h, (2, 0)), layer.pos_ff.proj.conv.weight), approximate="tanh")
            h = F.conv1d(F.pad(h, (2, 0)), layer.pos_ff.o_net.conv.weight)
            x = x + h.transpose(1, 2)
        return self.encoder.norm_out(x) * mask.unsqueeze(-1)


class DecoderPrefill(nn.Module):
    """Runs the 217 baked context frames of one speaker through the decoder, for both CFG rows.

    Inputs: speaker_idx (1,) int32, encoder_output (1, 256, 768), encoder_mask (1, 256).
    Outputs: sa_k_i, sa_v_i (2, 12, 217, 64) self-attention keys/values per layer (row 0
    conditional, row 1 unconditional) and xa_k_i, xa_v_i (1, 1, 256, 128) cross-attention
    keys/values of the text.
    """

    def __init__(self, model: MagpieTTSModel):
        super().__init__()
        self.decoder = model.decoder
        self.speakers = model.baked_context_embedding.weight.reshape(-1, CONTEXT, D)
        self.register_buffer("causal", torch.triu(torch.full((CONTEXT, CONTEXT), NEG), diagonal=1))

    def forward(
        self, speaker_idx: Tensor, encoder_output: Tensor, encoder_mask: Tensor
    ) -> tuple[Tensor, ...]:
        context = self.speakers[speaker_idx]  # (1, 217, 768)
        x = (
            torch.cat([context, torch.zeros_like(context)])
            + self.decoder.position_embeddings.weight[:CONTEXT]
        )
        text_mask = ((1 - encoder_mask) * NEG)[:, None, None, :]
        self_k, self_v, cross_k, cross_v = [], [], [], []
        for layer in self.decoder.layers:
            q, k, v = _self_qkv(layer, x)
            x = x + layer.self_attention.o_net(_attend(q, k, v, self.causal))
            xk, xv = _cross_kv(layer, encoder_output)
            cross = _cross_out(layer, _cross_probs(layer, x[:1], xk, text_mask), xv)
            x = x + torch.cat([cross, torch.zeros_like(cross)])
            x = x + _ffn(layer, x)
            self_k.append(k)
            self_v.append(v)
            cross_k.append(xk)
            cross_v.append(xv)
        return (*self_k, *self_v, *cross_k, *cross_v)


class DecoderStep(nn.Module):
    """One decoder position for both CFG rows, with the self-attention cache held as state.

    Inputs: audio_emb (1, 1, 768) embedding of the previous step's 16 codes (shared by both
    rows), position (1,) int32 absolute position of that input (217 for the BOS step),
    encoder_mask (1, 256), attn_prior (1, 256) multiplicative prior on the conditional row's
    cross-attention in layers 2-10 (all ones disables it), cfg_scale (1,), xa_k_i / xa_v_i
    (1, 1, 256, 128). States: sa_k_i, sa_v_i (2, 12, 480, 64), fp16 in Core ML (`cache_dtype`
    is float32 only for the PyTorch parity checks).
    Outputs: logits (1, 1, 16, 2024) CFG-combined, h_last (2, 1, 768) decoder output of both
    rows for the local transformer, alignment (1, 256) mean cross-attention of layers 4, 5, 8, 9.
    """

    def __init__(self, model: MagpieTTSModel, cache_dtype: torch.dtype):
        super().__init__()
        self.decoder = model.decoder
        self.final_proj = model.final_proj
        self.cache_dtype = cache_dtype
        for i in range(len(model.decoder.layers)):
            self.register_buffer(f"sa_k_{i}", torch.zeros(2, HEADS, CACHE, HEAD_DIM, dtype=cache_dtype))
            self.register_buffer(f"sa_v_{i}", torch.zeros(2, HEADS, CACHE, HEAD_DIM, dtype=cache_dtype))
        self.register_buffer("slots", torch.arange(CACHE, dtype=torch.float32))

    def forward(
        self,
        audio_emb: Tensor,
        position: Tensor,
        encoder_mask: Tensor,
        attn_prior: Tensor,
        cfg_scale: Tensor,
        *cross: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        layers = len(self.decoder.layers)
        cross_k, cross_v = cross[:layers], cross[layers:]
        at = position.to(torch.float32)
        write = (self.slots == at).to(self.cache_dtype)[:, None]  # (CACHE, 1)
        mask = ((self.slots > at).to(torch.float32) * NEG)[None, None, None, :]
        pos = F.embedding(position, self.decoder.position_embeddings.weight)  # (1, 768)
        x = (audio_emb + pos).expand(2, 1, D)
        text_mask = ((1 - encoder_mask) * NEG)[:, None, None, :]
        prior = attn_prior[:, None, None, :]
        alignment = []
        for i, layer in enumerate(self.decoder.layers):
            q, k, v = _self_qkv(layer, x)
            cache_k = getattr(self, f"sa_k_{i}")
            cache_v = getattr(self, f"sa_v_{i}")
            keys = cache_k * (1 - write) + k.to(self.cache_dtype) * write
            values = cache_v * (1 - write) + v.to(self.cache_dtype) * write
            cache_k[:] = keys
            cache_v[:] = values
            x = x + layer.self_attention.o_net(_attend(q, keys.to(q.dtype), values.to(q.dtype), mask))
            probs = _cross_probs(layer, x[:1], cross_k[i], text_mask)
            if i in PRIOR_LAYERS:
                probs = probs * prior
                probs = probs / probs.sum(dim=-1, keepdim=True)
            if i in ALIGNMENT_LAYERS:
                alignment.append(probs)
            cross_out = _cross_out(layer, probs, cross_v[i])
            x = x + torch.cat([cross_out, torch.zeros_like(cross_out)])
            x = x + _ffn(layer, x)
        h = self.decoder.norm_out(x)
        both = self.final_proj(h)  # (2, 1, 16 * 2024)
        logits = both[1:] + cfg_scale * (both[:1] - both[1:])
        return (
            logits.reshape(1, 1, CODES, VOCAB),
            h,
            torch.stack(alignment).mean(dim=0).reshape(1, -1),
        )


class LocalTransformer(nn.Module):
    """Samples the 16 codes of one decoder step (2 frames x 8 codebooks) autoregressively.

    Inputs: h_last (2, 1, 768) decoder output of both CFG rows, gumbel (16, 2024) Gumbel(0, 1)
    noise (zeros give greedy decoding), temperature (1,), cfg_scale (1,), forbid_eos (1,)
    1 to forbid AUDIO_EOS, 0 to allow it.
    Outputs: codes (1, 16) in codebook-major order (code j is codebook j % 8 of frame j // 8), as
    floats holding exact integers -- an int32 output keeps the whole graph off the ANE --, and
    audio_emb (1, 1, 768) mean of the 16 code embeddings, the next decoder input.

    Sampling is NeMo's: CFG-combine the logits, forbid special tokens, keep the top 80, then draw
    from softmax(logits / temperature) -- here as argmax(logits / temperature + gumbel).

    The 16 steps are unrolled. Keys and values go to fixed 16-slot buffers through one-hot writes,
    with a mask over unwritten slots: growing them by concatenation makes the ANE compiler reject
    the whole graph.
    """

    def __init__(self, model: MagpieTTSModel):
        super().__init__()
        self.transformer = model.local_transformer
        self.heads = model.local_transformer_out_projections
        self.embeddings = model.audio_embeddings
        forbidden = torch.zeros(VOCAB)
        forbidden[list(FORBIDDEN)] = NEG
        self.register_buffer("forbidden", forbidden)
        eos = torch.zeros(VOCAB)
        eos[AUDIO_EOS] = NEG
        self.register_buffer("eos", eos)
        self.register_buffer("ids", torch.arange(VOCAB, dtype=torch.float32))
        self.register_buffer("slots", torch.eye(CODES)[:, :, None])  # one-hot write of slot j
        self.register_buffer("visible", torch.triu(torch.full((CODES, CODES), NEG), diagonal=1))

    def forward(
        self, h_last: Tensor, gumbel: Tensor, temperature: Tensor, cfg_scale: Tensor, forbid_eos: Tensor
    ) -> tuple[Tensor, Tensor]:
        positions = self.transformer.position_embeddings.weight
        bias = self.forbidden + forbid_eos * self.eos
        empty = torch.zeros(2, HEADS, CODES, HEAD_DIM)
        keys = [empty for _ in self.transformer.layers]
        values = [empty for _ in self.transformer.layers]
        x = h_last + positions[0]
        codes, embedded = [], []
        for j in range(CODES):
            for i, layer in enumerate(self.transformer.layers):
                q, k, v = _self_qkv(layer, x)
                keys[i] = keys[i] + k * self.slots[j]
                values[i] = values[i] + v * self.slots[j]
                x = x + layer.self_attention.o_net(_attend(q, keys[i], values[i], self.visible[j]))
                x = x + _ffn(layer, x)
            both = self.heads[j](x[:, 0])  # (2, 2024)
            logits = both[1:] + cfg_scale * (both[:1] - both[1:]) + bias
            threshold = torch.topk(logits, TOP_K, dim=-1).values[:, -1:]
            scores = torch.where(logits < threshold, NEG, logits / temperature + gumbel[j : j + 1])
            code = torch.topk(scores, 1, dim=-1).indices.to(torch.float32)  # (1, 1)
            onehot = (self.ids == code).to(x.dtype)
            embedding = onehot @ self.embeddings[j].weight  # (1, 768)
            codes.append(code)
            embedded.append(embedding)
            x = (embedding + positions[j + 1]).expand(2, 1, D)
        return torch.cat(codes, dim=1), torch.stack(embedded).mean(dim=0)[None]


def _half_snake(x: Tensor, activation: nn.Module) -> Tensor:
    """Snake x + sin(alpha x)^2 / alpha on the first half of the channels, leaky ReLU on the rest."""
    half = activation.activation.snake_channels
    alpha = activation.activation.snake_act.alpha
    snake = x[:, :half]
    snake = snake + torch.sin(alpha * snake) ** 2 / (alpha + 1e-9)
    return torch.cat([snake, F.leaky_relu(x[:, half:], 0.01)], dim=1)


def _taps(full: Tensor, bias: Tensor | None, conv: nn.Conv1d, weight: Tensor) -> Tensor:
    """Unpadded convolution of `full` with `conv`'s dilation.

    The ANE rejects convolutions with dilation 5 and more than 3 taps, so those are split into
    groups of at most 3 taps over shifted inputs and summed.
    """
    kernel, dilation = conv.kernel_size[0], conv.dilation[0]
    if dilation < 5 or kernel <= 3:
        return F.conv1d(full, weight, bias, dilation=dilation)
    out = None
    for first in range(0, kernel, 3):
        last = min(first + 3, kernel)
        part = F.conv1d(
            full[..., first * dilation :],
            weight[:, :, first:last],
            bias if first == 0 else None,
            dilation=dilation,
        )
        trim = (kernel - last) * dilation
        part = part[..., :-trim] if trim else part
        out = part if out is None else out + part
    return out


def _phases(conv: nn.ConvTranspose1d, weight: Tensor) -> Tensor:
    """Kernel-2 convolution weights giving the s output phases of NeMo's causal transposed
    convolution (kernel 2s, stride s, one group per output channel, right-trimmed by s)."""
    stride, groups = conv.stride[0], conv.groups
    taps = torch.stack([weight[:, 0, stride:], weight[:, 0, :stride]], dim=-1)  # (in, s, [t-1, t])
    per_group = conv.in_channels // groups
    return taps.reshape(groups, per_group, stride, 2).transpose(1, 2).reshape(groups * stride, per_group, 2)


_SLOT = 64


class Codec(nn.Module):
    """NanoCodec's causal HiFi-GAN decoder, streaming.

    latents (1, 32, W) and one history tensor per channel count -> audio (1, W * 1024) and the
    updated histories.

    Every causal convolution needs the last (kernel - 1) * dilation samples of its input, and
    every upsampling its last input sample, from the previous call. Zeros are exactly the
    padding a whole-utterance decode starts with, so decoding chunk by chunk, feeding each call's
    histories to the next, reproduces it. Each convolution owns a 64-sample slot holding the
    last 64 samples of its input: concatenating a slot of any other length in front of the
    input makes the ANE copy it, and the codec four times slower. (As Core ML state, the
    histories make the model fail to load with execution plan error -14.)

    Upsampling is a kernel-2 convolution producing the s phases as channels, then interleaved:
    with conv_transpose followed by the causal trim, the next convolution on the ANE comes out
    wrong (SNR 3 dB against PyTorch).
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        # Weight-normed weights, materialized once; parametrize would recompute them in the graph.
        self.weights = {id(conv): conv.weight.detach().clone() for conv, _ in self._histories()}
        for upsample in decoder.up_sample_conv_layers:
            self.weights[id(upsample.conv)] = _phases(upsample.conv, upsample.conv.weight.detach())
        self.layout: dict[int, tuple[int, int, int]] = {}  # id(conv) -> (channels, slot start, length)
        self.histories: dict[int, int] = {}  # channels -> total length
        for conv, length in self._histories():
            channels = conv.in_channels
            self.layout[id(conv)] = (channels, self.histories.get(channels, 0), length)
            self.histories[channels] = self.histories.get(channels, 0) + _SLOT

    def _histories(self) -> list[tuple[nn.Module, int]]:
        """Every convolution in execution order with the input history it needs."""
        decoder = self.decoder

        def causal(conv: nn.Conv1d) -> tuple[nn.Module, int]:
            return conv, (conv.kernel_size[0] - 1) * conv.dilation[0]

        convs = [causal(decoder.pre_conv.conv)]
        for upsample, residual in zip(decoder.up_sample_conv_layers, decoder.res_layers, strict=True):
            convs.append((upsample.conv, 1))
            for block in residual.res_blocks:
                for unit in block.res_blocks:
                    convs += [causal(unit.input_conv.conv), causal(unit.skip_conv.conv)]
        return [*convs, causal(decoder.post_conv.conv)]

    def forward(self, latents: Tensor, *histories: Tensor) -> tuple[Tensor, ...]:
        decoder = self.decoder
        history = dict(zip(self.histories, histories, strict=True))
        written: dict[int, list[Tensor]] = {channels: [] for channels in self.histories}

        def extend(x: Tensor, conv: nn.Module) -> Tensor:
            """x preceded by the `length` input samples the previous call ended with."""
            channels, start, length = self.layout[id(conv)]
            slot = torch.cat([history[channels][..., start : start + _SLOT], x], dim=-1)
            written[channels].append(slot[..., -_SLOT:])
            return slot[..., _SLOT - length :]

        def causal(x: Tensor, conv: nn.Conv1d) -> Tensor:
            return _taps(extend(x, conv), conv.bias, conv, self.weights[id(conv)])

        x = causal(latents, decoder.pre_conv.conv)
        for activation, upsample, residual in zip(
            decoder.activations, decoder.up_sample_conv_layers, decoder.res_layers, strict=True
        ):
            up = upsample.conv
            stride, groups = up.stride[0], up.groups
            a = _half_snake(x, activation)
            bias = up.bias.repeat_interleave(stride)
            phases = F.conv1d(extend(a, up), self.weights[id(up)], bias, groups=groups)
            x = phases.unflatten(1, (groups, stride)).transpose(2, 3).flatten(2)
            branches = []
            for block in residual.res_blocks:
                y = x
                for unit in block.res_blocks:
                    skip = causal(_half_snake(y, unit.input_activation), unit.input_conv.conv)
                    y = y + causal(_half_snake(skip, unit.skip_activation), unit.skip_conv.conv)
                branches.append(y)
            x = sum(branches[1:], branches[0]) / len(branches)
        x = causal(_half_snake(x, decoder.post_activation), decoder.post_conv.conv)
        audio = torch.clamp(x, -1.0, 1.0)[:, 0]
        return audio, *(torch.cat(parts, dim=-1) for parts in written.values())
