#!/usr/bin/env python3
"""Export every ONNX asset the Parakeet server needs, straight from the NVIDIA
checkpoint, so we no longer depend on third-party conversions.

Produces, into --out/ (default training/assets/):

    config.json                     model config (tiny, generated)
    vocab.txt                       SentencePiece vocab + <blk> blank (8193 lines)
    nemo128.onnx                    mel preprocessor (test reference)
    encoder-model.onnx(.data)       fp32 encoder, FULL attention
    encoder-model.int8.onnx          int8 encoder, FULL attention
    encoder-local.onnx(.data)       fp32 encoder, LOCAL attention (rel_pos_local_attn)
    encoder-local.int8.onnx          int8 encoder, LOCAL attention
    decoder_joint-model.onnx         fp32 decoder + joint
    decoder_joint-model.int8.onnx     int8 decoder + joint
    manifest.txt                    sha256 of every artifact

The only external dependency is the NVIDIA checkpoint:

    https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/parakeet-tdt-0.6b-v3.nemo

Run from the repo root with the training venv active (see training/requirements.txt):

    python training/export.py --checkpoint training/checkpoint/parakeet-tdt-0.6b-v3.nemo --out training/assets

Why the local-attention forward is patched:
    NeMo 3.0 exports RelPositionMultiHeadAttentionLongformer through its
    sliding-chunks path, which reconstructs the attention band with
    `new_empty(...)` plus in-place slice assignments. The torchscript exporter
    mis-compiles those and the encoder outputs garbage. We replace the forward
    with an equivalent full-attention + band-mask implementation that uses only
    export-safe ops. See DESIGN_DECISIONS.md for the full story.
"""

import argparse
import hashlib
import os
import glob
import shutil
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from onnxruntime.quantization import quantize_dynamic, QuantType

CHECKPOINT_URL = (
    "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/"
    "parakeet-tdt-0.6b-v3.nemo"
)
DEFAULT_CHECKPOINT = "training/checkpoint/parakeet-tdt-0.6b-v3.nemo"

# Weight params of the shipped model. Keep in sync with models/config.json.
CONFIG = {
    "model_type": "nemo-conformer-tdt",
    "features_size": 128,
    "subsampling_factor": 8,
}

FLAG = 0


def log(msg):
    print(f"[export] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        log(f"checkpoint ya existe: {dest}")
        return dest
    log(f"descargando checkpoint -> {dest}")
    tmp = dest + ".part"
    import subprocess

    subprocess.run(
        ["curl", "-L", "-o", tmp, CHECKPOINT_URL], check=True
    )
    os.rename(tmp, dest)
    return dest


def load_model(checkpoint):
    import nemo.collections.asr.models as M

    log("cargando checkpoint en NeMo (cpu)...")
    m = M.EncDecRNNTBPEModel.restore_from(checkpoint, map_location="cpu")
    m.eval()
    return m


def export_vocab(model, out):
    path = os.path.join(out, "vocab.txt")
    vocab = model.tokenizer.vocab  # list of 8192 tokens in ID order
    with open(path, "w", encoding="utf-8") as f:
        for i, tok in enumerate(vocab):
            f.write(f"{tok} {i}\n")
        f.write(f"<blk> {len(vocab)}\n")
    log(f"vocab.txt ({len(vocab) + 1} lineas)")


def export_config(out):
    import json

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)


def export_preprocessor(model, out):
    log("exportando nemo128.onnx (preprocessor)...")
    # nemo128.onnx es solo una referencia de tests (no lo carga el servidor).
    # torch.stft devuelve complex en torch>=2.x y el exportador torchscript no lo
    # traza; lo dejamos best-effort y seguimos con el resto.
    try:
        pre = model.preprocessor

        class Wrapped(torch.nn.Module):
            def forward(self, waveforms, waveforms_lens):
                return pre(input_signal=waveforms, length=waveforms_lens)

        wrapped = Wrapped().eval()
        dummy_wave = torch.zeros(1, 16000, dtype=torch.float32)
        dummy_len = torch.tensor([16000], dtype=torch.long)
        path = os.path.join(out, "nemo128.onnx")
        torch.onnx.export(
            wrapped,
            (dummy_wave, dummy_len),
            path,
            opset_version=17,
            input_names=["waveforms", "waveforms_lens"],
            output_names=["features", "features_lens"],
            dynamo=False,
            dynamic_axes={
                "waveforms": {0: "batch_size", 1: "N"},
                "waveforms_lens": {0: "batch_size"},
                "features": {0: "batch_size", 2: "T"},
                "features_lens": {0: "batch_size"},
            },
        )
        log("nemo128.onnx OK")
    except Exception as e:
        log(f"AVISO: nemo128.onnx no exportable (STFT complejo): {type(e).__name__}")


def patch_local_attention():
    """Replace the sliding-chunks forward with an export-safe band-mask one."""
    import nemo.collections.asr.parts.submodules.multi_head_attention as mha

    def forward_local(self, query, key, value, pad_mask, pos_emb, cache=None):
        key, value, query, cache = self.update_cache(
            key=key, value=value, query=query, cache=cache
        )
        q, k, v = self.forward_qkv(query, key, value)  # (b, head, T, dk)
        b, h, T, dk = q.size()
        w = max(self.att_context_size[0], self.att_context_size[1])
        nbp = pos_emb.size(0)
        pld = T - 1 - w
        pos_pad = F.pad(pos_emb, (0, 0, pld, pld))  # (1, 2T-1, d_model)
        p = self.linear_pos(pos_pad).view(nbp, -1, self.h, self.d_k).transpose(1, 2)
        q_t = q.transpose(1, 2)
        q_u = (q_t + self.pos_bias_u).transpose(1, 2)
        q_v = (q_t + self.pos_bias_v).transpose(1, 2)
        mac = torch.matmul(q_u, k.transpose(-2, -1))
        mbd = torch.matmul(q_v, p.transpose(-2, -1))
        mbd = self.rel_shift(mbd)
        mbd = mbd[:, :, :, :T]
        scores = (mac + mbd) / self.s_d_k
        idx = torch.arange(T, device=q.device, dtype=q.dtype)
        band = ((idx[:, None] - idx[None, :]).abs() > w).unsqueeze(0)
        if pad_mask is not None:
            pm = pad_mask.bool()
            key_mask = pm.unsqueeze(1).expand(b, T, T) if pm.dim() == 2 else pm
            mask = key_mask | band
        else:
            mask = band.expand(b, T, T)
        out = self.forward_attention(v, scores, mask)
        return out if cache is None else (out, cache)

    mha.RelPositionMultiHeadAttentionLongformer.forward = forward_local


def consolidate_external_data(src_onnx, dst_onnx):
    """Fold any scattered external weights into one .onnx + one .data sidecar."""
    model = onnx.load(src_onnx, load_external_data=True)
    data_path = dst_onnx + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
    onnx.save_model(
        model,
        dst_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
        size_threshold=1024,
    )
    # onnx.save_model writes the .data next to the .onnx under its own naming
    written = dst_onnx
    log(f"  fp32 -> {os.path.basename(dst_onnx)} (+external data)")


def quantize(src_onnx, dst_onnx):
    quantize_dynamic(
        src_onnx,
        dst_onnx,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
    )
    log(f"  int8 -> {os.path.basename(dst_onnx)}")


def export_encoder(model, out, local_attention, att_context=(256, 256)):
    tag = "local" if local_attention else "full"
    log(f"exportando encoder ({tag} attention)...")

    if local_attention:
        patch_local_attention()
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=list(att_context),
        )
        model.eval()

    with tempfile.TemporaryDirectory(dir=os.path.abspath(out)) as tmp:
        model.export(output=os.path.join(tmp, "parakeet.onnx"))
        enc = glob.glob(os.path.join(tmp, "encoder-*.onnx"))[0]
        dec = glob.glob(os.path.join(tmp, "decoder_joint-*.onnx"))[0]

        prefix = "encoder-local" if local_attention else "encoder-model"
        fp32_dst = os.path.join(out, prefix + ".onnx")
        consolidate_external_data(enc, fp32_dst)
        quantize(fp32_dst, os.path.join(out, prefix + ".int8.onnx"))

        # keep the decoder from this run (identical across attention modes)
        dec_dst = os.path.join(out, "decoder_joint-model.onnx")
        shutil.copy(dec, dec_dst)
        quantize(dec, os.path.join(out, "decoder_joint-model.int8.onnx"))
    log(f"encoder {tag} OK")


def write_manifest(out):
    paths = sorted(
        p for p in os.listdir(out) if os.path.isfile(os.path.join(out, p))
    )
    with open(os.path.join(out, "manifest.txt"), "w") as f:
        for p in paths:
            if p == "manifest.txt":
                continue
            f.write(f"{sha256(os.path.join(out, p))}  {p}\n")
    log("manifest.txt OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--out", default="training/assets")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (export only)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt = download_checkpoint(args.checkpoint)

    model = load_model(ckpt)
    if args.device == "cuda":
        model = model.cuda()

    export_vocab(model, args.out)
    export_config(args.out)
    export_preprocessor(model, args.out)

    # Full attention (current server default), then local attention.
    export_encoder(model, args.out, local_attention=False)
    export_encoder(model, args.out, local_attention=True)

    write_manifest(args.out)
    log(f"listo. assets en {args.out}")


if __name__ == "__main__":
    main()