#!/usr/bin/env python3
"""Export the Nemotron-3-Diarization ONNX assets from the NVIDIA checkpoint.

Writes to --out (default: assets/):

    config.json, processor_config.json   as published upstream
    preprocessor_config.json             generated here, see write_preprocessor_config
    model.onnx                           fp32 streaming step
    model_quantized.onnx(+_data)         int8 streaming step
    manifest.txt                         revision, hashes, quantization scheme

The graph is one streaming step (shapes match internal/diarization/model.go):

    input_features  (1, T_mel, 128), cached_embeds (1, C, 512), attention_mask (1, C + T_mel/8)
    -> logits (1, C*8 + T_mel, 8), chunk_embeds (1, T_mel/8, 512), silence_embeds (512,)

The AOSC/FIFO speaker cache is not in the graph, it lives in Go.

Usage:

    python export.py                     # download the checkpoint, write assets/
    python export.py --checkpoint-dir X  # reuse an already downloaded checkpoint
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from transformers import AutoModelForAudioFrameClassification
# Imported from its module: the top-level alias is a decorated wrapper whose
# signature is (*args, **kwargs), and the mel defaults are read from it.
from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
    NemotronAsrStreamingFeatureExtractor,
)

MODEL_ID = "nvidia/Nemotron-3-Diarization"
# Pinned upstream revision. Bumping it means re-running the parity checks in
# self_check() and the end-to-end diarization test (testdata/diarization).
MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"

FAMILY_DIR = os.path.dirname(os.path.abspath(__file__))  # export/diarization
DEFAULT_CHECKPOINT = os.path.join(FAMILY_DIR, "checkpoint")
DEFAULT_OUT = os.path.join(FAMILY_DIR, "assets")

CHECKPOINT_FILES = ("config.json", "processor_config.json", "model.safetensors")
# Content hashes of the pinned revision. The repo publishes no tags, only a
# moving `main`, so the commit hash alone does not prove the files are the ones
# the export was validated against. Checked on every run.
CHECKPOINT_SHA256 = {
    "config.json": "dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873",
    "processor_config.json": "91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e",
    "model.safetensors": "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6",
}

OPSET = 20
# 8-bit weight-only, same scheme the previously shipped export used, so the
# quality of the int8 model does not regress.
QUANT_BITS = 8
QUANT_BLOCK_SIZE = 32
# The feature-stacking projection stays fp32: its output is `chunk_embeds`,
# fed back as `cached_embeds`, so quantizing it compounds error.
FP32_NODES = ("/embedder/projection/MatMul",)


def log(msg):
    print(f"[export] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(dest):
    """Fetch the pinned checkpoint files from Hugging Face and verify them."""
    from huggingface_hub import hf_hub_download

    os.makedirs(dest, exist_ok=True)
    for name in CHECKPOINT_FILES:
        target = os.path.join(dest, name)
        if os.path.exists(target):
            log(f"checkpoint file already present: {name}")
        else:
            log(f"downloading {MODEL_ID}@{MODEL_REVISION[:8]}/{name}")
            hf_hub_download(repo_id=MODEL_ID, filename=name, revision=MODEL_REVISION, local_dir=dest)

        want = CHECKPOINT_SHA256[name]
        got = sha256(target)
        if got != want:
            raise SystemExit(f"checkpoint file {name} does not match the pinned hash: {got} != {want}")
    return dest


def clear_graphs(out_dir):
    """Remove previous graphs: onnx.save appends to an existing external-data file."""
    for name in (
        "model.onnx",
        "model_quantized.onnx",
        "model_quantized.onnx_data",
        "model_quantized.onnx.tmp",
        "model_quantized.onnx.tmp.data",
    ):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)


def write_preprocessor_config(model_config, out_dir):
    """Emit the feature extractor config the Go front-end consumes."""
    signature = inspect.signature(NemotronAsrStreamingFeatureExtractor.__init__)
    defaults = {
        name: p.default
        for name, p in signature.parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    wanted = {"hop_length", "n_fft", "win_length", "preemphasis", "sampling_rate", "padding_value"}
    missing = wanted - defaults.keys()
    if missing:
        raise SystemExit(f"cannot read feature extractor defaults: {sorted(missing)}")

    cfg = {
        "feature_extractor_type": "NemotronAsrStreamingFeatureExtractor",
        "feature_size": model_config.audio_config.num_mel_bins,
        "hop_length": defaults["hop_length"],
        "n_fft": defaults["n_fft"],
        "padding_side": "right",
        "padding_value": defaults["padding_value"],
        "preemphasis": defaults["preemphasis"],
        "return_attention_mask": True,
        "sampling_rate": defaults["sampling_rate"],
        "win_length": defaults["win_length"],
        "processor_class": "Nemotron3DiarizationProcessor",
    }
    with open(os.path.join(out_dir, "preprocessor_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    return cfg


class DiarizationStep(nn.Module):
    """One streaming step, without the speaker-cache policy (it runs in Go).

    Position ids restart at zero on every chunk, as the streaming model expects.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, cached_embeds, attention_mask):
        tower = self.model.model.audio_tower
        chunk_embeds = tower.embedder(input_features)
        inputs_embeds = torch.cat([cached_embeds, chunk_embeds], dim=1)
        position_ids = torch.arange(inputs_embeds.shape[1], dtype=torch.long)[None, :]

        hidden = tower(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
        )[0]
        logits = self.model.classifier(self.model.model.upsampler(self.model.model.proj(hidden)))
        return logits, chunk_embeds, self.model.silence_embeds


def export_fp32(model, out_path):
    step = DiarizationStep(model).eval()
    audio = model.config.audio_config
    t_mel, cached = 800, 41  # 100 encoder frames + a non-trivial speaker cache

    torch.manual_seed(0)
    inputs = (
        torch.randn(1, t_mel, audio.num_mel_bins),
        torch.randn(1, cached, audio.hidden_size),
        torch.ones(1, cached + t_mel // audio.subsampling_factor, dtype=torch.int64),
    )

    log(f"exporting fp32 step graph (opset {OPSET})")
    with torch.no_grad():
        torch.onnx.export(
            step,
            inputs,
            out_path,
            input_names=["input_features", "cached_embeds", "attention_mask"],
            output_names=["logits", "chunk_embeds", "silence_embeds"],
            dynamic_axes={
                "input_features": {1: "num_mel_frames"},
                "cached_embeds": {1: "num_cached_frames"},
                "attention_mask": {1: "num_frames"},
                "logits": {1: "num_logit_frames"},
                "chunk_embeds": {1: "num_chunk_frames"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    return step


def quantize_int8(fp32_path, out_path):
    """8-bit weight-only quantization (MatMulNBits), weights on external data."""
    # The quantizer logs every node it walks; that is noise here.
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.ERROR)

    log(f"quantizing int8 (MatMulNBits {QUANT_BITS} bits, block {QUANT_BLOCK_SIZE}, symmetric)")
    config = DefaultWeightOnlyQuantConfig(
        block_size=QUANT_BLOCK_SIZE,
        is_symmetric=True,
        bits=QUANT_BITS,
    )
    quantizer = MatMulNBitsQuantizer(
        fp32_path,
        bits=QUANT_BITS,
        block_size=QUANT_BLOCK_SIZE,
        is_symmetric=True,
        nodes_to_exclude=list(FP32_NODES),
        algo_config=config,
    )
    quantizer.process()

    # Re-save through onnx: the quantizer's writer leaves dead weight bytes in
    # the external-data file.
    tmp = out_path + ".tmp"
    quantizer.model.save_model_to_file(tmp, use_external_data_format=True)
    model = onnx.load(tmp)
    onnx.save(
        model,
        out_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(out_path) + "_data",
        size_threshold=1024,
    )
    for leftover in (tmp, tmp + ".data", tmp + "_data"):
        if os.path.exists(leftover):
            os.remove(leftover)
    # onnx.save creates 0600 files; the images may be extracted as non-root.
    os.chmod(out_path, 0o644)
    os.chmod(out_path + "_data", 0o644)
    return out_path + "_data"


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def self_check(step, fp32_path, int8_path, int8_data):
    """Fail the export if the graphs drift from torch. Canary against silent breakage."""
    torch.manual_seed(0)
    sessions = [
        (kind, ort.InferenceSession(path, providers=["CPUExecutionProvider"]))
        for kind, path in (("fp32", fp32_path), ("int8", int8_path))
    ]

    audio = step.model.config.audio_config
    mel_bins, hidden, sub = audio.num_mel_bins, audio.hidden_size, audio.subsampling_factor

    cases = [(800, 0), (800, 41), (408, 264), (1600, 41)]
    with torch.no_grad():
        for t_mel, cached in cases:
            x = torch.randn(1, t_mel, mel_bins)
            c = torch.randn(1, cached, hidden)
            mask = torch.ones(1, cached + (t_mel + sub - 1) // sub, dtype=torch.int64)
            ref = step(x, c, mask)
            feeds = {
                "input_features": x.numpy(),
                "cached_embeds": c.numpy(),
                "attention_mask": mask.numpy(),
            }
            for name, sess in sessions:
                outs = sess.run(None, feeds)
                for tensor, got, label in zip(ref, outs, ("logits", "chunk_embeds")):
                    want = tensor.numpy()
                    if list(want.shape) != list(got.shape):
                        raise SystemExit(
                            f"{name} shape mismatch on {label} for T={t_mel} C={cached}: "
                            f"{want.shape} != {got.shape}"
                        )
                    score = cosine(want, got)
                    floor = 0.99999 if name == "fp32" else 0.999
                    if score < floor:
                        raise SystemExit(
                            f"{name} parity check failed on {label} for T={t_mel} C={cached}: "
                            f"cosine={score:.8f} < {floor}"
                        )
                log(f"self-check {name} T={t_mel} C={cached}: parity ok")

    int8_bytes = os.path.getsize(int8_path) + os.path.getsize(int8_data)
    if int8_bytes > 160 << 20:
        raise SystemExit(
            f"int8 graph is {int8_bytes >> 20} MB, expected ~110 MB: stale weights in the external-data file"
        )
    log(f"int8 graph size: {int8_bytes >> 20} MB")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: assets/)")
    parser.add_argument(
        "--checkpoint-dir", default=DEFAULT_CHECKPOINT, help="checkpoint directory (default: checkpoint/)"
    )
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    clear_graphs(out)

    checkpoint = download_checkpoint(os.path.abspath(args.checkpoint_dir))

    for name in ("config.json", "processor_config.json"):
        shutil.copyfile(os.path.join(checkpoint, name), os.path.join(out, name))

    log("loading checkpoint")
    model = AutoModelForAudioFrameClassification.from_pretrained(
        checkpoint, attn_implementation="eager"
    )
    model.eval()
    torch.set_grad_enabled(False)

    write_preprocessor_config(model.config, out)

    fp32_path = os.path.join(out, "model.onnx")
    step = export_fp32(model, fp32_path)
    log(f"fp32 graph: {os.path.getsize(fp32_path) >> 20} MB")

    int8_path = os.path.join(out, "model_quantized.onnx")
    int8_data = quantize_int8(fp32_path, int8_path)

    self_check(step, fp32_path, int8_path, int8_data)

    artifacts = [
        "config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "model.onnx",
        "model_quantized.onnx",
        "model_quantized.onnx_data",
    ]
    with open(os.path.join(out, "manifest.txt"), "w") as f:
        f.write(f"model: {MODEL_ID}\n")
        f.write(f"revision: {MODEL_REVISION}\n")
        f.write(f"opset: {OPSET}\n")
        f.write(f"quantization: matmul_nbits bits={QUANT_BITS} block_size={QUANT_BLOCK_SIZE} symmetric\n")
        f.write(f"fp32 nodes: {', '.join(FP32_NODES)}\n")
        f.write("checkpoint sha256:\n")
        for name in CHECKPOINT_FILES:
            f.write(f"  {CHECKPOINT_SHA256[name]}  {name}\n")
        f.write("artifacts sha256:\n")
        for name in artifacts:
            f.write(f"  {sha256(os.path.join(out, name))}  {name}\n")

    log(f"assets written to {out}")


if __name__ == "__main__":
    main()
