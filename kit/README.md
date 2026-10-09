# Kernel kit: DeepSeek-V4.1-Flash on TensorFold 1.0 (Zig), four GB10 nodes

This folder is the kit our served four-node lane loads (engine `ad70f4f`). `MANIFEST` lists all 356 files with their
sizes and sha256. 355 are byte for byte the served copies on our four nodes; `aot/aot.json` differs only in its
packer's note of where one kernel set came from, which named a private folder (the engine does not read the note, and
the kernels it lists are the same). Git holds 351 of the files (138 MB). The other five are made from public weights
instead of committed:

- the four RoPE tables `rope-{plain,compressed}-{cos,sin}.f32` (142,606,336 bytes each), which `tools/make_rope.py`
  computes on the CPU from the checkpoint's `config.json`;
- `vision/gate_bias_vl.safetensors` (67,192 bytes), the MoE gates' routing bias for image tokens: 43 small f32
  tensors of DeepSeek's weights, which `tools/make_bias_vl.py` copies out of DeepSeek's original checkpoint.

The `kit/` folder of our Hugging Face repo
[bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10/tree/main/kit)
holds the same files, those five included, for a download instead of a build.

## Make and check it

```sh
kit/verify_kit.sh <EXL3_MODEL_DIR> [<ORIGINAL_CHECKPOINT_DIR>]
```

This makes the RoPE tables (PyTorch on the CPU; without a local `torch` it runs with docker in
`nvcr.io/nvidia/pytorch:26.07-py3`, whose PyTorch made the served tables), and the bias file when you pass DeepSeek's
original checkpoint ([deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)). Then
it checks every file against `MANIFEST` and ends with `verify_kit: 356 of 356 files match MANIFEST` and exit 0. The
bias tensors sit in 43 of the original checkpoint's 48 shards, so without a local copy of it take
`vision/gate_bias_vl.safetensors` from the Hugging Face `kit/` instead (`MANIFEST` has its sha256). The EXL3 model
folder only has to hold `config.json`. `sha256sum -c SHA256SUMS` checks the same list.

## Use it

Point every rank at this folder (the recipe's `KIT_DIR`):

- rank 0 (`tensorfold-native serve`): `TF_DS_KIT=<KIT_DIR>` and `TENSORFOLD_CUDA_KERNELS=<KIT_DIR>/aot`;
- ranks 1-3 (`tf-dsv41-lanes`): `<KIT_DIR>` as the kit argument.

The kernels are built for the GB10 (sm_121) and the `nvcr.io/nvidia/pytorch:26.07-py3` container. The server of this
vision checkpoint refuses to start without `vision/`.

## What is in it

`KIT-LICENSES.md` gives each file's source and license; the license texts are in `LICENSES/`. In short: our Triton
kernels as binaries (`aot/`), the EXL3 kernels (`cubins/`), Engram's hash constants (`engram.json`), the RoPE tables
and their digests (`rope.json`), PyTorch's attention kernel for the vision tower (`vision/torch_fmha_sm120.cubin`) and
the image bias. 189 of the 356 files are byte for byte the same as in our two-node kit v0.6.0 (`MANIFEST` marks them
`tp2`); the four-node lane adds 162 Triton kernel binaries, its own `aot.json` and its EXL3 kernels. The model weights
are not here: serve [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw).

The engine branch carries the tools that made this folder (`tools/dsv41-zig/kit/`): `pack_kit.py`, which copies a
lane's kit and writes `MANIFEST` and `SHA256SUMS`, the two generators and `verify_kit.sh`.
