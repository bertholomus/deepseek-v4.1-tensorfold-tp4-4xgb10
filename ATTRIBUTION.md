# Attribution

This recipe and the engine branch behind it (`deepseek-v41-zig-tp4` of our TensorFold fork) were written
independently. They are clean-room with respect to other people's DeepSeek-V4.1 work: the jayleaton recipe and
upstream TensorFold's DeepSeek-V4.1 pull requests were never opened. The model math was re-implemented from
DeepSeek's MIT inference code and tech report; no code was copied. The engine branch's own `ATTRIBUTION.md` lists
every outside source of the engine; this file lists what the recipe uses.

| Source | License | What we used | How |
|---|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | MIT, Copyright (c) 2023 DeepSeek | The model; its original shards 47 and 48 (Engram tables); its 43 `ffn.gate.bias_vl` tensors; the rules of its reference code for the Engram constants (`kit/engram.json`) and the RoPE rotation (`kit/tools/make_rope.py`) | Shards loaded at run time, not redistributed. The bias tensors are copied byte for byte into `kit/vision/gate_bias_vl.safetensors` by `kit/tools/make_bias_vl.py`; git does not hold that file. Rules re-implemented; no code copied |
| [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | MIT, inherited from the base model | The weights we serve | Loaded at run time. Not redistributed |
| [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2 | Apache-2.0, Copyright 2026 TensorFold contributors | The serving engine and its OpenAI-compatible server | Forked: [bertholomus/TensorFold `deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4) |
| Our two-node recipe, [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) | Apache-2.0, same authors | The benchmark and gate clients in `tools/` and the layout of this recipe | Ours, reused |
| NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` (PyTorch BSD-3-Clause, Triton MIT, NCCL, Pillow MIT-CMU) | NVIDIA's container license and the components' own licenses | The run-time environment; its PyTorch on the CPU for the RoPE tables (`kit/verify_kit.sh`) | Used as supplied. Not redistributed, except the kernel below |
| [PyTorch](https://github.com/pytorch/pytorch) and [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) | BSD-3-Clause | PyTorch's memory-efficient attention kernel for the vision tower (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`, built on CUTLASS) | Extracted unmodified from the container's `libtorch_cuda.so`; redistributed in binary form as `kit/vision/torch_fmha_sm120.cubin` with the license texts in `kit/LICENSES/` |
| [Triton](https://github.com/triton-lang/triton) | MIT | The compiler of our kernels in `kit/aot/` | Compiled with; license text in `kit/LICENSES/` |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | MIT, Copyright (c) 2025 Turboderp | The EXL3 format the kernels in `kit/cubins/` read (through TensorFold's EXL3 module) | Format followed; license text in `kit/LICENSES/` |

## About the files in `results/`

All files in `results/` are our own measurements of our own engine, with private host names, addresses and paths
removed. Nothing else in them was changed.

## Names

- "DeepSeek" and "DeepSeek-V4.1-Flash" belong to DeepSeek. "DGX Spark" and "GB10" belong to NVIDIA. We use the names
  only to say what this recipe runs and on what hardware.
- This recipe is not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors or Mia-AiLab.
