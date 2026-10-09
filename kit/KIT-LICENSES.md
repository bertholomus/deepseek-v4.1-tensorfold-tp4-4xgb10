# Kit licenses

Every file in this kit, where it comes from and under which license. The license texts are in `LICENSES/`. `MANIFEST`
lists each file's size and sha256, says which files git holds and which `verify_kit.sh` makes, and marks the files that
are byte for byte the same as in our two-node kit v0.6.0 (the same `dsv41` family). `SHA256SUMS` is the same list for
`sha256sum -c`.

| Files | What they are | Source | License |
|---|---|---|---|
| `aot/aot.json`, `aot/cubins/*` | The `dsv41` family's Triton kernels, compiled for sm_121 and packed for TensorFold 1.0's Triton AOT loader: the two-node set plus the four-node lane's specializations of the 2D shapes and of every prompt-length class it serves | Kernels written by Bertholomus AI (github.com/bertholomus/TensorFold, branch `deepseek-v41-tp2`), compiled by Triton; packed with TensorFold 1.0's `aot_pack.py` | Apache-2.0 (`LICENSES/Apache-2.0.txt`); Triton, which generated the binaries: MIT (`LICENSES/Triton-MIT.txt`) |
| `cubins/experts.cubin`, `experts_cb.cubin`, `linear.cubin`, `linear_grouped.cubin` | EXL3 linear and expert kernels, sm_121, as the four-node lane loads them | TensorFold's EXL3 module (TensorFold contributors) with Bertholomus AI's additions | Apache-2.0 (`LICENSES/Apache-2.0.txt`); the EXL3 format follows ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp; `LICENSES/ExLlamaV3-MIT.txt`) |
| `engram.json` | Engram's bucket primes, offsets, multipliers and hash test cases | Computed by our Python family from the rules of DeepSeek's reference code | DeepSeek-V4.1-Flash code: MIT, Copyright (c) 2023 DeepSeek (`LICENSES/DeepSeek-MIT.txt`) |
| `rope.json`; made here: `rope-plain-cos.f32`, `rope-plain-sin.f32`, `rope-compressed-cos.f32`, `rope-compressed-sin.f32` | fp32 RoPE tables, 1,114,112 rows, and their digests | Computed on the CPU by `tools/make_rope.py` (our Python family's `ops.rope_cs`, whose rotation follows DeepSeek's reference `precompute_freqs_cis` with YaRN) | Apache-2.0 (`LICENSES/Apache-2.0.txt`); DeepSeek's reference code: MIT (`LICENSES/DeepSeek-MIT.txt`) |
| `vision/torch_fmha_sm120.cubin` | PyTorch's memory-efficient attention kernel (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`), sm_120 binary, unmodified | Extracted from `libtorch_cuda.so` in `nvcr.io/nvidia/pytorch:26.07-py3` | PyTorch: BSD-3-Clause (`LICENSES/PyTorch-BSD-3-Clause.txt`); built on NVIDIA CUTLASS: BSD-3-Clause, Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES (`LICENSES/CUTLASS-BSD-3-Clause.txt`) |
| Made here: `vision/gate_bias_vl.safetensors` | The 43 `*.ffn.gate.bias_vl` f32 tensors (the MoE gates' routing bias for image-span tokens), byte for byte | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) weights (the EXL3 checkpoint leaves them out); `tools/make_bias_vl.py` copies them out of the original checkpoint. Model weights: not in git | MIT, Copyright (c) 2023 DeepSeek (`LICENSES/DeepSeek-MIT.txt`) |
| `tools/*`, `verify_kit.sh`, `README.md`, `KIT-LICENSES.md`, `MANIFEST`, `SHA256SUMS` | The kit's scripts and documents | Bertholomus AI | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |

No part of this kit comes from the MiaAI-Lab vLLM kit or from other DeepSeek-V4.1 recipes. "DeepSeek" belongs to
DeepSeek; "DGX Spark" and "GB10" belong to NVIDIA; "PyTorch" belongs to the PyTorch Foundation. This kit is not
affiliated with or endorsed by any of them, the Triton authors, the TensorFold authors or Mia-AiLab.
