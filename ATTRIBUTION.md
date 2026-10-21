# Attribution

This recipe and the engine branch behind it (`deepseek-v41-zig-tp4` of our TensorFold fork) were written
independently. They are clean-room with respect to other people's DeepSeek-V4.1 work: the jayleaton recipe and
upstream TensorFold's DeepSeek-V4.1 pull requests were never opened. The model math was re-implemented from
DeepSeek's MIT inference code and tech report; no code was copied. The engine branch's own `ATTRIBUTION.md` lists
every outside source of the engine; this file lists what the recipe uses.

| Source | License | What we used | How |
|---|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) and [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | MIT, Copyright (c) 2023 DeepSeek | The model; the 1017 checkpoint's original shards 47 and 48 (Engram tables) and its 43 `ffn.gate.bias_vl` tensors (V4.1-Flash's up to v0.1.1); the rules of its reference code for the Engram constants (`kit/engram.json`) and the RoPE rotation (`kit/tools/make_rope.py`); its reference encoding (`encoding.py`), which our chat template renders | Shards loaded at run time, not redistributed. The bias tensors are copied byte for byte into `kit/vision/gate_bias_vl.safetensors` by `kit/tools/make_bias_vl.py`; git does not hold that file. Rules and the chat format re-implemented (`chat/chat_template.jinja` is ours); no code copied |
| [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw) | MIT, inherited from the base model | The weights we serve since v0.2.0 (our EXL3 quant, made with ExLlamaV3's converter) | Loaded at run time. Not redistributed. `tools/serving_1017.py` adds six of its tensors under the engine's names and types to a download |
| [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | MIT, inherited from the base model | The weights we served up to v0.1.1 | Loaded at run time. Not redistributed |
| [TensorFold](https://github.com/ashhart/TensorFold) 1.0.12 (1.0.2 up to v0.1.1) | Apache-2.0 from TensorFold 0.6.0, Copyright 2026 TensorFold contributors; code written before 0.6.0 keeps its MIT notice (`kit/LICENSES/TensorFold-MIT.txt`) | The serving engine and its OpenAI-compatible server; compiled in `kit/`: its EXL3 kernels (`kit/cubins/`, with our additions) and its grammar shim (`kit/grammar/libtfgrammar.so`) | Forked: [bertholomus/TensorFold `deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4) |
| Our two-node recipe, [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) | Apache-2.0, same authors | The benchmark and gate clients in `tools/` and the layout of this recipe | Ours, reused |
| NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` (PyTorch BSD-3-Clause, Triton MIT, NCCL, Pillow MIT-CMU) | NVIDIA's container license and the components' own licenses | The run-time environment; its PyTorch on the CPU for the RoPE tables (`kit/verify_kit.sh`) | Used as supplied. Not redistributed, except the kernel below |
| [PyTorch](https://github.com/pytorch/pytorch) and [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) | BSD-3-Clause | PyTorch's memory-efficient attention kernel for the vision tower (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`, built on CUTLASS) | Extracted unmodified from the container's `libtorch_cuda.so`; redistributed in binary form as `kit/vision/torch_fmha_sm120.cubin` with the license texts in `kit/LICENSES/` |
| [Triton](https://github.com/triton-lang/triton) | MIT | The compiler of our kernels in `kit/aot/` | Compiled with; license text in `kit/LICENSES/` |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | MIT, Copyright (c) 2025 Turboderp | The EXL3 format the kernels in `kit/cubins/` read (through TensorFold's EXL3 module); its converter made our quant | Format followed; license text in `kit/LICENSES/` |
| [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8, with the [dlpack](https://github.com/dmlc/dlpack) and [picojson](https://github.com/kazuho/picojson) headers it bundles | Apache-2.0, Copyright (c) 2024 by XGrammar Contributors (xgrammar); Apache-2.0 (dlpack); BSD-2-Clause, Copyright 2009-2010 Cybozu Labs, Inc. and 2011-2014 Kazuho Oku (picojson) | The grammar library behind structured output and forced tool calls | Compiled unmodified with TensorFold 1.0.12's shim into `kit/grammar/libtfgrammar.so`; redistributed in binary form, with xgrammar's license and notice and dlpack's license in `kit/grammar/` and picojson's in `kit/LICENSES/` |

## Contributors

Since v0.1.1 the engine branch carries two pull requests to our fork: [#4](https://github.com/bertholomus/TensorFold/pull/4)
by Lucas Peters ([@lucas-peters](https://github.com/lucas-peters)), the prompt indexer's aligned row blocks, and
[#18](https://github.com/bertholomus/TensorFold/pull/18) by [@lepetitprince716-prog](https://github.com/lepetitprince716-prog),
the faster load. Its loop guard's novelty signal follows a report by [@Capicua25x](https://github.com/Capicua25x)
([two-node recipe issue #18](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/18)). Thank
you. The engine branch's `ATTRIBUTION.md` credits each of them.

Since v0.2.0 the engine branch carries five more contributions:
[#26](https://github.com/bertholomus/TensorFold/pull/26) by Manrique ([@hashd1ve](https://github.com/hashd1ve)),
which keeps a conversation's prompt states when another one shares its prefix, and ports of Manrique's
[#28](https://github.com/bertholomus/TensorFold/pull/28) (prompts with images keep their states) and
[#29](https://github.com/bertholomus/TensorFold/pull/29) (`TF_FILL_CHUNKS`, off unless set) from the two-node
branch, whose commits name Manrique as co-author;
[#30](https://github.com/bertholomus/TensorFold/pull/30) by Scott Woolsey Biggart
([@swbiggart](https://github.com/swbiggart)), the neighbor transport (off unless set); and
[#31](https://github.com/bertholomus/TensorFold/pull/31) by [@mrsergiopascal](https://github.com/mrsergiopascal), a
test and a reproducer of the replay cut's positions. Its stream keepalive follows a report by
[@den13501](https://github.com/den13501) ([two-node recipe issue #12](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/12)).
Thank you.

Our commits to this recipe, and many of ours to the engine branch, name Claude (Anthropic's AI assistant), which
helped write them, in a `Co-Authored-By` trailer.

## About the files in `results/`

All files in `results/` are our own measurements of our own engine, with private host names, addresses and paths
removed. Nothing else in them was changed, except where `results/README.md` says a file holds only a part of a
check's output.

## Names

- "DeepSeek", "DeepSeek-V4.1-Flash" and "DeepSeek-V4.1-Flash-1017" belong to DeepSeek. "DGX Spark" and "GB10" belong to NVIDIA. We use the names
  only to say what this recipe runs and on what hardware.
- This recipe is not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors or Mia-AiLab.
