---
license: apache-2.0
library_name: tensorfold
pipeline_tag: image-text-to-text
tags:
  - deepseek
  - deepseek-v4.1
  - exl3
  - tensorfold
  - dgx-spark
  - gb10
  - speculative-decoding
  - tensor-parallel
  - moe
  - vision
---

# DeepSeek-V4.1-Flash on four DGX Spark (GB10) nodes: a TensorFold TP4 engine

Weights: [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
(MIT), an EXL3 quant of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
This repository is a recipe: the served launch, our measurements, the gates the build passed and, in `kit/`, the
kernel kit the engine loads. It holds no weights and no engine source.

The engine is branch `deepseek-v41-zig-tp4` of our TensorFold fork,
[bertholomus/TensorFold](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4). The file
`ENGINE_COMMIT` pins the exact commit. It adds a DeepSeek-V4.1-Flash family to TensorFold 1.0.2's native (Zig) engine
and runs it tensor-parallel over four NVIDIA GB10 nodes, one rank per node, linked by RoCE. It ports our two-node
family ([deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)) and
extends it to an exact two-dimensional split. We wrote it independently and re-implemented the model math from
DeepSeek's MIT inference code and tech report (see [License and attribution](#license-and-attribution)).

What it does:

- Four-node exact split: in our gates its replies equal the two-node family's token for token.
- Image input in user turns and tool results (the vision tower runs on rank 0), as DeepSeek's encoding places it.
- Up to 1,048,576 tokens a request, 16 streams sharing a 2,097,152-position window, with kept-prompt reuse: a request
  that repeats a kept prompt's prefix resumes from it instead of prefilling again.
- Exact DSpark speculative decoding: drafted output equals serial output token for token.
- OpenAI-compatible HTTP API (TensorFold's server): chat and text completions, streaming, reasoning output, tool calls.

## Results

All numbers are ours, measured with our own clients (`kit_bench.py`, `long_check.py` and the gate tools of the engine
branch) on the four nodes. Each table names the engine commit and the one configuration it ran. Greedy decoding,
thinking off, unless a row says otherwise.

### The served build (engine `ad70f4f`, the served settings below)

| Row | Result |
|---|---|
| Decode, one stream, set b (384 tokens, median of 3): code / prose / structured | 160.3 / 95.4 / 219.5 tok/s |
| Two streams together (256 tokens each, mean of 6) | 123.0 tok/s aggregate |
| Four streams together (256 tokens each, mean of 6) | 166.8 tok/s aggregate |
| Prefill, 8,192-token prompt (median of 6) | 2,843 tok/s |

### Long context and kept prompts (engine `ad70f4f`, the served settings below)

| Check | Result |
|---|---|
| Four 499,931-token prompts and a repeat of the first, sent together, then each alone | Burst replies equal the solo replies 5/5; needles found 4/4 in both; drafted equals serial; burst 895.9 s; every solo rerun resumes from its kept prompt (499,712 tokens cached), 5.8 s for all five |
| The same with 159,925-token prompts, right after on the same server | Equal 5/5; needles 4/4; burst 244.6 s; every solo rerun resumes (159,744 cached), 4.1 s for all five |
| 299 prompt lengths from 1 to 20,483 tokens | 0 errors; every reply equal to the reference lane's |
| Lowest free memory a node during the 500K check | 50 / 51 / 52 / 52 GiB |

### Throughput rows (engine `1ca1c10`: the same model code without the kept-prompt scheduling commits; a test server at 1,048,576 tokens a request, 48-row rounds, 16 streams, no shared window)

| Row | Result |
|---|---|
| Decode, one stream, set a (512 tokens, median of 3): code / prose / structured | 138.1 / 87.2 / 165.6 tok/s |
| Decode, one stream, set b (384 tokens): code / prose / structured | 158.8 / 95.5 / 220.2 tok/s |
| 2 / 4 streams together (256 tokens each) | 121.6 / 166.6 tok/s aggregate |
| 8 / 16 streams, sustained for 90 s (256 tokens each) | 272.0 / 308.4 tok/s aggregate |
| Prefill 8K / 32K / 128K tokens (median of 3) | 2,778 / 2,900 / 2,744 tok/s |

### Gates

On the served build (`ad70f4f`):

- The lane gates (12/12, 7/7, 19/19 over NCCL, 19/19 over the RDMA rings) and the server gate (3/3), each against
  recorded replies of the reference lane.
- Promotion gates on the served start: 6 of 6 API gates; 16 of 16 replies equal to the two-node reference; burst and
  staggered requests equal to solo 16/16.
- Vision: the image gate 6/6; 9 of 9 image prompts with token ids equal to the reference lane's; the same 9 four at a
  time, twice, 18/18 equal (concurrent equals solo).

On `1ca1c10` (the same model code): the layer gate, all points equal on 4 of 4 nodes over NCCL and over the RDMA
rings; 48 of 48 image probes (21 formats, message layouts, refusals and their words) equal to the reference lane.

### Run-time inputs from public weights (engine `5940208`, v0.1.1)

| Check | Result |
|---|---|
| Each node's weight file and token map from the checkpoint, the four nodes at once (`scripts/serve.sh inputs`) | 109 s; all four weight files (56.2 GB each) and the token map byte for byte the ones our served lane loads |
| The kit from a fresh clone of this repository (`kit/verify_kit.sh`) | 356 of 356 files match `kit/MANIFEST`; the four RoPE tables made in 11 s on the CPU |
| The recipe's launch from those inputs and the kit (`scripts/serve.sh start`) | 6 of 6 API gates; 16 of 16 replies equal to the two-node reference; burst and staggered equal solo 16/16; the image gate 6/6, 9/9 and 18/18 |

`results/` holds the raw outputs these tables come from.

## The served settings

Rank 0 runs `tensorfold-native serve` with `TF_TP_WORLD=4`; ranks 1-3 run `tf-dsv41-lanes`, which follows it. The
settings as served: `--context 1048576 --parallel 16`, `TF_DS_WINDOW=2097152` (the streams' shared positions),
`TF_DS_ROUND_ROWS=48`, `TF_DS_STREAMS=16`, `TF_DS_2D_GU=parity` (the balanced experts split), `TF_DS_SERVED_K=solo`,
the measured round costs in `TF_DS_ROUND_MS`, CUDA graphs, the side stream, the paced L2 prefetch, Engram reads by AIO,
and the decode-size exchanges over the RDMA rings. `scripts/serve.sh` spells the launch out; `config.env.example`
holds the site settings.

## Requirements

- Four NVIDIA GB10 nodes on one RoCE network, each with the NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`
  (Pillow 12.3.0 inside it decodes the image formats other than PNG).
- The weights: [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw),
  and DeepSeek's original shards 47 and 48 for the Engram tables, on every node.
- The engine built from `ENGINE_COMMIT` (`zig build` with the CUDA toolchain; see `RUNNING.md`).
- The run-time inputs, made from the public weights (see `RUNNING.md`): each node's weight file and the compressed
  token map, one command a node (`node_inputs.sh` of the engine branch; `scripts/serve.sh inputs` runs it on all four),
  and the kernel kit in `kit/`, whose RoPE tables and image bias `kit/verify_kit.sh` makes before it checks every file
  against `kit/MANIFEST`. The `kit/` folder of
  [bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10/tree/main/kit) holds the
  whole kit as a download.

## License and attribution

This recipe is licensed under Apache-2.0 (`LICENSE`). The engine branch keeps TensorFold's Apache-2.0 license, lists
the upstream files it changes in its `NOTICE` and every outside source in its `ATTRIBUTION.md`. The weights are MIT
(DeepSeek; the EXL3 quant by Mia-AiLab inherits it). `kit/` holds compiled kernels: our own (Apache-2.0) and
PyTorch's attention kernel for the vision tower, unmodified (BSD-3-Clause); `kit/KIT-LICENSES.md` gives each file's
source and license. `ATTRIBUTION.md` here lists what this recipe uses. "DeepSeek"
belongs to DeepSeek and "DGX Spark" and "GB10" to NVIDIA; this recipe is not affiliated with or endorsed by DeepSeek,
NVIDIA, the TensorFold authors or Mia-AiLab.
