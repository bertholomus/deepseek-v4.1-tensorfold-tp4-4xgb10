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

# DeepSeek-V4.1-Flash-1017 on four DGX Spark (GB10) nodes: a TensorFold TP4 engine

Weights: [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw)
(MIT), our EXL3 quant of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(MIT). This repository is a recipe: the served launch, our measurements, the gates the build passed, the chat
template, the step that readies the downloaded weights and, in `kit/`, the kernel kit the engine loads. It holds no
weights and no engine source.

The engine is branch `deepseek-v41-zig-tp4` of our TensorFold fork,
[bertholomus/TensorFold](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4). The file
`ENGINE_COMMIT` pins the exact commit. It adds a DeepSeek-V4.1-Flash family to TensorFold 1.0.12's native (Zig) engine
and runs it tensor-parallel over four NVIDIA GB10 nodes, one rank per node, linked by RoCE. It ports our two-node
family ([deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)) and
extends it to an exact two-dimensional split. We wrote it independently and re-implemented the model math from
DeepSeek's MIT inference code and tech report (see [License and attribution](#license-and-attribution)).

What it does:

- Four-node exact split: on the 2.9 bpw weights of v0.1.1 its replies equal the two-node family's token for token in
  our gates; on the 1017 weights, burst and staggered requests equal the solo replies 16 of 16.
- Image input in user turns and tool results (the vision tower runs on rank 0), as DeepSeek's encoding places it.
- Up to 1,048,576 tokens a request, 16 streams sharing a 2,097,152-position window, with kept-prompt reuse: a request
  that repeats a kept prompt's prefix resumes from it instead of prefilling again.
- Exact DSpark speculative decoding: drafted output equals serial output token for token.
- OpenAI-compatible HTTP API (TensorFold's server): chat and text completions, streaming, reasoning output, tool calls.

## v0.2.0: what changed

| | v0.1.1 | v0.2.0 |
|---|---|---|
| Weights | [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw): DeepSeek-V4.1-Flash, 2.9 bpw | [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw): DeepSeek-V4.1-Flash-1017, 4.02 bpw |
| Engine | `5940208`, on TensorFold 1.0.2 | `b65c8b9`, on TensorFold 1.0.12 |
| Chat template | the checkpoint's own | ours, `chat/chat_template.jinja` (the 1017 checkpoint ships none): 71 of 71 chats tokenize to the same ids as DeepSeek's reference encoding |
| Readying the weights | none | `tools/serving_1017.py`: six tensors under the names and types the engine reads, and the template |
| Kernel kit | 356 files | 358 files: the 1017 checkpoint's image routing bias and two more Triton kernel variants (an indexer score and a sparse attention part, both byte for byte our two-node kit's) |
| Decode, one stream, set b: code / prose / structured | 160.3 / 95.4 / 219.5 tok/s | 135.9 / 79.1 / 186.6 tok/s (-15.2 / -17.1 / -15.0 %) |
| Two / four streams together | 123.0 / 166.8 tok/s | 101.6 / 141.7 tok/s (-17.4 / -15.0 %) |
| Prefill, 8,192-token prompt | 2,843 tok/s | 2,793 tok/s (-1.8 %) |
| Longest request checked | 499,931 tokens: needles 4/4 | 999,843 tokens: needles 3/3 |
| Lowest free memory a node | 50 / 51 / 52 / 52 GiB (the 500K check) | 28 / 33 / 33 / 33 GiB (the 1M and 160K checks) |

The 4.02 bpw weights decode 15 to 17 % slower than the 2.9 bpw ones; prefill is level. The served settings did not
change, and the model id stays `DeepSeek-V4.1-Flash-TP4`.

Engine changes since v0.1.1 (the fork's v0.2.0 merge commit lists them):

- TensorFold 1.0.12 under the family. Two of its reply changes reach clients: with tools offered, a reply with tool
  calls keeps its prose in `message.content` (it was null), and the `tensorfold` block's `runtime.seconds` is the
  decode's seconds.
- Each rank checks its settings against rank 0's at the start. A launch with no captured Triton variant for a prompt
  length runs the closest more general one. The image decoder's child process starts again if it dies. A request the
  pool cannot hold yet waits for a stream to end.
- From contributors: the prompt indexer passes each row block 16-byte aligned, so every prompt length answers (pull
  request #4, Lucas Peters); the loader reads each layer's expert run from the weight file whole, a faster load (pull
  request #18, lepetitprince716-prog).
- Off unless set: the stream keepalive (`TENSORFOLD_SSE_KEEPALIVE`), the loop guard's novelty signal (it acts only
  under `--loop-guard`; after a report by @Capicua25x) and `TF_DS_IDX_SPLIT` (the keyed indexer layers' long rows
  split by columns over the nodes).

## Results

All numbers are ours, measured with our own clients (`kit_bench.py`, `long_check.py` and the gate tools of the engine
branch) on the four nodes, on the served launch: engine `5c97635`, the 1017 weights, this kit and the served settings
below. `ENGINE_COMMIT` (`b65c8b9`) is that engine's source with the fork's documents, three comment lines and two test
lines changed, and it builds the same binaries but for nvcc's build ids (see `RUNNING.md`). Greedy decoding, thinking
off, unless a row says otherwise. `results/served-1017/` holds the raw outputs.

### Speed

| Row | Result |
|---|---|
| Decode, one stream, set b (384 tokens, median of 3): code / prose / structured | 135.9 / 79.1 / 186.6 tok/s |
| Two streams together (256 tokens each, mean of 6) | 101.6 tok/s aggregate |
| Four streams together (256 tokens each, mean of 6) | 141.7 tok/s aggregate |
| Four streams sustained for 90 s (256 tokens each) | 183.1 tok/s aggregate |
| Prefill, 8,192-token prompt (median of 3) | 2,793 tok/s |
| Prefill, 131,072-token prompt (median of 2) | 2,679 tok/s |
| Decode after a 131,072-token prompt (256 tokens) | 140.1 tok/s |

### Long context and kept prompts

| Check | Result |
|---|---|
| Three needles in one shared 999,843-token haystack, at depths 0.1, 0.5 and 0.9 | Found 3/3; the first request prefills it (543.5 s), the next two resume from the kept prompt (999,424 tokens cached) in 2.4 and 2.2 s |
| Four prompts of about 160,000 tokens and a repeat of the first, sent together, then each alone | Burst replies equal the solo replies 5/5; needles found 4/4 in both; drafted equals serial; burst 250.3 s; every solo rerun resumes from its kept prompt (159,744 tokens cached), 4.8 s for all five |
| Lowest free memory a node during these two checks | 28 / 33 / 33 / 33 GiB |

### Gates

- Exact chat: 71 of 71 chats (system and developer prompts, tools and tool calls, tool results, reasoning and its
  effort, multi-turn, response formats) give the same token ids through the server's `/tokenize` as DeepSeek's reference encoding
  (`encoding.py`, MIT) gives for them, on the gate launch and again on the served port.
- The API gates 6 of 6: the model list, a chat probe, a streamed reasoning, a tool call, drafted equals serial, and a
  burst of four.
- 16 replies sent as a burst and staggered: equal to their solo replies 16/16 both ways.
- Vision: the image gate 6/6; 9 image prompts' token ids, then the same 9 four at a time, twice: 18/18 equal to the
  solo run.

### Run-time inputs from the public weights

| Check | Result |
|---|---|
| The download of [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw) (revision `a2f91e27`) against the weights the served lane loads | All 41 shards, the index and the configs byte for byte our copy of the converter's output; 192,492 of its 192,496 tensors equal (sha256 per tensor) to the served lane's, and the other four are the Engram q/k that the served lane holds widened to F32 |
| `tools/serving_1017.py` on that download | Its six tensors equal the served lane's (name, type, shape, sha256); `serving-1017.safetensors` sha256 `d88fe41f...` and `chat_template.jinja` `b581d19d...` (`results/served-1017/inputs.txt`) |
| Each node's weight file from that download (`tf-dsv41-rank-cache --parity --index-only`, four ranks) | The four indexes hash to the names of the served lane's four weight files; without `serving-1017.safetensors` the writer stops on a missing tensor |
| The kit from a fresh clone of this repository (`kit/verify_kit.sh`) | 358 of 358 files match `kit/MANIFEST` |

### Earlier releases

v0.1.0 and v0.1.1 served the 2.9 bpw weights on engine `ad70f4f` and `5940208`. Their tables are in the v0.1.1
README (tag `v0.1.1`), and their raw outputs stay in `results/` (`served-ad70f4f/`, `inputs-5940208/`,
`throughput-1ca1c10/`, `probes-1ca1c10/`).

## The served settings

Rank 0 runs `tensorfold-native serve` with `TF_TP_WORLD=4`; ranks 1-3 run `tf-dsv41-lanes`, which follows it. The
settings as served: `--context 1048576 --parallel 16`, `TF_DS_WINDOW=2097152` (the streams' shared positions),
`TF_DS_ROUND_ROWS=48`, `TF_DS_STREAMS=16`, `TF_DS_2D_GU=parity` (the balanced experts split), `TF_DS_SERVED_K=solo`,
the measured round costs in `TF_DS_ROUND_MS`, CUDA graphs, the side stream, the paced L2 prefetch, Engram reads by AIO,
and the decode-size exchanges over the RDMA rings. They are v0.1.1's, unchanged. The round costs are the table
measured on the 2.9 bpw weights: they guide how many tokens a round drafts, and drafted output equals serial output, so
no reply depends on them. `scripts/serve.sh` spells the launch out; `config.env.example` holds the site settings.

## Requirements

- Four NVIDIA GB10 nodes on one RoCE network, each with the NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`
  (Pillow 12.3.0 inside it decodes the image formats other than PNG).
- The weights: [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw)
  (about 275 GB), readied by `tools/serving_1017.py`, and the original checkpoint's shards 47 and 48 for the Engram
  tables, on every node.
- The engine built from `ENGINE_COMMIT` (`zig build` with the CUDA toolchain; see `RUNNING.md`).
- The run-time inputs, made from the public weights (see `RUNNING.md`): each node's weight file and the compressed
  token map, one command a node (`node_inputs.sh` of the engine branch; `scripts/serve.sh inputs` runs it on all four),
  and the kernel kit in `kit/`, whose RoPE tables and image bias `kit/verify_kit.sh` makes before it checks every file
  against `kit/MANIFEST`.

## License and attribution

This recipe is licensed under Apache-2.0 (`LICENSE`). The engine branch keeps TensorFold's Apache-2.0 license, lists
the upstream files it changes in its `NOTICE` and every outside source in its `ATTRIBUTION.md`. The weights are MIT
(DeepSeek; our EXL3 quant inherits it). The chat template is ours, written to render what DeepSeek's reference
encoding (MIT) renders. `kit/` holds compiled kernels: our own (Apache-2.0) and PyTorch's attention kernel for the
vision tower, unmodified (BSD-3-Clause); `kit/KIT-LICENSES.md` gives each file's source and license. `ATTRIBUTION.md`
here lists what this recipe uses. "DeepSeek" belongs to DeepSeek and "DGX Spark" and "GB10" to NVIDIA; this recipe is
not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors or Mia-AiLab.
