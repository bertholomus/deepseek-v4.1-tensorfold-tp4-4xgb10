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
- Up to 1,048,576 tokens a request, 16 streams sharing a 20,971,520-position window, with kept-prompt reuse: a request
  that repeats a kept prompt's prefix, images included, resumes from it instead of prefilling again.
- Structured output: `response_format` (a JSON schema or a JSON object), `guided_choice` and `guided_regex` hold a
  reply to its grammar.
- Tool calls in DeepSeek's DSML; with `tool_choice` "required" or a named function, the reply is one calls block, each
  call an offered tool with its parameters within their schema.
- Exact DSpark speculative decoding: drafted output equals serial output token for token.
- OpenAI-compatible HTTP API (TensorFold's server): chat and text completions, streaming, reasoning output, tool calls.

## v0.3.3: what changed

| | v0.3.2 | v0.3.3 | Measured in |
|---|---|---|---|
| Engine | `51b0c63`, on TensorFold 1.0.12 | `b116c47`, on TensorFold 1.0.12 | |
| Kernel kit | 362 files | the same 362 files | |
| The streams' shared window, the KV cache pool (`TF_DS_WINDOW`; the followers' `--pool`) | 20,971,520 positions
  | the same 20,971,520 positions | |
| A short chat behind a cold 93,335-token prefill, time to first token (median of 3) | 0.76 s (the gate was 1.9
  s) | 0.68 s (the gate is 0.89 s) | `zyx` |
| The long prompt's time to first token, the same session | 34.04 s | 32.00 s | `zyx` |
| Prefill, a 93,335-token prompt (median of 3, `cmp1021_bench.py`) | 2,742 tok/s, derived from the long-prompt
  rows (the gate ran no 93,335-token prefill bench on it) | 2,949 tok/s | `row34` |
| Decode, one stream, set b: code / prose / structured | 135.8 / 78.3 / 183.7 tok/s | 135.6 / 78.6 / 183.7 tok/s
  (level) | `zyx` |
| Two / four streams together | 100.6 / 140.3 tok/s | 100.4 / 139.9 tok/s (-0.2 / -0.3 %) | `zyx` |
| Eight / sixteen streams sustained for 90 s | 216.7 / 256.5 tok/s | 217.6 / 255.6 tok/s (+0.4 / -0.4 %) | `zyx`
  |
| Prefill, 8,192 / 32,768 / 131,072-token prompts | 2,767 / 2,915 / 2,920 tok/s | 2,974 / 3,107 / 3,140 tok/s
  (+7.5 / +6.6 / +7.5 %) | `zyx` |
| Lowest free memory a node during the speed rows | 11 / 13 / 13 / 13 GiB | 11 / 13 / 13 / 14 GiB | `zyx` |

Measured in: each row's numbers come from the gate slot its "Measured in" column names, on our four nodes; their
  raw
outputs are in `results/served-v033/`. `zyx` is the gate run of the build v0.3.3 serves (engine `39155041`,
the served settings; the slot's phase c-v033). In its comparison rows the v0.3.2 column holds the build served
  before
it's, measured in the same session (`zyx`'s phase a). Against that build in the same session, v0.3.3's build
  moved
  its
16 speed rows by -0.4 % (set a prose) to +7.5 % (prefill 128K); the decode rows moved
  -0.2 / +0.3 / 0.0 %, inside the run-to-run spread — decode is unchanged.

The 93,335-token prefill row: v0.3.3's 2,949 tok/s is the median of three runs of `cmp1021_bench.py prefill 3`
  on
the 93,335-token prompt on the served build, client-only on the served port after the promotion (its time to
  first
  token 31.65 s; `row34`). v0.3.1's README row, 2,552 tok/s, is the same bench on the same prompt
  (`cmpp25`): **++15.6 % since v0.3.1**. The same session's build served before it read 2,917 tok/s,
  derived from its three long-prompt time-to-first-token rows (the gate ran no 93,335-token prefill bench in the
  slot; +6.4 % against v0.3.3's bench number, and the 8,192 / 32,768 / 131,072-token rows above are
  the like-for-like session comparison). v0.3.2's own README row was 2,757 tok/s, derived the same way in its
  slot.

Against v0.3.1's README rows (its gate run `z3334`, engine `37df889`): the 93,335-token prefill 2,552 →
  2,949 tok/s (++15.6 %, the same bench both times); prefill 8,192 / 32,768 / 131,072 tokens
  2,770 / 2,833 / 2,779 → 2,974 / 3,107 / 3,140 tok/s (+7.3 / +9.7 / +13.0 %, the same row method both slots);
  set b's decode
  136.7 / 78.6 / 185.3 → 135.6 / 78.6 / 183.7 tok/s (-0.8 / 0.0 / -0.9 %, level — decode is unchanged since
  v0.3.1). The 300K
  needle: three runs (seeds 101, 102, 103) of a 299,865-token prompt with 3 needles each, cached 0 every
  run, all 3 found in every run, 98.65 s median against v0.3.2's 106.59 s on the same check
  (-7.4 % time; `row34`).

Memory: the window is v0.3.2's 20,971,520 positions and stays there; the nodes' lowest free memory in `zyx`'s
  speed
  rows was 11 / 13 / 13 / 14 GiB (the floor is 9 GiB).

Engine changes since v0.3.2 (one commit; the fork's v0.3.3 commit is it):

- `TF_DS_2D_YX=1` with `NCCL_NCHANNELS_PER_NET_PEER=8` (both now on as served; `TF_DS_2D_YX` was off unless
  set):
  the prompt's cross-rank gathers carried the four fp32 quarters of the sublayer partials, and the posting
  kernels
  take only their pair-summed bf16 y from them. With the switch on, the pair partners swap their fp32 partials
  of
  each half of the rows, sum their own half in rank order, and the four nodes all-gather the bf16 halves: a node
  receives 5 n·w bytes instead of 12 n·w, and the quarters copies become one widen. The gathered values are the
  same bits (an unused gather slot holds -0.0, which the kernels read as before), so the GPU kernels are
  unchanged.
  The pair swap runs on eight NCCL channels per peer (the default is one or two); without the channels the
  gathers
  alone bought nothing (+0.1 % on the 93,335-token prompt, `pnc`), and with them that prompt read 2,921 tok/s
  (+6.1 %, `pnc`) before the gate's final rows above.
- The 300K needle run above is a measurement, not an engine change.

Off unless set (unset as served): `TF_DS_SOLO_OVERHEAD=1` (pull request #33), `TF_DS_NO_COPIES=1` (pull request
  #34), `TENSORFOLD_REPLY_IDS=1` and `TF_DS_PASS_ROWS` / `TF_DS_PASS_FROM` (the prompt-pass rule at its default:
  2,048-row passes below position 8,192 and 4,096 above), as in v0.3.2.

Thank you to [@Blackwellboy](https://github.com/Blackwellboy) for pull requests #33, #34 and #38 and to Niklas
  Lenz
  ([niklaslenz.com](https://niklaslenz.com)) for the bring-up report behind v0.3.2's two recipe fixes, as in
  v0.3.2. v0.3.3 adds no pull request; the engine branch's `ATTRIBUTION.md` and this recipe's `ATTRIBUTION.md`
  are
  unchanged.
## v0.3.2: what changed

| | v0.3.1 | v0.3.2 | Measured in |
|---|---|---|---|
| Engine | `a16a167`, on TensorFold 1.0.12 | `51b0c63`, on TensorFold 1.0.12 | |
| Kernel kit | 362 files | the same 362 files | |
| The streams' shared window, the KV cache pool (`TF_DS_WINDOW`; the followers' `--pool`) | 25,165,824 positions
  | 20,971,520 positions (4,194,304 fewer, 16.7 %) | |
| A short chat behind a cold 93,335-token prefill, time to first token (median of 3) | 35.60 s, the
  build served before it in the same session | 0.76 s (`TF_FILL_CHUNKS=1 TF_FILL_MANY=1`; the gate is
  1.9 s) | `zsalt` |
| The long prompt's time to first token, the same session | 36.54 s | 33.85 s | `zsalt` |
| Prefill, a 93,335-token prompt (median of 3) | 2,552 tok/s | 2,757 tok/s (+7.9
  %) | `zsalt` |
| Decode, one stream, set b: code / prose / structured | 136.7 / 78.6 / 185.3 tok/s | 136.2 / 78.4 / 184.1 tok/s (+0.4 / -0.1 / -1.3
  %) | `zsalt` |
| Two / four streams together | 100.8 / 140.9 tok/s | 100.0 / 140.1 tok/s (-0.6 / -0.5 %) | `zsalt` |
| Eight / sixteen streams sustained for 90 s | 220.8 / 261.7 tok/s | 217.6 / 257.0 tok/s (-0.7 / -1.4 %) | `zsalt` |
| Prefill, 8,192 / 32,768 / 131,072-token prompts | 2,770 / 2,833 / 2,779 tok/s | 2,789 / 2,938 / 2,944 tok/s (+0.9 / +3.1 / +5.2 %) |
  `zsalt` |
| Lowest free memory a node during the speed rows | 11 / 12 / 13 / 13 GiB | 12 / 13 / 14 / 14 GiB | `zsalt` |

Measured in: each row's numbers come from the gate slot its "Measured in" column names, on our four nodes; their
  raw
outputs are in `results/served-v032/`. `zsalt` is the gate run of the build v0.3.2 serves (engine
  `1935b48a`,
the served settings; run 4 of the slot). In its comparison rows the v0.3.1 column holds v0.3.1's own numbers,
  from
the gate run of the build v0.3.1 serves (`z3334`, engine `37df889`); the two time-to-first-token rows and the
93,335-token prefill row name no v0.3.1 number (v0.3.1 carried no short-chat row), so their v0.3.1 column is the
build served before it in the same session (`zsalt`'s phase a). Against that build in the same session, v0.3.2's
build moved its 16 speed rows by -1.4 % (16 streams) to +5.2 % (prefill 128K).

Memory: the smaller window frees memory on every node. At 20,971,520 positions the lane leaves about
  12 GiB free
on each Spark: our nodes' lowest free memory was 12 / 13 / 14 / 14 GiB in `zsalt`'s speed rows. Nodes that need
  more
room can use 16,777,216 positions (`TF_DS_WINDOW=16777216` and the followers' `--pool 16777216`;
  `config.env.example`
says where): in v0.3.1's sessions they left 18 / 20 / 20 / 20 GiB free at the lowest, with every reply equal (README,
  v0.3.1).

Engine changes since v0.3.1 (the fork's v0.3.2 merge commit lists them):

- `TF_FILL_CHUNKS=1` with `TF_FILL_MANY=1` (both now on as served; `TF_FILL_MANY` was off unless set): a request
  is
  admitted while a prompt pass fills, and a pass steps its chunks only while a queued request, another pass, a
  decoding stream or a cancel waits — a pass alone runs as a whole one. A short chat behind a cold
  93,335-token
  prefill answers in 0.76 s (35.60 s for the build served before it, same session); the long prompt's
  first
  token came at 33.85 s (36.54 s).
- cache_salt (engine-side, no flag): a request may carry a salt, and its prompt states are kept and resumed only
  for
  that salt, vLLM semantics. A fresh salt caches 0 of a 6,014-token prompt, the same salt 4,096, no salt after a
  salted one 0 again and no salt after that 4,096 (10 of 10 checks); requests without a salt are as before.
- The prompt passes run by position: 2,048-row passes below position 8,192 and 4,096 above (`TF_DS_PASS_ROWS`
  and
  `TF_DS_PASS_FROM` set the rule; unset as served). The Engram read-ahead is sized by the next pass, so a
  shorter
  pass takes rows read ahead that start at it and cover it. The same bits at any rule: a pass ends on the
  2,048-row
  resume grid either way.
- The pool is 25,165,824 → 20,971,520 positions, from free memory (the floor 9 GiB + margin 2 rule on the
  gate's MiB
  watch): 12 / 13 / 14 / 14 GiB lowest during the speed rows.
- TF #38 by @Blackwellboy, cherry-picked (the commit keeps its author): a refused Engram row id no longer leaks
  the
  AIO batch's buffers.

Two durable recipe fixes from an outside tester's bring-up (Niklas Lenz, his report's 5.2 and 5.3):
`scripts/serve.sh` now runs a node that is this host locally — by name, by the name its ssh config gives it, or
  by
its address — so no ssh key to itself is needed, and it gained `preflight`, which checks on rank 0's node every
  file
rank 0 opens at start (the model folder's small files, each file the index maps a vision tower tensor to — the
vision tower's shard `model-00041-of-00041.safetensors` —, the node's weight file, the token map, the kit and
  the
engine), names any missing one, and starts and stops nothing; `start` runs it first. `RUNNING.md` section 2
  lists
the files rank 0 opens.

Off unless set (unset as served): `TF_DS_SOLO_OVERHEAD=1` (pull request #33), `TF_DS_NO_COPIES=1` (pull request
  #34)
and `TENSORFOLD_REPLY_IDS=1`, as in v0.3.1.

Thank you to [@Blackwellboy](https://github.com/Blackwellboy) for pull requests #33, #34 and #38, and to Niklas
  Lenz
([niklaslenz.com](https://niklaslenz.com)) for the bring-up report behind the two recipe fixes. The engine
  branch's
`ATTRIBUTION.md` credits Blackwellboy, and this recipe's `ATTRIBUTION.md` credits both.
## v0.3.1: what changed

| | v0.3.0 | v0.3.1 | Measured in |
|---|---|---|---|
| Engine | `5172621`, on TensorFold 1.0.12 | `a16a167`, on TensorFold 1.0.12 | |
| Kernel kit | 362 files | the same 362 files | |
| The streams' shared window, the KV cache pool (`TF_DS_WINDOW`; the followers' `--pool`) | 4,194,304 positions | 25,165,824 positions, six times as many | |
| The larger window against the smaller one, one launch after the other on v0.3.0's build: launch and settings, the API gates, 16 replies, vision, exact chat, the shield, sampling, structured output, reply ids, DSML calls, draft statistics, kept states, the keepalive and the fill | | 47 of 47 checks pass in each session; every reply equal | `cmpp16`, `cmpp25` |
| Prefill, a 93,335-token prompt (median of 3) | 2,548 tok/s | 2,552 tok/s (+0.1 %) | `cmpp16`, `cmpp25` |
| Decode, one stream, set b: code / prose / structured | 135.8 / 78.7 / 185.5 tok/s | 136.7 / 78.6 / 185.3 tok/s (+0.7 / -0.2 / -0.1 %) | `z3334` |
| Two / four streams together | 100.7 / 140.9 tok/s | 100.8 / 140.9 tok/s (+0.1 / 0.0 %) | `z3334` |
| Eight / sixteen streams sustained for 90 s | 221.1 / 262.0 tok/s | 220.8 / 261.7 tok/s (-0.2 / -0.1 %) | `z3334` |
| Prefill, 8,192 / 32,768 / 131,072-token prompts | 2,786 / 2,835 / 2,797 tok/s | 2,770 / 2,833 / 2,779 tok/s (-0.6 / -0.1 / -0.6 %) | `z3334` |
| Lowest free memory a node during the speed rows | 29 / 30 / 31 / 31 GiB | 11 / 12 / 13 / 13 GiB | `z3334` |

Measured in: each row's numbers come from the sessions it names, on our four nodes; their raw outputs are in
`results/served-v031/<session>/`. `z3334` is the gate run of the build v0.3.1 serves (engine `37df889`, the served
settings); in its rows the v0.3.0 column holds v0.3.0's own numbers (`zreqfix`, at 4,194,304 positions). In the same
session, against the build served before it (v0.3.0's engine `4964e38`, at 25,165,824 positions), v0.3.1's build moved
its 16 speed rows by -1.0 % (prefill 128K) to +1.2 % (set b code). `cmpp16` and `cmpp25` ran v0.3.0's build twice, one
launch after the other, with the window raised in between: 4,194,304 then 16,777,216 positions (`cmpp16`), and
16,777,216 then 25,165,824 (`cmpp25`). Their rows take the v0.3.0 column from `cmpp16`'s first launch and the v0.3.1
column from `cmpp25`'s second; at 16,777,216 positions the same prompt read 2,542 and 2,548 tok/s.

Memory: the larger window takes memory on every node. At 25,165,824 positions the lane leaves about 10 GiB free on each
Spark: our nodes' lowest free memory was 11 / 12 / 13 / 13 GiB, in `cmpp25`'s checks and in `z3334`'s speed rows alike.
Nodes that need more room can use 16,777,216 positions (`TF_DS_WINDOW=16777216` and the followers' `--pool 16777216`;
`config.env.example` says where): in `cmpp16` they left 18 / 20 / 20 / 20 GiB free at the lowest, and every reply
equaled those at 4,194,304.

Engine changes since v0.3.0: two settings for a drafted stream, from pull requests by @Blackwellboy, both off unless set
(unset as served; the fork's v0.3.1 merge commit lists them). Each was measured in `z3334` against the same build with
both unset. With either or both set, every reply the gates compare (the 16 replies, the image prompts, structured
output, reply ids, the DSML rows) equaled v0.3.0's build's.

- `TF_DS_SOLO_OVERHEAD=1` (pull request #33): a lone stream prices its head windows at the fixed draft allowance. Every
  speed row moved -0.8 % to +1.0 %.
- `TF_DS_NO_COPIES=1` (pull request #34): a drafted stream drops its suffix-lookup copy windows, so the draft head
  proposes alone. Replies that copy JSON from the prompt decoded +4.9 %, copied code +2.2 % and set a's structured
  replies +3.2 %, while quoted text decoded -2.5 %; the other rows moved -1.3 % to +1.1 %. With both set: +5.2, +2.0,
  +3.6 and -3.0 %, the other rows -0.9 % to +0.9 %.

Thank you to [@Blackwellboy](https://github.com/Blackwellboy) for pull requests #33 and #34. The engine branch's
`ATTRIBUTION.md` credits them.

## v0.3.0: what changed

| | v0.2.0 | v0.3.0 | Measured in |
|---|---|---|---|
| Engine | `b65c8b9`, on TensorFold 1.0.12 | `5172621`, on TensorFold 1.0.12 | |
| Kernel kit | 358 files | 362 files: `grammar/` adds libtfgrammar, the grammar library for structured output and forced tool calls, with its licenses | |
| The streams' shared window (`TF_DS_WINDOW`) | 2,097,152 positions | 4,194,304 positions | |
| Kept prompt states (`--prompt-cache-gib`) | 0.5 GiB | 2 GiB | |
| `response_format`, and `tool_choice` "required" or a named function | refused (HTTP 400) | answered within the grammar: structured requests 11 of 11 as asked, sampled forced calls 36 of 36 valid | `zreqfix` |
| Eight conversations of 399,951 tokens sent together, two turns each | 4 at a time; second turns resumed 0 of 8; turns 1,206 + 1,203 s | 8 at a time; second turns resumed 8 of 8; turns 1,206 + 6 s | `zbigsolo` |
| Prefill, 131,072 / 262,144 / 524,288-token prompts | 2,667 / 2,492 / 2,211 tok/s | 2,797 / 2,749 / 2,623 tok/s (+4.9 / +10.3 / +18.6 %) | `z1017x` |
| A conversation's next turn after an image | prefilled again (0 tokens cached) | resumes past the image: 4,096 of 6,288 tokens cached | `zkeptk`, `zreqfix` |
| A streamed reply that waits for a 97,510-token prompt (two at once) | silent for 34.9 and 68.8 s | a `: keepalive` comment every 15 s: silent for 15.0 and 15.0 s | `zbigsolo`, `zreqfix` |
| Decode, one stream, set b: code / prose / structured | 135.9 / 79.1 / 186.6 tok/s | 135.8 / 78.7 / 185.5 tok/s (-0.1 / -0.5 / -0.6 %) | `zreqfix` |
| Two / four streams together | 101.6 / 141.7 tok/s | 100.7 / 140.9 tok/s (-0.9 / -0.5 %) | `zreqfix` |
| Prefill, 8,192-token prompt | 2,793 tok/s | 2,786 tok/s (-0.2 %) | `zreqfix` |
| Prefill, 131,072-token prompt | 2,679 tok/s | 2,797 tok/s (+4.4 %) | `zreqfix` |
| Lowest free memory a node during the eight conversations | 25 / 32 / 33 / 33 GiB | 23 / 30 / 31 / 31 GiB | `zbigsolo` |

Measured in: each row's numbers come from the sessions it names, on our four nodes; their raw outputs are in
`results/served-v030/<session>/`. `zreqfix` is the gate run of the build v0.3.0 serves (engine `4964e38`, its
served settings); in the rows that name only it, the v0.2.0 column holds v0.2.0's own numbers. The rows of `zbigsolo`
and `z1017x` compare two launches of one session: `zbigsolo` ran engine `00da56a` (the v0.3.0 line before its
kept-state fixes) with v0.2.0's window, prompt cache and no keepalive, then with v0.3.0's; `z1017x` ran v0.2.0's served
engine `5c97635` with `TF_DS_IDX_SPLIT` unset, then set. A row that names two sessions takes its v0.2.0 column from the
first: for the image turns, `zkeptk`'s engine `e3c6a7d` (before pull request #28); for the keepalive, `zbigsolo`'s
first launch. Decode and the streams are level with v0.2.0 (within 1 %); prefill gains with the prompt's length; large
conversations now run together and resume.

Engine changes since v0.2.0, all on in the served settings (the fork's v0.3.0 merge commit lists them):

- Structured output: a request with `response_format` (`json_schema` or `json_object`), `guided_choice` or
  `guided_regex` draws within its grammar, drafted rows checked against it too. The grammar comes from libtfgrammar,
  built from xgrammar and loaded from the kit (`TENSORFOLD_GRAMMAR_LIB`).
- Forced tool calls: `tool_choice` "required" or a named function answers with one DSML calls block, each call an
  offered tool (the named one) with its parameters within their schema. The DSML marker is always its own token, so
  sampled calls stay valid (36 of 36; 29 of 36 before that fix).
- The prompt shield (`TENSORFOLD_PLAIN_SPECIALS=1`): text in a user turn, a tool result or a tool call's arguments that
  spells one of the tokenizer's special tokens is encoded as ordinary text; only the chat template's own markers and
  the operator's text are control tokens. A prompt in which no sender wrote a marker is tokenized as before.
- Kept prompt states: prompts with images keep and resume their states, matched by the images' digests, and a resumed
  prompt skips the vision tower for the images it holds (pull request #28 by Manrique, ported); a prefix two
  conversations share no longer stales the first one's states (pull request #26 by Manrique); a prompt whose newest
  matching state ends too near its end resumes from an earlier one instead of from the start.
- The keyed indexer layers' long rows split by columns over the four nodes (`TF_DS_IDX_SPLIT=1`): the same columns
  out, prefill faster the longer the prompt.
- The 4,194,304-position window and 2 GiB of kept prompt states (`--prompt-cache-gib 2`): eight 400K-token
  conversations run at once, and their next turns resume.
- The stream keepalive (`TENSORFOLD_SSE_KEEPALIVE=15`): a streamed reply that has sent nothing for 15 s sends
  `: keepalive`, an SSE comment clients skip, so a long prefill does not trip a client's idle timer.
- Draft statistics: `/metrics` counts the draft head's accepted chains by depth, and `TENSORFOLD_STATS_EVERY=60` writes
  a stats line to rank 0's log at most once a minute.
- Sampling: `top_k` 0 or -1 (off) and `top_k` above 56 draw as `top_p` allows instead of failing the request.

Off unless set (unset as served):

- `TF_FILL_CHUNKS=1` (pull request #29 by Manrique, ported): a prompt pass runs in pieces between the other streams'
  rounds. While a 65,010-token prompt filled, another stream's longest stall fell from 23.4 s to
  0.8 s, and the long prompt's first token came +19.2 % later (23.5 s to 28.0 s;
  `zkeptk`, engine `1cd544e`).
- `TENSORFOLD_REPLY_IDS=1`: a reply's end of reasoning and its DSML calls are read at their token ids, not in the
  decoded text, so a reply that quotes those markers as text keeps them as words. Every other reply was equal and the
  speed rows moved -1.2 % to +0.4 % (`zdirdk`, engine `e3c6a7d`).
- `TF_DS_NEIGHBOR_RDMA=1` (pull request #30 by Scott Woolsey Biggart): the neighbor transport, adjacent-edge RDMA
  exchanges for four nodes cabled as a physical ring, in place of the RDMA rings. Not measured on our switched network;
  the engine's `docs/recipes/deepseek-v41-neighbor.md` describes it, with its author's receipts.

Pull request #31 by mrsergiopascal adds a CPU test and a reproducer of the replay cut's position alignment, with its
write-up; the branch keeps its own replay cut, which the test checks.

Thank you to Manrique ([@hashd1ve](https://github.com/hashd1ve)) for pull requests #26, #28 and #29, Scott Woolsey
Biggart ([@swbiggart](https://github.com/swbiggart)) for #30, [@mrsergiopascal](https://github.com/mrsergiopascal) for
#31, [@den13501](https://github.com/den13501), whose report led to the keepalive, and
[@Capicua25x](https://github.com/Capicua25x), whose report led to the loop guard's novelty signal. The engine branch's
`ATTRIBUTION.md` credits each of them, and the ideas taken from DeepSeek's DeepInfer and from xgrammar.

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

All numbers are ours, measured with our own clients (`kit_bench.py` and the gate tools of the engine branch) on the

four nodes, on the served launch: engine `39155041`, the 1017 weights, this kit and the served settings below,
unless a row names another session. `ENGINE_COMMIT` (`b116c47`) is that engine's source (`39155041`,
which builds the same binaries but for nvcc's build ids) on the branch's public line, which adds pull requests #30
(off unless `TF_DS_NEIGHBOR_RDMA` is set) and #31 (a test and its write-up; see `RUNNING.md`). Greedy decoding,
thinking off, unless a row says otherwise. `results/served-v033/`, `results/served-v032/` and `results/served-v031/`
hold the raw outputs, a folder for each session (`results/README.md`).

### Speed

The gate run of the served build (`z3334`):

The gate run of the served build (`zyx`, phase c-v033):

| Row | Result |
|---|---|
| Decode, one stream, set b (384 tokens, median of 3): code / prose / structured | 135.6 / 78.6 / 183.7 tok/s |
| Decode, one stream, set a (512 tokens, median of 3): code / prose / structured | 107.6 / 70.9 / 138.5 tok/s |
| Decode, one stream, set c, replies that copy from the prompt (512 tokens, median of 6): code / JSON / quoted text | 151.8 / 152.4 / 163.2 tok/s |
| Two streams together (256 tokens each, mean of 6) | 100.4 tok/s aggregate |
| Four streams together (256 tokens each, mean of 6) | 139.9 tok/s aggregate |
| Eight streams sustained for 90 s (256 tokens each) | 217.6 tok/s aggregate |
| Sixteen streams sustained for 90 s (256 tokens each) | 255.6 tok/s aggregate |
| Prefill, 8,192-token prompt (median of 3) | 2,974 tok/s |
| Prefill, 32,768-token prompt (median of 3) | 3,107 tok/s |
| Prefill, 131,072-token prompt (median of 3) | 3,140 tok/s |
| Lowest free memory a node during these rows | 11 / 13 / 13 / 14 GiB |

### Long context and kept prompts

These rows are v0.3.0's, at its 4,194,304-position window. On v0.3.1's build the kept-state, resume, keepalive
and fill checks gave replies equal to v0.3.0's build's (`z3334`). On v0.3.2's build they ran in its gate's
phase b and equaled the build served before it's (`zsalt`). On v0.3.3's build they ran in its gate's phase b
(the candidate with both settings unset): the kept-state replies 8 of 8, the resume probe's 4 of 4 and the
fill replies 2 of 2 equal to the build served before it's, and the keepalive re-ran on the served port after
the promotion, replies 2 of 2 equal (`zyx`). The 300K needle ran on the served build after the promotion
(`row34`): 3 of 3 needles found in each of three runs, median 98.65 s. The 999,843-token needle
check still has not run on the served settings, as in v0.3.1 and v0.3.2 (the README keeps v0.2.0's needle
record and says what it ran on).

| Check | Result |
|---|---|
| Eight conversations of 399,951 tokens, two turns; each turn's eight requests sent together, each with `max_tokens` 32,768 (`zbigsolo`) | All eight run at once; the second turns resume from their kept prompts 8 of 8; turns 1,206 + 6 s; the 16 replies equal those of v0.2.0's settings |
| The same with four conversations (`zbigsolo`) | Four at once with both settings; second turns resumed 4 of 4; turns 605.2 + 3.5 s (v0.2.0's settings: 604.0 + 3.6 s) |
| Lowest free memory a node during the eight | 23 / 30 / 31 / 31 GiB (v0.2.0's settings: 25 / 32 / 33 / 33 GiB) |
| Image turns on a 2,822-token system prompt (`zreqfix`) | The first image turn resumes the system prompt's kept state (2,048 of 6,267 tokens cached), the next turn resumes past the image (4,096 of 6,288), and that turn with another image in its place resumes up to the image (2,048 of 6,288). Engine `e3c6a7d`, before pull request #28, ran all three from the start (`zkeptk`) |
| A prompt whose newest matching kept state ends within its last 128 tokens (`zreqfix`) | It resumes from an earlier state: 4,096 of 6,204 tokens cached, and the next turn 6,144 of 8,492. Engine `00da56a`, before the fix, cached 0 and 4,096 (`zbigsolo`) |
| Nine prompt lengths from 65,535 to 524,287 tokens with the split set (`z1017x`) | All 9 answer, each reply equal to the one with the split unset |

The 999,843-token needle check and the four 160K prompts of v0.2.0 ran with the split unset and the 2,097,152-position
window; their results are in the v0.2.0 README.

### Gates

On the gate launch of the served build (`z3334`), and where a line says so, again on the served port after the

promotion and in the slot's earlier phases. Every reply these gates compare equals that of the build served before
it (v0.3.2's engine `1935b48a`) in the same session:

- Exact chat: 71 of 71 chats (system and developer prompts, tools and tool calls, tool results, reasoning and its
  effort, multi-turn, response formats) give the same token ids through the server's `/tokenize` as DeepSeek's
  reference encoding (`encoding.py`, MIT) gives for them; 71 of 71 on the served port after the promotion.
- The API gates 6 of 6: the model list, a chat probe, a streamed reasoning, a tool call, drafted equals serial, and
  a burst of four; 6 of 6 on the served port.
- 16 replies sent as a burst and staggered: equal to their solo replies 16/16 and 16/16, and the solo replies equal
  the build served before it's.
- Vision: the image gate 6/6; 9 image prompts' token ids, then the same 9 four at a time, twice: 18/18 equal to the
  solo run.
- The prompt shield: 14 of 14 cases in the slot's phase b (in 5 no sender wrote a marker, and they tokenize as
  before; in 9 one did, and the marker stays text); 14 of 14 on the served port.
- Structured output: 11 of 11 requests as asked (a JSON schema, a JSON object, a choice, a regex; greedy, sampled
  and with thinking), burst and staggered equal to solo; 11 of 11 on the served port.
- Forced tool calls: 16 rows (8 requests, whole and streamed) of "required", named and "auto" calls, the forced ones
  each an offered tool within its schema and the "auto" ones equal to those of the build served before it
  (v0.3.2's engine `1935b48a`); and sampled forced requests: 36 of 36 valid, 36 of 36 on the served port.
- cache_salt: 10 of 10 checks. A fresh salt caches 0 of the 6,012-token prompt, the same salt 4,096, no salt after a
  salted one 0 again and no salt after that 4,096 (the completions and the chat forms both).
- A short chat behind the cold 93,335-token prompt: 0.68 s to the first token (gate 0.89 s),
  against 0.76 s for the build served before it in the same session; the long prompt itself
  32.00 s (34.04 s), and the six replies equal.

### Run-time inputs from the public weights

| Check | Result |
|---|---|
| The download of [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw) (revision `a2f91e27`) against the weights the served lane loads | All 41 shards, the index and the configs byte for byte our copy of the converter's output; 192,492 of its 192,496 tensors equal (sha256 per tensor) to the served lane's, and the other four are the Engram q/k that the served lane holds widened to F32 |
| `tools/serving_1017.py` on that download | Its six tensors equal the served lane's (name, type, shape, sha256); `serving-1017.safetensors` sha256 `d88fe41f...` and `chat_template.jinja` `b581d19d...` (`results/served-1017/inputs.txt`) |
| Each node's weight file from that download (`tf-dsv41-rank-cache --parity --index-only`, four ranks) | The four indexes hash to the names of the served lane's four weight files; without `serving-1017.safetensors` the writer stops on a missing tensor |
| The kit from a fresh clone of this repository (`kit/verify_kit.sh`) | 362 of 362 files match `kit/MANIFEST` |

The weights, their readying and each node's weight file are v0.2.0's: the served lane loads the same four weight files
(`results/served-1017/inputs.txt`), so the first three rows are v0.2.0's checks.

### Earlier releases

v0.3.2 served engine `1935b48a` with a 20,971,520-position window; its tables are in the v0.3.2 README (tag
`v0.3.2`)
and its raw outputs in `results/served-v032/`. v0.3.1 served engine `37df889` with a 25,165,824-position
window; its tables are in the v0.3.1 README (tag `v0.3.1`)
and its raw outputs in `results/served-v031/`. v0.3.0 served engine `4964e38` with a 4,194,304-position window; its
tables are in the v0.3.0 README (tag `v0.3.0`)
and its raw outputs in `results/served-v030/`. v0.2.0 served engine `5c97635` on these weights; its tables are in the
v0.2.0 README (tag `v0.2.0`) and its raw outputs in `results/served-1017/`. v0.1.0 and v0.1.1 served the 2.9 bpw
weights on engine `ad70f4f` and `5940208`. Their tables are in the v0.1.1 README (tag `v0.1.1`), and their raw outputs
stay in `results/` (`served-ad70f4f/`, `inputs-5940208/`, `throughput-1ca1c10/`, `probes-1ca1c10/`).

## The served settings

Rank 0 runs `tensorfold-native serve` with `TF_TP_WORLD=4`; ranks 1-3 run `tf-dsv41-lanes`, which follows it. The

settings as served: `--context 1048576 --parallel 16` and `--prompt-cache-gib 2` on rank 0, `TF_DS_WINDOW=20971520`
(the streams' shared positions; the followers' `--pool 20971520`), `TF_DS_ROUND_ROWS=48`, `TF_DS_STREAMS=16`,
`TF_DS_2D_GU=parity` (the balanced experts split), `TF_DS_SERVED_K=solo`, the measured round costs in
`TF_DS_ROUND_MS`, `TF_DS_IDX_SPLIT=1`, `TF_FILL_CHUNKS=1`, `TF_FILL_MANY=1`, `TF_DS_2D_YX=1`,
`NCCL_NCHANNELS_PER_NET_PEER=8`, `TENSORFOLD_PLAIN_SPECIALS=1`, `TENSORFOLD_GRAMMAR_LIB` (the kit's
`grammar/libtfgrammar.so`), `TENSORFOLD_SSE_KEEPALIVE=15`, `TENSORFOLD_STATS_EVERY=60`, CUDA graphs, the side
stream, the paced L2 prefetch, Engram reads by AIO, and the decode-size exchanges over the RDMA rings. New since
v0.3.1: `TF_FILL_MANY` with `TF_FILL_CHUNKS` (a request admitted while a prompt pass fills; a pass alone runs as a
whole one), cache_salt (engine-side, no flag), the prompt passes by position, and the 20,971,520-position window
(25,165,824 before; nodes that need more room can use 16,777,216, see v0.3.2's section). New since v0.3.2:
`TF_DS_2D_YX=1` with `NCCL_NCHANNELS_PER_NET_PEER=8` (the prompt's cross-rank gathers carry the pair-summed bf16 y
instead of the four fp32 quarters, and the ring's send/recv pairs run on eight channels a peer).
`TENSORFOLD_REPLY_IDS`, `TF_DS_NEIGHBOR_RDMA`, `TF_DS_SOLO_OVERHEAD` and `TF_DS_NO_COPIES` are unset.
`TF_DS_PASS_ROWS` and `TF_DS_PASS_FROM` are unset (the pass rule at its default). The round costs are the table
measured on the 2.9 bpw weights: they guide how many tokens a round drafts, and drafted output equals serial
output, so no reply depends on them. `scripts/serve.sh` spells the launch out; `config.env.example` holds the site
settings.

## Requirements

- Four NVIDIA GB10 nodes on one RoCE network, each with the NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`
  (Pillow 12.3.0 inside it decodes the image formats other than PNG).
- Memory: the served window leaves about 12 GiB free on each node; nodes that need more room can use the
  16,777,216-position window (see v0.3.2's section).
- The weights: [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw)
  (about 275 GB), readied by `tools/serving_1017.py`, and the original checkpoint's shards 47 and 48 for the Engram
  tables, on every node.
- The engine built from `ENGINE_COMMIT` (`zig build` with the CUDA toolchain; see `RUNNING.md`).
- The run-time inputs, made from the public weights (see `RUNNING.md`): each node's weight file and the compressed
  token map, one command a node (`node_inputs.sh` of the engine branch; `scripts/serve.sh inputs` runs it on all four),
  and the kernel kit in `kit/`, whose RoPE tables and image bias `kit/verify_kit.sh` makes before it checks every file
  against `kit/MANIFEST`.

## License and attribution

This recipe is licensed under Apache-2.0 (`LICENSE`). The engine branch keeps TensorFold's licenses (Apache-2.0 from
TensorFold 0.6.0, and the MIT notice of the code written before it), lists the upstream files it changes in its
`NOTICE` and every outside source in its `ATTRIBUTION.md`. The weights are MIT (DeepSeek; our EXL3 quant inherits it).
The chat template is ours, written to render what DeepSeek's reference encoding (MIT) renders. `kit/` holds compiled
code: our own Triton kernels (Apache-2.0), the EXL3 kernels of TensorFold's EXL3 module with our additions
(Apache-2.0, and TensorFold's MIT notice for its code written before 0.6.0), PyTorch's attention kernel for the vision
tower, unmodified (BSD-3-Clause), and libtfgrammar, xgrammar (Apache-2.0) built with TensorFold's shim and the headers
it bundles (dlpack, Apache-2.0; picojson, BSD-2-Clause); `kit/KIT-LICENSES.md` gives each file's source and license.
`ATTRIBUTION.md` here lists what this recipe uses. "DeepSeek" belongs to DeepSeek and "DGX Spark" and "GB10"
to NVIDIA; this recipe is not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors or Mia-AiLab.
