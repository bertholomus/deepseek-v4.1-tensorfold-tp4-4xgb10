# Results

Our own measurements of our own engine on four GB10 nodes, as the README's tables report them. Private host names,
addresses and paths are not in these files; nothing else was changed, except where a section says a file holds only
a part of a check's output.

## `served-v030/`: v0.3.0, a folder for each session on the four nodes (2026-10-20 and 2026-10-21)

Each of the README's v0.3.0 rows names the session it comes from. The gate runs used the served launch on a second
port (18095); `*-served.json` were taken on the served port after the build was promoted. A `parallel-summary.json` is
the 16-reply check's summary; the two sweep files leave out the prompt's token ids (one text for both).

### `zreqfix/`: the gate run of the build v0.3.0 serves, engine `4964e38`, the served settings (2026-10-21)

| File | What |
|---|---|
| `rows-1.json` to `rows-6.json` | `kit_bench.py` decode set a and set b (median of 3), 2 and 4 streams (six reps), 8 and 16 streams sustained for 90 s, prefill 8K / 32K / 128K (three reps each) |
| `mem.txt` | each node's free memory (MemAvailable, GiB) about every 15 s during the rows, nodes named by rank |
| `exact-chat.json`, `exact-chat-served.json` | the exact-chat check: 71 chats, each case's `/tokenize` ids against those of DeepSeek's reference encoding (`equal`) |
| `api-gates.json`, `api-gates-served.json`, `parallel-summary.json` | `gates.py` (6 API gates); 16 replies as a burst and staggered, against solo |
| `vision-gate.json`, `visionref.json`, `vision-four-at-a-time.json` | the image gate (6 checks); 9 image prompts' token ids; the same 9 four at a time, twice |
| `shield.json`, `shield-served.json` | the prompt shield: 14 conversations' token ids, with and without spelled special tokens |
| `structured.json`, `structured-served.json` | 12 structured requests solo, as a burst and staggered. The 12th, `tool_choice` "required", answers with a call; the check was written when the served build refused such requests, so its verdict counts that case as failed (the forced-call files judge the calls) |
| `dsml.json`, `dsml-served.json`, `dsml-before.json` | 16 rows of "required", named and "auto" tool calls (eight requests, whole and streamed); `dsml-before.json` on the build served before (engine `1cd544e`), every row equal to the other two |
| `forced-calls.json`, `forced-calls-served.json` | 36 sampled forced requests: each row's calls, finish and counts of the DSML marker's token |
| `forced-calls-before.json` | the same 36 on the build served before the marker fix (engine `1cd544e`, the session's first launch) |
| `accept-stats.json`, `accept-stats-served.json` | the draft head's accepted chains by depth, from `/metrics` |
| `sampler.json` | sampling with `top_k` off and above 56 |
| `reply-ids.json` | 16 replies that quote the reasoning end or the DSML marker, `TENSORFOLD_REPLY_IDS` unset |
| `keepalive.json`, `keepalive-served.json` | two streamed 97,510-token prompts at once: each SSE comment's time and text, the longest silence before the first token |
| `kept-states.json` | eight turns over kept states: image turns, and two conversations on one system prompt |
| `resume.json` | four prompts of one text: the resume when the newest kept state ends near a prompt's end |

### `zbigsolo/`: engine `00da56a`, v0.2.0's window, prompt cache and no keepalive (a), then v0.3.0's (b) (2026-10-21)

| File | What |
|---|---|
| `big4-a.json`, `big4-b.json`, `big8-a.json`, `big8-b.json` and their `.log` | four and eight conversations of 399,951 tokens, two turns each sent together: each reply's ids, prompt and cached tokens and times; each log's last line holds the turns' wall times and the most requests running at once |
| `big8-a-mem.txt`, `big8-b-mem.txt` | each node's free memory during the eight, nodes named by rank |
| `keepalive-a.json`, `keepalive-b.json` | the keepalive check unset, then at 15 s |
| `resume-a.json` | the resume probe before the kept-state fix |

### `z1017x/`: v0.2.0's served engine `5c97635`, `TF_DS_IDX_SPLIT` unset, then set (2026-10-20)

| File | What |
|---|---|
| `unset/rows-1.json` to `rows-4.json`, `set/rows-1.json` to `rows-4.json` | prefill 128K (three reps), 256K (two) and 512K (one, twice) |
| `sweep-unset.json`, `sweep-set.json` | nine prompt lengths from 65,535 to 524,287 tokens, each reply's ids (set: `equal` to unset) |

### `zkeptk/`: engines `e3c6a7d` (a) and `1cd544e` (b) (2026-10-21)

| File | What |
|---|---|
| `kept-states-a.json` | the kept-states check before pull request #28: image turns run from the start |
| `fill-unset.json`, `fill-set.json` | a stream's stalls while a 65,010-token prompt fills, `TF_FILL_CHUNKS` unset, then set |

### `zdirdk/`: engine `e3c6a7d`, `TENSORFOLD_REPLY_IDS` unset (b), then set (c) (2026-10-21)

| File | What |
|---|---|
| `rows-unset/rows-1.json` to `rows-6.json`, `rows-set/rows-1.json` to `rows-6.json` | the speed rows of both launches |

## `served-1017/`: v0.2.0, the served build on the 1017 weights, the served settings (2026-10-19)

Engine `5c97635` (`ENGINE_COMMIT` `b65c8b9` is its source with the fork's documents), bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw,
the kit in `kit/`. The gate run used the served launch on a second port (18095); `*-served.json` were taken on the
served port after it.

| File | What |
|---|---|
| `rows-2.json`, `rows-3.json`, `rows-4.json` | `kit_bench.py` decode set b (median of 3), 2 and 4 streams (six reps), 4 streams sustained for 90 s |
| `rows-6.json`, `rows-7.json`, `rows-8.json` | prefill 8K (three reps) and 128K (two reps); decode after a 128K prompt |
| `exact-chat.json`, `exact-chat-served.json` | the exact-chat check: 71 chats, each case's `/tokenize` ids against those of DeepSeek's reference encoding (`equal`) |
| `api-gates.json`, `api-gates-served.json`, `parallel-summary.json` | `gates.py` (6 API gates); 16 replies as a burst and staggered, against solo |
| `vision-gate.json`, `visionref.json`, `vision-four-at-a-time.json` | the image gate (6 checks); 9 image prompts' token ids; the same 9 four at a time, twice, each against the solo run (`equal`) |
| `needles-1m.json` | `needle.py`: three needles in one shared 999,843-token haystack (the second and third resume from the kept prompt) |
| `long-160k.json` | `long_check.py`: four prompts of about 160,000 tokens and a repeat of the first, together then alone |
| `inputs.txt` | the four weight files and the token map the served lane loads, and the sha256 of the two files `tools/serving_1017.py` writes |

## `served-ad70f4f/`: v0.1.0, the served build on the 2.9 bpw weights, the served settings (2026-10-09)

| File | What |
|---|---|
| `rows-2.json`, `rows-3.json`, `rows-6.json` | `kit_bench.py` decode set b (median of 3), 2 and 4 streams (six reps), 8K prefill (six reps) |
| `long-500k.json` | `long_check.py`: four 499,931-token prompts and a repeat of the first, together then alone (the `summary` holds burst == solo, needles, drafted == serial, the cached tokens of each solo rerun) |
| `long-160k-after-500k.json` | the same with 159,925-token prompts, right after on the same server |
| `lensweep.json` | `lensweep.py --lengths wide`: 299 prompt lengths, each reply against the reference lane's (`equal`) |
| `api-gates.json`, `parallel-summary.json` | `gates.py` (6 API gates); 16 replies against the two-node reference, burst and staggered against solo |
| `vision-gate.json`, `visionref.json`, `vision-four-at-a-time.json` | the image gate (6 checks); 9 image prompts' token ids; the same 9 four at a time, twice |

## `inputs-5940208/`: v0.1.1, the run-time inputs from the public 2.9 bpw weights, and the recipe's launch on them (2026-10-09)

| File | What |
|---|---|
| `inputs.sha256` | each node's weight file and the token map as `node_inputs.sh` made them (sha256, bytes): byte for byte the served lane's |
| `api-gates.json`, `parallel-summary.json` | on `scripts/serve.sh start` from those inputs and the kit: `gates.py` (6 API gates); 16 replies against the two-node reference, burst and staggered against solo |
| `vision-gate.json`, `visionref.json`, `vision-four-at-a-time.json` | the image gate (6 checks); 9 image prompts' token ids; the same 9 four at a time, twice |

## `throughput-1ca1c10/`: the same model code, a test server at 1,048,576 tokens a request (2026-10-09)

`rows-1.json` decode set a, `rows-2.json` set b, `rows-3.json` 2 and 4 streams, `rows-4.json` / `rows-5.json` 8 and 16
streams sustained for 90 s, `rows-6.json` prefill 8K / 32K / 128K.

## `probes-1ca1c10/`

`vprobe.json`: 48 image cases (formats, message layouts, refusals); `vprobe-cmp.txt`: all 48 equal to the reference
lane's (`tools/vision/lane-probe.json`).
