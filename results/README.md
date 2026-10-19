# Results

Our own measurements of our own engine on four GB10 nodes, as the README's tables report them. Private host names,
addresses and paths are not in these files; nothing else was changed.

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
