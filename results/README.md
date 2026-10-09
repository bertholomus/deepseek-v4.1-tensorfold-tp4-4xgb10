# Results

Our own measurements of our own engine on four GB10 nodes, as the README's tables report them. Private host names,
addresses and paths are not in these files; nothing else was changed.

## `served-ad70f4f/`: the served build, the served settings (2026-10-09)

| File | What |
|---|---|
| `rows-2.json`, `rows-3.json`, `rows-6.json` | `kit_bench.py` decode set b (median of 3), 2 and 4 streams (six reps), 8K prefill (six reps) |
| `long-500k.json` | `long_check.py`: four 499,931-token prompts and a repeat of the first, together then alone (the `summary` holds burst == solo, needles, drafted == serial, the cached tokens of each solo rerun) |
| `long-160k-after-500k.json` | the same with 159,925-token prompts, right after on the same server |
| `lensweep.json` | `lensweep.py --lengths wide`: 299 prompt lengths, each reply against the reference lane's (`equal`) |
| `api-gates.json`, `parallel-summary.json` | `gates.py` (6 API gates); 16 replies against the two-node reference, burst and staggered against solo |
| `vision-gate.json`, `visionref.json`, `vision-four-at-a-time.json` | the image gate (6 checks); 9 image prompts' token ids; the same 9 four at a time, twice |

## `throughput-1ca1c10/`: the same model code, a test server at 1,048,576 tokens a request (2026-10-09)

`rows-1.json` decode set a, `rows-2.json` set b, `rows-3.json` 2 and 4 streams, `rows-4.json` / `rows-5.json` 8 and 16
streams sustained for 90 s, `rows-6.json` prefill 8K / 32K / 128K.

## `probes-1ca1c10/`

`vprobe.json`: 48 image cases (formats, message layouts, refusals); `vprobe-cmp.txt`: all 48 equal to the reference
lane's (`tools/vision/lane-probe.json`).
