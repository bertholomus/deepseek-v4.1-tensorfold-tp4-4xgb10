# Results

Our own measurements of our own engine on four GB10 nodes, as the README's tables report them. Private host names,
addresses and paths are not in these files; nothing else was changed, except where a section says a file holds only
a part of a check's output.

## `served-v035/`: v0.3.5, the A/B and gate slots' files on the four nodes (2026-10-23)

Two slots, each on a side port with the served lane stopped, then the promotion. A = v0.3.3's served build (engine
`39155041`, `NCCL_NCHANNELS_PER_NET_PEER=8`), B = v0.3.5's (engine `ebf6ae33`, `TF_DS_WAVES=1 NCCL_MAX_NCHANNELS=4`).
Private host names and paths are out; nothing else changed.

| File | What |
|---|---|
| `pnc4/{a1,b1,a2,b2}-{8k,32k,128k}.json` | `cmp1021_bench.py prefill`, cold (fresh prompts, cached 0): 8K and 32K three reps, 128K two; the launches ran in the order a1, b1, a2, b2 |
| `pnc4/{a1,b1,a2,b2}-dec.json` | 512-token greedy decode, three reps a launch, with each reply's sha256 |
| `pnc4/summary.txt` | the A/B run's log |
| `pnc4g/result.txt` | the gate's 43 verdicts, A then B, and the 93,335-token prefill line |
| `pnc4g/prefill-93335-{a,b}.json` | the 93,335-token prefill, three reps each |
| `pnc4g/*.json`, `pnc4g/done-b.txt` | B's gate files: gates, parallel (A's and B's), exact chat, vision, structured, DSML, sampler, reply ids, accept stats, shield, keepalive, kept states, resume, fill; B's request log |
| `promote.log` | the promotion: the build to the four nodes, the served port's gates and the 16 replies |

## `served-v033/`: v0.3.3, the gate slot's files on the four nodes (2026-10-22)

The README's v0.3.3 rows come from slot `zyx` (chain-zyx). Each launch is a phase: a = the build served before
  it
  (engine
`1935b48a`, the settings off), b = the candidate `39155041` with `TF_DS_2D_YX` and
  `NCCL_NCHANNELS_PER_NET_PEER`
unset, c-v033 = the build v0.3.3 serves (engine `39155041`) with `TF_DS_2D_YX=1
  NCCL_NCHANNELS_PER_NET_PEER=8`
at the 20,971,520 pool. The gate files below are c-v033's, with a's and b's where a name says so; the kept-
  state,
  resume,
fill and keepalive checks ran in b (a records the served build's), and the served port was re-checked after the
promotion. The 300K needle and the 93,335-token prefill ran on the served build after the promotion (client-
  only;
  `needle300k.json`, `prefill-93335.json`; v0.3.2's needle run is in the project's PROGRESS.md). Each file is as
  the slot
wrote it, with private host names and paths taken out; a file it names is named without its folder.

| File | What |
|---|---|
| `result.txt` | the slot's verdicts: a line a phase with each check and the 16 rows against a's, then the
  promotion |
| `zyx.out` | the run's log: each phase's line, the promotion and the served port's re-check |
| `rows-a/`, `rows-b/`, `rows-v033/`: `rows-1.json` to `rows-7.json` | `rows-1.json` to `rows-6.json`:
  `kit_bench.py` decode set a (512 tokens) and set b (384 tokens), median of 3; 2 and 4 streams (six reps); 8
  and 16 streams sustained for 90 s; prefill 8K / 32K / 128K (three reps each). `rows-7.json`: set c, replies
  that copy code, JSON or quoted text from the prompt (512 tokens, median of 6), from our lane tool |
| `rows-*/mem.txt` | each node's free memory (MemAvailable, GiB) about every 15 s during the rows, nodes named
  by rank |
| `rowcmp-b.log`, `rowcmp-v033.log` | each phase's 16 rows against a's (one decimal), then the verdict |
| `behind-a.json`, `behind-v033.json` | the time-to-first-token rows: a short chat behind the cold 93,335-token
  prompt and the long prompt itself, three reps (`summary` holds the medians; the 93,335-token prefill derived
  from them is median 93335/ttft_s over the long rows) |
| `salt-a.json`, `salt-v033.json` | the cache_salt check: ten cases, `want` and `cached` per case; a's is the
  served build's record (not gated) |
| `gates-v033.json`, `gates-served.json` | `gates.py` (6 API gates), the candidate's and the served port's after
  the promotion |
| `exact-chat.json` | the exact-chat check on the served port after the promotion: 71 chats against DeepSeek's
  reference encoding |
| `parallel-summary.json` | 16 replies as a burst and staggered, against solo |
| `vision-gate.json`, `visionref.json`, `vision-four-at-a-time.json` | the image gate (6 checks); 9 image
  prompts' token ids; the same 9 four at a time |
| `structured.json` | 12 structured requests solo, as a burst and staggered |
| `dsml.json` | 16 rows of "required", named and "auto" tool calls (eight requests, whole and streamed) |
| `forced-calls.json` | 36 sampled forced requests: each row's calls, finish and counts of the DSML marker's
  token |
| `accept-stats.json` | the draft head's accepted chains by depth, from `/metrics` |
| `sampler.json` | sampling with `top_k` off and above 56 |
| `reply-ids.json` | 16 replies that quote the reasoning end or the DSML marker, `TENSORFOLD_REPLY_IDS` unset |
| `shield.json` | the prompt shield on b: 14 conversations' token ids, with and without spelled special tokens;
  the served port's is in `zyx.out` |
| `keepalive.json` | two streamed 97,510-token prompts at once, on b: each SSE comment's time and text, the
  longest silence before the first token; the served port's is in `zyx.out` |
| `kept-states.json`, `resume.json`, `fill.json` | the kept-state, resume and fill checks, on b |
| `needle300k.json` | the 300K needle on the served build: three runs (seeds 101-103), each 3 of 3 needles
  found, cached 0 |
| `prefill-93335.json` | the 93,335-token prefill on the served build: `cmp1021_bench.py prefill 3`, three reps
  and the median |
| `done-v033.txt`, `start-*.log` | the phase's request log and the three launches' records (the launch lines and
  flags) |

## `served-v032/`: v0.3.2, the gate slot's files on the four nodes (2026-10-22)

The README's v0.3.2 rows come from gate run 4 of slot `zsalt` (chain-zsalt). Each launch is a phase: a = the
  build
served before it (engine `37df889`, the switches off), b = the TF_FILL_MANY and cache_salt build with both
  switches
unset, c-v032 = the build v0.3.2 serves (engine `1935b48a`) with `TF_FILL_CHUNKS=1 TF_FILL_MANY=1` at
  the
20,971,520 pool. The gate files below are c-v032's, with a's and b's where a name says so; the kept-state,
  resume,
fill and keepalive checks ran in b (a records the served build's), and the served port was re-checked after the
promotion. Each file is as the slot wrote it, with private host names and paths taken out; a file it names is
  named
without its folder.

| File | What |
|---|---|
| `result.txt` | the slot's verdicts: a line a phase with each check and the 16 rows against a's, then the
  promotion |
| `zsalt.out` | the run's log: the slot's start, each phase's line, `d alone`, the promotion and the served
  port's re-check |
| `rows-a/`, `rows-b/`, `rows-v032/`: `rows-1.json` to `rows-7.json` | `rows-1.json` to `rows-6.json`:
  `kit_bench.py` decode set a (512 tokens) and set b (384 tokens), median of 3; 2 and 4 streams (six reps); 8
  and 16 streams sustained for 90 s; prefill 8K / 32K / 128K (three reps each). `rows-7.json`: set c, replies
  that copy code, JSON or quoted text from the prompt (512 tokens, median of 6), from our lane tool |
| `rows-*/mem.txt` | each node's free memory (MemAvailable, GiB) about every 15 s during the rows, nodes named
  by rank |
| `rowcmp-b.log`, `rowcmp-v032.log` | each phase's 16 rows against a's (one decimal), then the verdict |
| `behind-a.json`, `behind-v032.json` | the time-to-first-token rows: a short chat behind the cold 93,335-token
  prompt and the long prompt itself, three reps (`summary` holds the medians; the 93,335-token prefill is median
  93335/ttft_s over the long rows) |
| `salt-a.json`, `salt-v032.json` | the cache_salt check: ten cases, `want` and `cached` per case; a's is the
  served build's FAIL record |
| `gates-b.json`, `gates-v032.json` | `gates.py` (6 API gates) |
| `exact-chat.json` | the exact-chat check on b: 71 chats against DeepSeek's reference encoding; the served
  port's re-check is in `zsalt.out` |
| `parallel-summary.json` | 16 replies as a burst and staggered, against solo |
| `vision-gate.json`, `visionref.json`, `vision-four-at-a-time.json` | the image gate (6 checks); 9 image
  prompts' token ids; the same 9 four at a time |
| `structured.json` | 12 structured requests solo, as a burst and staggered |
| `dsml.json` | 16 rows of "required", named and "auto" tool calls (eight requests, whole and streamed) |
| `forced-calls.json` | 36 sampled forced requests: each row's calls, finish and counts of the DSML marker's
  token |
| `accept-stats.json` | the draft head's accepted chains by depth, from `/metrics` |
| `sampler.json` | sampling with `top_k` off and above 56 |
| `reply-ids.json` | 16 replies that quote the reasoning end or the DSML marker, `TENSORFOLD_REPLY_IDS` unset |
| `shield.json` | the prompt shield on b: 14 conversations' token ids, with and without spelled special tokens |
| `keepalive.json` | two streamed 97,510-token prompts at once, on b: each SSE comment's time and text, the
  longest silence before the first token; the served port's is in `zsalt.out` |
| `kept-states.json`, `resume.json`, `fill.json` | the kept-state, resume and fill checks, on b |
| `done-v032.txt`, `start-*.log` | the phase's request log and the three launches' records (the launch lines and
  flags) |

## `served-v031/`: v0.3.1, a folder for each session on the four nodes (2026-10-21)

Each of the README's v0.3.1 rows names the session it comes from. The gate runs used the served launch on a second
port (18095); `*-served.json` and `mem-served-watch.txt` were taken on the served port after the build was promoted.
Each session's `result.txt` holds its verdicts as the session wrote them, with private host names and paths taken
out; a file it names is named without its folder.

### `z3334/`: the build v0.3.1 serves, engine `37df889`, against the build served before, engine `4964e38` (2026-10-21)

The window was 25,165,824 positions in every launch: engine `4964e38` (a), then `37df889` with both new
settings unset (b), with `TF_DS_SOLO_OVERHEAD=1` (s), with `TF_DS_NO_COPIES=1` (n) and with both (sn). The gate files
below are b's, with a's and the served port's where a name says so.

| File | What |
|---|---|
| `result.txt` | the session's verdicts: a line a launch with each check and the 16 rows against a's, then the promotion |
| `rows-a/` to `rows-sn/`: `rows-1.json` to `rows-7.json` | `rows-1.json` to `rows-6.json`: `kit_bench.py` decode set a (512 tokens) and set b (384 tokens), median of 3; 2 and 4 streams (six reps); 8 and 16 streams sustained for 90 s; prefill 8K / 32K / 128K (three reps each). `rows-7.json`: set c, replies that copy code, JSON or quoted text from the prompt (512 tokens, median of 6), from our lane tool |
| `rows-a/mem.txt` to `rows-sn/mem.txt` | each node's free memory (MemAvailable, GiB) about every 15 s during the rows, nodes named by rank |
| `rowcmp-b.log`, `rowcmp-s.log`, `rowcmp-n.log`, `rowcmp-sn.log` | each launch's 16 rows against a's (one decimal), then the verdict |
| `exact-chat.json`, `exact-chat-served.json` | the exact-chat check: 71 chats, each case's `/tokenize` ids against those of DeepSeek's reference encoding (`equal`) |
| `api-gates.json`, `api-gates-served.json`, `parallel-summary.json` | `gates.py` (6 API gates); 16 replies as a burst and staggered, against solo |
| `vision-gate.json`, `visionref.json`, `vision-four-at-a-time.json` | the image gate (6 checks); 9 image prompts' token ids; the same 9 four at a time, twice |
| `shield.json`, `shield-served.json` | the prompt shield: 14 conversations' token ids, with and without spelled special tokens |
| `structured.json`, `structured-served.json` | 12 structured requests solo, as a burst and staggered. As in v0.3.0, the verdict counts the 12th, `tool_choice` "required", as failed (the check expects a refusal); the forced-call files judge the calls |
| `dsml.json`, `dsml-served.json`, `dsml-before.json` | 16 rows of "required", named and "auto" tool calls (eight requests, whole and streamed); `dsml-before.json` on a, every row equal to the other two |
| `forced-calls.json`, `forced-calls-served.json`, `forced-calls-before.json` | 36 sampled forced requests: each row's calls, finish and counts of the DSML marker's token; `forced-calls-before.json` on a |
| `accept-stats.json`, `accept-stats-served.json` | the draft head's accepted chains by depth, from `/metrics` |
| `sampler.json` | sampling with `top_k` off and above 56 |
| `reply-ids.json` | 16 replies that quote the reasoning end or the DSML marker, `TENSORFOLD_REPLY_IDS` unset |
| `keepalive.json`, `keepalive-served.json` | two streamed 97,510-token prompts at once: each SSE comment's time and text, the longest silence before the first token |
| `kept-states.json` | eight turns over kept states: image turns, and two conversations on one system prompt |
| `resume.json` | four prompts of one text: the resume when the newest kept state ends near a prompt's end |
| `fill.json` | a stream's stalls while a 65,010-token prompt fills, `TF_FILL_CHUNKS` unset |

### `cmpp16/` and `cmpp25/`: the window raised on v0.3.0's build, engine `4964e38` (2026-10-21)

Each session launched the served build twice, one launch after the other: `cmpp16` with 4,194,304 positions (a),
then 16,777,216 (b); `cmpp25` with 16,777,216 (a), then 25,165,824 (b). b ran a's checks and compared every
reply with a's; then the session promoted b and checked the served port.

| File | What |
|---|---|
| `result.txt` | the session's 47 check lines, its prefill and memory lines, and the promotion |
| `knob-a-prefill.json`, `knob-b-prefill.json` | a 93,335-token prompt's prefill on a and on b, three reps (`summary` holds the median); the prompt file is named without its folder |
| `mem-a-watch.txt`, `mem-b-watch.txt`, `mem-served-watch.txt` | each node's free memory (MemAvailable, GiB) about every 15 s during a's checks, b's checks and on the served port, nodes named by rank |

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
