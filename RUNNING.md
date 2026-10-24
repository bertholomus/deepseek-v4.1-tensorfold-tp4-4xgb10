# Running it

What follows is how we run the served build. `scripts/serve.sh` is our served launch with the site values moved to
`config.env`; nothing else in it changed but its `inputs` command, which the launch never runs. v0.3.5 changes the
served settings (section 3); the kit is v0.3.0's, and the weights and how they are readied are v0.2.0's.

## 1. The engine

Clone our TensorFold fork at the commit in `ENGINE_COMMIT` (branch `deepseek-v41-zig-tp4`). That commit is the
served engine's source (`ebf6ae33`: v0.3.3's `39155041` and the one waves commit v0.3.5 adds) carried on
the branch's public line as `0a0b69a` (v0.3.3's `b116c47` and that commit; section 3). The public
line adds two pull requests the served build did not carry: #30, the neighbor transport, which runs only with
`TF_DS_NEIGHBOR_RDMA=1`, and #31, a CPU test of the replay cut. `ebf6ae33`, built as below, gives the
served binaries but for nvcc's build ids. `ENGINE_COMMIT`'s binaries also hold #30's code, and their ranks exchange a
48-byte settings frame at the start (40 before), so run the same build on all four nodes. Build the native engine for
the GB10's aarch64 + sm_121 with Zig 0.17.0 and the CUDA toolchain of `nvcr.io/nvidia/pytorch:26.07-py3`:

```sh
zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig build test        # the host-side unit tests (no GPU)
```

Copy `zig-out/native/bin/tensorfold-native` and `zig-out/bin/tf-dsv41-lanes` to `ENGINE_DIR` on every node, and for
section 2 `zig-out/bin/tf-dsv41-rank-cache` with `tools/dsv41-zig/inputs/node_inputs.sh` and `make_token_map.py`.
The served launch also passes `rdma_gather.fatbin`, the RDMA kernels of `zig/kernels/cuda/dsv41/rdma_gather.cu` built
as a fatbin for sm_121; without `TF_DS_RDMA_KERNELS` the engine uses its embedded copy.

## 2. Weights and run-time inputs (on every node)

- `MODEL_DIR`: [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw)
  (we checked revision `a2f91e27`), readied for the engine by this repository's `tools/serving_1017.py`:

  ```sh
  python3 tools/serving_1017.py MODEL_DIR            # standard library only, no GPU; reads about 130 MB
  python3 tools/serving_1017.py MODEL_DIR --check    # exit 0 when both files are there and right
  ```

  It adds two files and changes no downloaded one. `serving-1017.safetensors` holds six tensors under the names and
  types the engine reads: the MTP markov head under `mtp.2.markov_head.embed.weight` and `.head.weight` (byte copies of
  the converter's `markov_w1` and `markov_w2`), and the Engram `q_weight` and `k_weight` of layers 1 and 14 widened from
  BF16 to F32 (exact). The engine maps every `*.safetensors` file of the folder in name order, a later file winning, so
  these four replace the BF16 ones. `chat_template.jinja` is `chat/chat_template.jinja`: the 1017 checkpoint ships no
  chat template, and the server needs one for chat requests. Both files' sha256 are in
  `results/served-1017/inputs.txt`.
- `ENGRAM_DIR`: the original checkpoint's shards 47 and 48 of
  [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (the Engram
  tables).
- `CACHE_DIR`: the node's weight file `rank-cache/rank<g>of4-<hash>.bin` (its slices of the checkpoint for the
  four-node split with the balanced experts, in the loader's order: about 76 GB) and `dsv41_token_map.json` (Engram's
  compressed token map). Make both from the readied checkpoint, one command a node, no GPU:

  ```sh
  ENGINE_DIR/node_inputs.sh MODEL_DIR CACHE_DIR <g>     # g: the node's rank, 0-3, in the order of NODES
  scripts/serve.sh inputs                               # or: the same on the four nodes at once
  ```

  `node_inputs.sh` runs `tf-dsv41-rank-cache`, the engine's writer of the weight file, and `make_token_map.py`, which
  needs the `tokenizers` package (without it on the host, it runs in `IMAGE` with docker). `CACHE_DIR/rank-cache` must
  not hold another file of the same rank. On our nodes the four took 146 s together. Each file's name holds the hash
  of its index; `results/served-1017/inputs.txt` lists the four names our served lane loads and the token map's sha256
  (the 1017 tokenizer is V4.1-Flash's, so the map is v0.1.1's). v0.3.0, v0.3.1, v0.3.2,
v0.3.3 and v0.3.5 load the same four files.
- `KIT_DIR`, the kernel kit: this repository's `kit/` (see `kit/README.md`): `aot/` (the family's Triton kernels,
  compiled and captured by the recorder in the engine's `tools/dsv41-zig/rec/`), `cubins/` (its extension kernels),
  `engram.json`, the RoPE tables and their digests (`rope.json`), for images `vision/`, and `grammar/`
  (`libtfgrammar.so`, the grammar library for structured output and forced tool calls, built for aarch64 Linux, with
  its licenses). Git holds all of it but the four RoPE tables and `vision/gate_bias_vl.safetensors` (43 small tensors
  of DeepSeek's 1017 weights). Make those and check every file against `kit/MANIFEST`:

  ```sh
  kit/verify_kit.sh MODEL_DIR [ORIGINAL_DIR]     # ends with "verify_kit: 362 of 362 files match MANIFEST"
  ```

  It computes the RoPE tables on the CPU (in `nvcr.io/nvidia/pytorch:26.07-py3` with docker when the host has no
  PyTorch) and copies the image bias out of the original 1017 checkpoint when you pass its folder as `ORIGINAL_DIR`
  (its index and the 43 shards that hold the bias tensors; our EXL3 quant has 40 of them, not the three MTP layers').
  Without one, take `vision/gate_bias_vl.safetensors` (sha256 `4745da4b...` in `kit/MANIFEST`) from the `kit/` folder
  of [bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10/tree/main/kit),
  which also holds the whole kit as a download; `verify_kit.sh` refuses any other file. The file v0.1.1 used comes
  from DeepSeek-V4.1-Flash and does not match. Copy the checked `kit/` to `KIT_DIR` on every node. A vision
  checkpoint's server refuses to start without `vision/`.
- What rank 0 opens at start. Rank 0 reads its weights from its weight file, but it also opens these, so its node needs
  them even when the other model shards are not there:
  - in `MODEL_DIR`: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` (from
    `tools/serving_1017.py`), `model.safetensors.index.json`, and the vision tower's shard,
    `model-00041-of-00041.safetensors` (266 tensors: `vision.*`, `aligner.*`, `image_start`, `image_newline`,
    `image_end`; the server loads the tower at start);
  - in `CACHE_DIR`: `rank-cache/rank0of4-<hash>.bin` and `dsv41_token_map.json`;
  - in `KIT_DIR`: the kit (`aot/aot.json` and the rest of it); in `ENGINE_DIR`: `tensorfold-native` and
    `rdma_gather.fatbin`; `ENGRAM_DIR`.

  `scripts/serve.sh start` checks these on rank 0's node first and, when one is missing, names it and starts
  nothing (it does not stop a running lane either). `scripts/serve.sh preflight` runs the check alone. It finds the
  vision shard through the index, so it also works with another shard layout.

## 3. Serve

```sh
cp config.env.example config.env     # fill in every <PLACEHOLDER>
scripts/serve.sh preflight           # exit 0 when every file rank 0 opens at start is there (section 2)
scripts/serve.sh print               # what start will run on each node
scripts/serve.sh start               # preflight, then ranks 3, 2, 1, then rank 0; /v1/models, then a 4-token chat
scripts/serve.sh check
scripts/serve.sh stop
```

The server listens on `HOST:PORT` of rank 0 with TensorFold's OpenAI-compatible API. Images go in user turns or tool
results as `image_url` parts (data URLs). Rank 0 prints `{"vision": "Pillow 12.3.0 decodes the formats other than
PNG"}` when it serves images. The chat template's header lists the request fields it reads (`thinking_mode` or
`enable_thinking`, `reasoning_effort`, the tools and the message fields).

Structured output takes `response_format` (`{"type": "json_schema", ...}` or `{"type": "json_object"}`),
`guided_choice` or `guided_regex`; with tools offered, `tool_choice` "required" or a named function answers with a
call. Rank 0 loads `TENSORFOLD_GRAMMAR_LIB` (the kit's `grammar/libtfgrammar.so`, mounted at `/kit`) at the first
such request. A streamed reply that has sent nothing for 15 s sends `: keepalive`, an SSE comment clients skip; rank
0's log gets a stats line (the draft head's accepted chains by depth) at most once a minute, and `/metrics` counts
them.

`config.env.example` holds the served settings: `ENGINE_ENV` (every rank's environment), `FOLLOW_FLAGS` (ranks 1-3) and
`RANK0_FLAGS` (rank 0's own flags after the site's `SERVER_FLAGS`). `TF_FILL_CHUNKS=1`, `TF_FILL_MANY=1`,
`TF_DS_2D_YX=1`, `TF_DS_WAVES=1` and `NCCL_MAX_NCHANNELS=4` are set as served; leave `TENSORFOLD_REPLY_IDS`,
`TF_DS_NEIGHBOR_RDMA`, `TF_DS_SOLO_OVERHEAD` and `TF_DS_NO_COPIES` unset to serve as we do; the README says what
each does when set.

The served window, 20,971,520 positions (`TF_DS_WINDOW` in `ENGINE_ENV`, `--pool` in `FOLLOW_FLAGS`), leaves about
12 GiB free on each node (12 to 14 GiB on ours at the lowest). On nodes that need more room, set 16,777,216 in both
places and in `WINDOW`, which records it: 18 to 20 GiB stayed free on ours in v0.3.1's sessions, with every reply
equal (README, v0.3.2).

## 4. Check

`tools/` holds the clients that produced `results/`: `kit_bench.py` (the speed rows), `long_check.py` (long prompts:
concurrent equals solo, needles, kept-prompt reuse), `needle.py` (needles in one long haystack; with `--shared` the
follow-up questions resume from the kept prompt), `lensweep.py` (prompt lengths), `gates.py` (the API gates), and
`tools/vision/` (the image gate `vision_http.py`; `visionref.py` and `vconc.py` against `lane-ref.json`, our reference
lane's token ids for the pictures in `tools/vision/images/cmp`; `vprobe.py` and `vprobe_cmp.py` against
`lane-probe.json`, with the format corpus `vfmt.py` builds). `lane-ref.json` and `lane-probe.json` were recorded on
the 2.9 bpw weights; on the 1017 weights we compare the four-at-a-time run with the solo run instead. The exact-chat
check (71 cases with DeepSeek's reference renderings), the 16-reply check, the copy rows (set c) and the checks of
structured output, forced tool calls, the prompt shield, kept states and large conversations are our lane tools and
are not in this release; their reports are in `results/served-v033/`, `results/served-v032/`, `results/served-v031/` and
`results/served-v030/`.
