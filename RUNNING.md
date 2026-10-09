# Running it

What follows is how we run the served build. `scripts/serve.sh` is our served launch with the site values moved to
`config.env`; nothing else in it changed but its `inputs` command (new in v0.1.1), which the launch never runs.

## 1. The engine

Clone our TensorFold fork at the commit in `ENGINE_COMMIT` (branch `deepseek-v41-zig-tp4`; the commit is the served
model code plus the fork's NOTICE, ATTRIBUTION.md and README note, and the tools that make each node's inputs and the
kit's generated files) and build the native engine for the GB10's aarch64 + sm_121 with Zig 0.17.0 and the CUDA
toolchain of `nvcr.io/nvidia/pytorch:26.07-py3`:

```sh
zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig build test        # the host-side unit tests (no GPU)
```

Copy `zig-out/native/bin/tensorfold-native` and `zig-out/bin/tf-dsv41-lanes` to `ENGINE_DIR` on every node, and for
section 2 `zig-out/bin/tf-dsv41-rank-cache` with `tools/dsv41-zig/inputs/node_inputs.sh` and `make_token_map.py`.
The served launch also passes `rdma_gather.fatbin`, the RDMA kernels of `zig/kernels/cuda/dsv41/rdma_gather.cu` built
as a fatbin for sm_121; without `TF_DS_RDMA_KERNELS` the engine uses its embedded copy.

## 2. Weights and run-time inputs (on every node)

- `MODEL_DIR`: [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw).
- `ENGRAM_DIR`: DeepSeek's original shards 47 and 48 of
  [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (the Engram tables).
- `CACHE_DIR`: the node's weight file `rank-cache/rank<g>of4-<hash>.bin` (its slices of the checkpoint for the
  four-node split with the balanced experts, in the loader's order: about 56 GB) and `dsv41_token_map.json` (Engram's
  compressed token map). Make both from the checkpoint, one command a node, no GPU:

  ```sh
  ENGINE_DIR/node_inputs.sh MODEL_DIR CACHE_DIR <g>     # g: the node's rank, 0-3, in the order of NODES
  scripts/serve.sh inputs                               # or: the same on the four nodes at once
  ```

  `node_inputs.sh` runs `tf-dsv41-rank-cache`, the engine's writer of the weight file, and `make_token_map.py`, which
  needs the `tokenizers` package (without it on the host, it runs in `IMAGE` with docker). `CACHE_DIR/rank-cache` must
  not hold another file of the same rank. On our nodes the four took 109 s together;
  `results/inputs-5940208/inputs.sha256` lists what they came out as (byte for byte the files our served lane loads),
  so `sha256sum` checks yours.
- `KIT_DIR`, the kernel kit: this repository's `kit/` (see `kit/README.md`): `aot/` (the family's Triton kernels,
  compiled and captured by the recorder in the engine's `tools/dsv41-zig/rec/`), `cubins/` (its extension kernels),
  `engram.json`, the RoPE tables and their digests (`rope.json`), and for images `vision/`. Git holds all of it but the
  four RoPE tables and `vision/gate_bias_vl.safetensors` (43 small tensors of DeepSeek's weights). Make those and check
  every file against `kit/MANIFEST`:

  ```sh
  kit/verify_kit.sh MODEL_DIR [ORIGINAL_DIR]     # ends with "verify_kit: 356 of 356 files match MANIFEST"
  ```

  It computes the RoPE tables on the CPU (in `nvcr.io/nvidia/pytorch:26.07-py3` with docker when the host has no
  PyTorch) and copies the image bias out of DeepSeek's original checkpoint when you pass its folder as `ORIGINAL_DIR`.
  Without one, take `vision/gate_bias_vl.safetensors` from the `kit/` folder of
  [bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP4-4xGB10/tree/main/kit),
  which also holds the whole kit as a download. Copy the checked `kit/` to `KIT_DIR` on every node. A vision
  checkpoint's server refuses to start without `vision/`.

## 3. Serve

```sh
cp config.env.example config.env     # fill in every <PLACEHOLDER>
scripts/serve.sh print               # what start will run on each node
scripts/serve.sh start               # ranks 3, 2, 1, then rank 0; waits for /v1/models, then a 4-token chat
scripts/serve.sh check
scripts/serve.sh stop
```

The server listens on `HOST:PORT` of rank 0 with TensorFold's OpenAI-compatible API. Images go in user turns or tool
results as `image_url` parts (data URLs). Rank 0 prints `{"vision": "Pillow 12.3.0 decodes the formats other than
PNG"}` when it serves images.

## 4. Check

`tools/` holds the clients that produced `results/`: `kit_bench.py` (the speed rows), `long_check.py` (long prompts:
concurrent equals solo, needles, kept-prompt reuse), `lensweep.py` (prompt lengths), `gates.py` (the API gates), and
`tools/vision/` (the image gate `vision_http.py`; `visionref.py` and `vconc.py` against `lane-ref.json`, our reference
lane's token ids for the pictures in `tools/vision/images/cmp`; `vprobe.py` and `vprobe_cmp.py` against
`lane-probe.json`, with the format corpus `vfmt.py` builds). The lane and server gates and the 16-reply check compare
against recorded reference replies this release does not ship.
