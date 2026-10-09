# Running it

What follows is how we run the served build. `scripts/serve.sh` is our served launch with the site values moved to
`config.env`; nothing else in it changed.

## 1. The engine

Clone our TensorFold fork at the commit in `ENGINE_COMMIT` (branch `deepseek-v41-zig-tp4`; the commit is the served
model code plus the fork's NOTICE, ATTRIBUTION.md and README note) and build the native engine for the GB10's
aarch64 + sm_121 with Zig 0.17.0 and the CUDA toolchain of `nvcr.io/nvidia/pytorch:26.07-py3`:

```sh
zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig build test        # the host-side unit tests (no GPU)
```

Copy `zig-out/native/bin/tensorfold-native` and `zig-out/bin/tf-dsv41-lanes` to `ENGINE_DIR` on every node. The
served launch also passes `rdma_gather.fatbin`, the RDMA kernels of `zig/kernels/cuda/dsv41/rdma_gather.cu` built
as a fatbin for sm_121; without `TF_DS_RDMA_KERNELS` the engine uses its embedded copy.

## 2. Weights and run-time inputs (on every node)

- `MODEL_DIR`: [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw).
- `ENGRAM_DIR`: DeepSeek's original shards 47 and 48 of
  [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (the Engram tables).
- `CACHE_DIR/rank-cache`: each rank's weight file (the four-rank split with the balanced experts), and
  `CACHE_DIR/dsv41_token_map.json`, the compressed token map. The engine reads both; our Python family writes them
  (the four-rank file with its four-node line, not published yet). v0.1.0 does not script them.
- `KIT_DIR`, the kernel kit: `aot/` (the family's Triton kernels, compiled and captured by the recorder in the engine's
  `tools/dsv41-zig/rec/`), `cubins/` (its extension kernels), the RoPE tables (`rope-*.f32`, `rope.json`), `engram.json`,
  and for images `vision/gate_bias_vl.safetensors` (the checkpoint's own `ffn.gate.bias_vl` tensors, read from the
  original shards) and `vision/torch_fmha_sm120.cubin` (PyTorch's attention kernel, extracted from the container's
  `libtorch_cuda.so`). A vision checkpoint's server refuses to start without `vision/`. The kit holds compiled
  third-party kernels and is not redistributed; v0.1.0 does not script it either.

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
