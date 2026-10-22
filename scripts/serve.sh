#!/bin/bash
# The served DeepSeek-V4.1-Flash launch on four GB10 nodes (README "The served settings"), our served script with the
# site values moved to config.env. Ranks 3, 2 and 1 start `tf-dsv41-lanes` following rank 0, then rank 0 starts
# `tensorfold-native serve`; each runs in a container tp4-lane on its node (scripts/node_run.sh, fed over ssh, or run
# directly on the node that runs this script: no ssh key to itself needed).
#   scripts/serve.sh start    preflight, stop any tp4-lane, start the four ranks, wait for /v1/models, then check
#   scripts/serve.sh preflight  exit 0 when every file rank 0 opens at start is there (RUNNING.md section 2), else
#                             name each missing one; start runs it first and starts nothing when one is missing
#   scripts/serve.sh stop     remove tp4-lane on the four nodes
#   scripts/serve.sh check    exit 0 when rank 0 lists the model and answers a 4-token chat
#   scripts/serve.sh print    print what start runs on each node, run nothing
#   scripts/serve.sh inputs   make each node's weight file and token map in CACHE_DIR (ENGINE_DIR/node_inputs.sh on
#                             the four nodes at once; RUNNING.md section 2); the launch itself never runs it
set -u
here=$(cd "$(dirname "$0")" && pwd)
CFG=${DSV41_CONFIG:-$here/../config.env}
[ -r "$CFG" ] || { echo "no $CFG: copy config.env.example to config.env and fill it in" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
if grep -v '^[[:space:]]*#' "$CFG" | grep -q '<[A-Z0-9_]*>'; then echo "$CFG still holds a <PLACEHOLDER>" >&2; exit 1; fi
read -ra N <<< "$NODES"
[ ${#N[@]} = 4 ] || { echo "NODES must name four nodes, rank 0 first" >&2; exit 1; }
U=http://127.0.0.1:$PORT
PRINT=0
SITE="IMAGE=$IMAGE ENGINE_DIR=$ENGINE_DIR KIT_DIR=$KIT_DIR MODEL_DIR=$MODEL_DIR ENGRAM_DIR=$ENGRAM_DIR CACHE_DIR=$CACHE_DIR LOG_DIR=$LOG_DIR"
LINK="NCCL_IB_HCA=$RDMA_HCA TF_RDMA_DEVICES=$RDMA_HCA NCCL_SOCKET_IFNAME=$NET_IF"

# This host's names and addresses: a node in NODES that is one of them runs here, not over ssh to itself.
SELF=" localhost 127.0.0.1 ::1 $(hostname 2>/dev/null) $(hostname -s 2>/dev/null) $(hostname -f 2>/dev/null) $(hostname -I 2>/dev/null) "
is_local() {   # is_local NODE: NODE is this host (by name, by the name its ssh config gives it, or by its address)
  local real ip n
  real=$(ssh -G "$1" 2>/dev/null | awk '$1 == "hostname" { print $2; exit }')
  for n in "$1" "${real:-$1}"; do
    case "$SELF" in *" $n "*) return 0 ;; esac
    ip=$(getent ahosts "$n" 2>/dev/null | awk '{ print $1; exit }')
    [ -n "$ip" ] && case "$SELF" in *" $ip "*) return 0 ;; esac
  done
  return 1
}
via() {   # via NODE CMD: CMD on NODE (here when NODE is this host, else over ssh), stdin passed through
  if is_local "$1"; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi
}
on() { via "$1" "$2" </dev/null; }
rank() {   # rank NODE ENVS CMD: CMD in tp4-lane on NODE with ENVS (VAR=value words without spaces) in its environment
  if [ $PRINT = 1 ]; then echo "$1: env $2 $SITE $LINK node_run.sh tp4-lane '$3'"; return 0; fi
  via "$1" "env $2 $SITE $LINK bash -s -- tp4-lane '$3'" < "$here/node_run.sh" >/dev/null
}
# Every file rank 0 opens at start (RUNNING.md section 2): the model folder's small files, each file the index maps a
# vision tower tensor to (model-00041-of-00041.safetensors in the download), the node's weight file, the token map, the
# kit's kernel set and the engine. Checked on rank 0's node before anything starts or stops.
preflight() {
  local miss
  miss=$(via "${N[0]}" "bash -s -- '$MODEL_DIR' '$CACHE_DIR' '$KIT_DIR' '$ENGINE_DIR' '$ENGRAM_DIR'" <<'EOF'
m=$1
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja model.safetensors.index.json; do
  [ -s "$m/$f" ] || echo "MODEL_DIR/$f"
done
if [ -s "$m/model.safetensors.index.json" ]; then
  v=$(grep -oE '"(vision|aligner|image_(start|newline|end))[^"]*": *"[^"]+"' "$m/model.safetensors.index.json" |
      sed 's/.*: *"//; s/"$//' | sort -u)
  [ -n "$v" ] || echo "MODEL_DIR/model.safetensors.index.json names no vision tower file"
  for f in $v; do [ -s "$m/$f" ] || echo "MODEL_DIR/$f (the vision tower; rank 0 reads it at start)"; done
fi
ls "$2"/rank-cache/rank0of4-*.bin >/dev/null 2>&1 || echo "CACHE_DIR/rank-cache/rank0of4-<hash>.bin (scripts/serve.sh inputs)"
[ -s "$2/dsv41_token_map.json" ] || echo "CACHE_DIR/dsv41_token_map.json (scripts/serve.sh inputs)"
[ -s "$3/aot/aot.json" ] || echo "KIT_DIR/aot/aot.json"
for f in tensorfold-native rdma_gather.fatbin; do [ -s "$4/$f" ] || echo "ENGINE_DIR/$f"; done
[ -d "$5" ] || echo "ENGRAM_DIR $5"
EOF
  ) || { echo "preflight: could not reach rank 0's node ${N[0]}" >&2; return 1; }
  [ -z "$miss" ] && return 0
  echo "preflight: rank 0 (${N[0]}) is missing these files, nothing was started or stopped:" >&2
  echo "$miss" | sed 's/^/  /' >&2
  return 1
}
check() {
  on "${N[0]}" "curl -sf -m 10 $U/v1/models | grep -q $SERVED_NAME && curl -sf -m 120 -H 'Content-Type: application/json' $U/v1/chat/completions -d '{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":4,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}' | grep -q choices"
}
stop() {
  local h
  for h in "${N[@]}"; do on "$h" "docker rm -f tp4-lane >/dev/null 2>&1; true"; done
}
inputs() {
  local g rc=0 pids=()
  for g in 0 1 2 3; do
    ( set -o pipefail; on "${N[$g]}" "IMAGE=$IMAGE $ENGINE_DIR/node_inputs.sh $MODEL_DIR $CACHE_DIR $g" 2>&1 | sed "s/^/rank $g: /" ) &
    pids+=($!)
  done
  for g in 0 1 2 3; do wait "${pids[$g]}" || { echo "rank $g: node_inputs.sh failed" >&2; rc=1; }; done
  return $rc
}
start() {
  local g fam envs up=0
  if [ $PRINT = 0 ]; then preflight || return 1; stop; fi
  fam="--pool $CONTEXT --engram /engram --token-map /cache/dsv41_token_map.json --rdma $RDMA_HCA"
  fam="$fam --rdma-kernels /engine/rdma_gather.fatbin --graphs 1 --side 1 --prefetch 1 --aio 1 $FOLLOW_FLAGS"
  for g in 3 2 1; do
    rank "${N[$g]}" "$ENGINE_ENV" "/engine/tf-dsv41-lanes /model /cache/rank-cache $g 4 $MASTER_IP $TP_PORT /kit $fam" \
      || { echo "start rank $g failed" >&2; return 1; }
  done
  envs="TF_TP_WORLD=4 TF_TP_MASTER=$MASTER_IP TF_TP_PORT=$TP_PORT TF_DS_RANK_CACHE=/cache/rank-cache TF_DS_KIT=/kit"
  envs="$envs TENSORFOLD_CUDA_KERNELS=/kit/aot TF_DS_ENGRAM=/engram TF_DS_TOKEN_MAP=/cache/dsv41_token_map.json"
  envs="$envs TF_DS_RDMA_KERNELS=/engine/rdma_gather.fatbin TF_DS_GRAPHS=1 TF_DS_HC_SIDE=1 TF_DS_L2_PREFETCH=1"
  envs="$envs TF_DS_ENGRAM_AIO=1 $ENGINE_ENV"
  rank "${N[0]}" "$envs" "/engine/tensorfold-native serve /model --host $HOST --port $PORT --name $SERVED_NAME --context $CONTEXT --parallel $PARALLEL $SERVER_FLAGS --backend cuda${RANK0_FLAGS:+ $RANK0_FLAGS}" \
    || { echo "start rank 0 failed" >&2; return 1; }
  [ $PRINT = 1 ] && return 0
  for _ in $(seq 1 240); do
    sleep 5
    if on "${N[0]}" "curl -sf -m 5 $U/v1/models | grep -q $SERVED_NAME"; then up=1; break; fi
    on "${N[0]}" "docker ps --format '{{.Names}}' | grep -qx tp4-lane" || break
  done
  [ $up = 1 ] || { echo "rank 0 did not come up: see $LOG_DIR/tp4-lane.log on ${N[0]}" >&2; return 1; }
  check && echo "up and answering on ${N[0]}:$PORT" || { echo "up, but the chat check failed" >&2; return 1; }
}
case ${1:?start|stop|check|print|inputs|preflight} in
  start) start ;;
  preflight) preflight && echo "preflight: rank 0 (${N[0]}) has every file it opens at start" ;;
  stop) stop ;;
  check) check ;;
  print) PRINT=1; start ;;
  inputs) inputs ;;
  *) echo "start|stop|check|print|inputs|preflight" >&2; exit 2 ;;
esac
