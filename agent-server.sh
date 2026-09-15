#!/bin/bash
# ============================================================================
#  agent-server.sh — run a local model as the benchmark orchestrator
# ============================================================================
#
#  THE PROBLEM THIS SOLVES
#
#  You want an agent to drive the benchmarks. But the agent is itself a model, and
#  it needs memory. On a single GPU the agent and the model under test compete for
#  the same VRAM:
#
#    - an agent on the GPU takes ~12 GB, so a 18 GB model under test no longer
#      fits alongside its KV cache, and
#    - every result is depressed anyway, because the card is doing two jobs at once.
#
#  SOLUTION: run the agent on the CPU (-ngl 0). The GPU stays 100% free for the
#  model being measured. Orchestration is a dozen tool calls, so 5-10 tok/s from
#  the CPU is entirely sufficient. The measurement is the expensive part; talking
#  to the agent is not.
#
#  This is a deliberate trade-off, not a limitation. With more than one GPU you can
#  give the agent its own card - see the note at the bottom of this file.
#
#  USAGE
#    ./agent-server.sh              start on CPU (default, recommended)
#    ./agent-server.sh --gpu        start on the GPU (ONLY when not measuring)
#    ./agent-server.sh --stop       stop it
#    ./agent-server.sh --status     is it running, and on what
#
#  ENVIRONMENT
#    AGENT_MODEL   path to the .gguf to serve   (default: $HOME/models/agent.gguf)
#    AGENT_PORT    listen port                  (default: 8080)
#    AGENT_HOST    bind address                 (default: 127.0.0.1)
#    LLAMA_SERVER  path to llama-server         (default: $HOME/llama.cpp/build/bin/llama-server)
#
#  NOTE ON BINDING: the default is 127.0.0.1 on purpose. llama-server has no
#  authentication; bind it to 0.0.0.0 only on a network you trust, or put it behind
#  an SSH tunnel:  ssh -L 8080:127.0.0.1:8080 <host>
#
#  License: MIT
# ============================================================================

set -uo pipefail

MODEL="${AGENT_MODEL:-$HOME/models/agent.gguf}"
PORT="${AGENT_PORT:-8080}"
HOST="${AGENT_HOST:-127.0.0.1}"
SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
PIDFILE=/tmp/agent-server.pid
LOGFILE="${AGENT_LOG:-/tmp/agent-server.log}"
THREADS="$(nproc)"

usage() { sed -n '1,45p' "$0"; exit 0; }

status() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        local pid mode
        pid=$(cat "$PIDFILE")
        if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qE '\-ngl 0'; then
            mode=CPU
        else
            mode=GPU
        fi
        echo "  running (pid $pid), mode: $mode, port $PORT"
        if [[ "$mode" == "GPU" ]]; then
            echo "  WARNING: the agent is holding the GPU. Do NOT benchmark now."
        else
            echo "  GPU is free for benchmarking - this is what you want"
        fi
        curl -sS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
            && echo "  API responding: http://127.0.0.1:$PORT/v1" \
            || echo "  API not up yet (model still loading)"
    else
        echo "  not running"
    fi
}

stop() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        kill "$(cat "$PIDFILE")" && echo "  stopped"
        rm -f "$PIDFILE"
    else
        echo "  was not running"
    fi
}

MODE=cpu
case "${1:-}" in
    --stop)    stop; exit 0 ;;
    --status)  status; exit 0 ;;
    --gpu)     MODE=gpu ;;
    --cpu|"")  MODE=cpu ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1 (try -h)"; exit 1 ;;
esac

[[ -f "$MODEL" ]]  || { echo "ERROR: model not found: $MODEL"; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: llama-server not found: $SERVER"; exit 1; }

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Already running:"; status; exit 0
fi

if [[ "$MODE" == "cpu" ]]; then
    # -ngl 0 = zero layers on the GPU. The card stays free for measurement.
    # Blanking HIP_VISIBLE_DEVICES makes sure nothing touches the GPU at all.
    echo "Starting agent on CPU ($THREADS threads) - GPU stays free for benchmarks"
    HIP_VISIBLE_DEVICES="" nohup "$SERVER" -m "$MODEL" -ngl 0 -t "$THREADS" \
        --host "$HOST" --port "$PORT" --ctx-size 16384 --jinja \
        > "$LOGFILE" 2>&1 &
else
    echo "Starting agent on the GPU - do NOT run benchmarks while this is up"
    nohup "$SERVER" -m "$MODEL" -ngl 999 \
        --host "$HOST" --port "$PORT" --ctx-size 32768 --jinja \
        > "$LOGFILE" 2>&1 &
fi

echo $! > "$PIDFILE"
echo "  pid $(cat "$PIDFILE"), log: $LOGFILE"
echo "  waiting for the API..."
for _ in $(seq 1 90); do
    if curl -sS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "  ready: http://127.0.0.1:$PORT/v1"
        echo
        echo "  Point your agent at that endpoint as an OpenAI-compatible provider,"
        echo "  and give it docs/AGENT-PLAYBOOK.md as its system prompt."
        exit 0
    fi
    sleep 2
done
echo "  API did not respond within 180 s - check $LOGFILE"
exit 1

# ---------------------------------------------------------------------------
# WITH MORE THAN ONE GPU
#
# Give the agent its own card so it stops being slow:
#
#   HIP_VISIBLE_DEVICES=3 ./agent-server.sh --gpu
#
# and benchmark on the others. Remember to narrow HIP_VISIBLE_DEVICES in the
# benchmark config too, otherwise bench.py will see the agent's card and the
# result describes two processes sharing one GPU rather than the model.
# ---------------------------------------------------------------------------
