#!/usr/bin/env bash
# Forward the voice UI from the GPU VM to http://localhost:8000/
# usage: deploy/tunnel.sh <vm-ip> [ssh-key]
set -euo pipefail
IP="${1:?usage: tunnel.sh <vm-ip> [ssh-key]}"
KEY="${2:-$HOME/.ssh/verda_cpu_runner_20260830}"
tmux kill-session -t glm4voice-ui-tunnel 2>/dev/null || true
tmux new-session -d -s glm4voice-ui-tunnel \
  "ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -i '$KEY' -L 8000:127.0.0.1:8890 root@$IP"
sleep 2
curl -s -o /dev/null -w "http://localhost:8000/ -> %{http_code}\n" http://localhost:8000/
