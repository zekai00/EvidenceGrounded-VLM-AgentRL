#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper with a v0.6 name. The underlying script keeps its old
# filename for backward compatibility but now defaults to v0.6 settings.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_verl_trajectory_grpo_smoke_v0_5.sh" "$@"
