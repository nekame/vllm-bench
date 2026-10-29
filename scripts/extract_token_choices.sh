#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash extract_token_choices.sh token-8210.json" >&2
    exit 2
fi

exec jq '.matches[] | {
  request,
  position,
  cycle,
  token_id,
  source,
  source_row,
  accepted,
  proposed,
  actual_target_argmax,
  raw_top2,
  processed_top2,
  conditioning_input_ids,
  conditioning_positions,
  previous_verification
}' "$1"
