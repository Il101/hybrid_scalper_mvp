#!/usr/bin/env bash
set -euo pipefail
# One-command: dataset -> train -> enable
SYMBOL=${SYMBOL:-BTCUSDT}
TF=${TF:-5m}
HORIZON=${HORIZON:-20}
TAKE_BPS=${TAKE_BPS:-25}
STOP_BPS=${STOP_BPS:-18}
EXCHANGE=${EXCHANGE:-bybit}
MARKET_TYPE=${MARKET_TYPE:-futures}
LIMIT=${LIMIT:-2000}

python3 -m model.retrain \
  --symbol "$SYMBOL" \
  --tf "$TF" \
  --horizon "$HORIZON" \
  --take_bps "$TAKE_BPS" \
  --stop_bps "$STOP_BPS" \
  --exchange "$EXCHANGE" \
  --market_type "$MARKET_TYPE" \
  --limit "$LIMIT"
