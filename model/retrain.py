"""
End-to-end automation:
1) Build dataset (real OHLCV, orderbook features, news/smart money) -> data/features.parquet
2) Train meta-model (XGBoost/LogReg) -> model/artifacts/{meta_model.pkl, calibrator.pkl, report.json}
3) Enable meta-model in config.yaml

Usage:
  python -m model.retrain --symbol BTCUSDT --tf 5m --horizon 20 --take_bps 25 --stop_bps 18 \
    --exchange bybit --market_type futures --limit 2000

Requires:
  - Real-data pipeline configured (ccxt, CryptoPanic token in .env)
"""
from __future__ import annotations
import os, sys, argparse, json, yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 1) Build dataset
from backtest.build_dataset import build as build_dataset
# 2) Train meta
from model.train_meta import main as train_meta_main

CFG_PATH = "config.yaml"

def patch_config_enable_meta(path_model: str, path_calib: str, cfg_path: str = CFG_PATH):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mm = cfg.get("meta_model", {})
    mm["enabled"] = True
    mm["path"] = path_model
    mm["calibrator_path"] = path_calib
    # if thresholds are missing, set sane defaults
    mm.setdefault("thresholds", {}).setdefault("default", {"p_win": 0.55, "ev_min_pct": 0.05})
    cfg["meta_model"] = mm
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--take_bps", type=int, default=25)
    ap.add_argument("--stop_bps", type=int, default=18)
    ap.add_argument("--exchange", default="bybit")
    ap.add_argument("--market_type", default="futures")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--dataset_out", default="data/features.parquet")
    ap.add_argument("--artifacts_dir", default="model/artifacts")
    args = ap.parse_args()

    # Step 1: Dataset
    print("[1/3] Building dataset ...", flush=True)
    res = build_dataset(symbol=args.symbol, tf=args.tf, horizon=args.horizon,
                        take_bps=args.take_bps, stop_bps=args.stop_bps,
                        exchange_for_ob=args.exchange, out_path=args.dataset_out,
                        limit_rows=args.limit)
    print(f"Dataset built: {res}", flush=True)

    # Step 2: Train meta
    print("[2/3] Training meta-model ...", flush=True)
    # emulate CLI for train_meta
    sys.argv = ["train_meta", "--data", args.dataset_out, "--outdir", args.artifacts_dir]
    train_meta_main()

    model_path = os.path.join(args.artifacts_dir, "meta_model.pkl")
    calib_path = os.path.join(args.artifacts_dir, "calibrator.pkl")
    if not (os.path.exists(model_path) and os.path.exists(calib_path)):
        raise RuntimeError("Training did not produce meta_model.pkl or calibrator.pkl")

    # Step 3: Enable in config
    print("[3/3] Enabling meta-model in config.yaml ...", flush=True)
    patch_config_enable_meta(model_path, calib_path, CFG_PATH)
    print("Done. Meta-model is now enabled.", flush=True)

if __name__ == "__main__":
    main()
