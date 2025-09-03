#!/usr/bin/env python3
"""
Model observability viewer - analyze training history and artifacts.

Usage:
    python scripts/view_model_history.py                    # Show recent training runs
    python scripts/view_model_history.py --detailed         # Show detailed metrics  
    python scripts/view_model_history.py --compare          # Compare versions
"""
from __future__ import annotations
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_training_log(log_path: Path) -> pd.DataFrame:
    """Load training log JSONL into DataFrame"""
    if not log_path.exists():
        return pd.DataFrame()
    
    records = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
    return df.sort_values('datetime')


def show_summary(df: pd.DataFrame):
    """Show training runs summary"""
    if df.empty:
        print("No training runs found.")
        return
        
    print("=== MODEL TRAINING HISTORY ===")
    print(f"Total runs: {len(df)}")
    print(f"Date range: {df['datetime'].min().strftime('%Y-%m-%d %H:%M')} to {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # Recent runs
    print("RECENT TRAINING RUNS:")
    recent = df.tail(5)[['timestamp', 'model_type', 'accuracy', 'calibrated_accuracy', 'auc', 'calibrated_auc', 'training_duration']]
    recent['acc'] = recent['accuracy'].apply(lambda x: f"{x:.3f}")
    recent['cal_acc'] = recent['calibrated_accuracy'].apply(lambda x: f"{x:.3f}")
    recent['auc'] = recent['auc'].apply(lambda x: f"{x:.3f}")
    recent['cal_auc'] = recent['calibrated_auc'].apply(lambda x: f"{x:.3f}")
    recent['duration'] = recent['training_duration'].apply(lambda x: f"{x:.2f}s")
    
    display_cols = ['timestamp', 'model_type', 'acc', 'cal_acc', 'auc', 'cal_auc', 'duration']
    print(recent[display_cols].to_string(index=False))


def show_detailed(df: pd.DataFrame):
    """Show detailed analysis of training runs"""
    if df.empty:
        print("No training runs found.")
        return
        
    print("=== DETAILED PERFORMANCE ANALYSIS ===")
    
    # Performance trends
    print("\nPERFORMance STATISTICS:")
    metrics = ['accuracy', 'calibrated_accuracy', 'auc', 'calibrated_auc', 'training_duration']
    stats = df[metrics].describe()
    print(stats.round(4))
    
    # Best performers
    print("\nBEST PERFORMING RUNS:")
    best_cal_auc = df.loc[df['calibrated_auc'].idxmax()]
    best_cal_acc = df.loc[df['calibrated_accuracy'].idxmax()]
    fastest = df.loc[df['training_duration'].idxmin()]
    
    print(f"Best Calibrated AUC: {best_cal_auc['calibrated_auc']:.4f} ({best_cal_auc['timestamp']})")
    print(f"Best Calibrated Accuracy: {best_cal_acc['calibrated_accuracy']:.4f} ({best_cal_acc['timestamp']})")
    print(f"Fastest Training: {fastest['training_duration']:.3f}s ({fastest['timestamp']})")


def compare_versions(df: pd.DataFrame, n: int = 3):
    """Compare last N training versions"""
    if len(df) < 2:
        print("Need at least 2 training runs to compare.")
        return
        
    recent = df.tail(n)
    print(f"=== COMPARING LAST {len(recent)} VERSIONS ===")
    
    for i, (_, run) in enumerate(recent.iterrows()):
        print(f"\nVersion {i+1}: {run['timestamp']}")
        print(f"  Model: {run['model_type']}")
        print(f"  Accuracy: {run['accuracy']:.4f} → {run['calibrated_accuracy']:.4f} (calibrated)")
        print(f"  AUC: {run['auc']:.4f} → {run['calibrated_auc']:.4f} (calibrated)")
        print(f"  Training time: {run['training_duration']:.3f}s")
        print(f"  Samples: {run['n_samples']}, Features: {run['n_features']}")
    
    # Show improvements/degradations
    if len(recent) >= 2:
        latest = recent.iloc[-1]
        previous = recent.iloc[-2]
        
        print(f"\nCHANGES (latest vs previous):")
        acc_change = latest['calibrated_accuracy'] - previous['calibrated_accuracy']
        auc_change = latest['calibrated_auc'] - previous['calibrated_auc']
        time_change = latest['training_duration'] - previous['training_duration']
        
        print(f"  Calibrated Accuracy: {acc_change:+.4f}")
        print(f"  Calibrated AUC: {auc_change:+.4f}")
        print(f"  Training Time: {time_change:+.3f}s")


def main():
    parser = argparse.ArgumentParser(description="View model training history and observability")
    parser.add_argument('--log-path', default='model/artifacts/training_log.jsonl',
                        help='Path to training log file')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed performance analysis')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare recent training versions')
    parser.add_argument('--compare-n', type=int, default=3,
                        help='Number of recent versions to compare (default: 3)')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_path)
    df = load_training_log(log_path)
    
    if df.empty:
        print(f"No training data found in {log_path}")
        print("Run model training first:")
        print("  python model/train_meta.py --data data/features.parquet --outdir model/artifacts --model_type rf")
        return
    
    if args.compare:
        compare_versions(df, args.compare_n)
    elif args.detailed:
        show_detailed(df)
    else:
        show_summary(df)


if __name__ == '__main__':
    main()
