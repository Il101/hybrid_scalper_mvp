"""
Train meta-model on generated feature dataset.
Loads data/features.parquet, trains XGBoost + calibration, saves artifacts.
Enhanced with observability: auto-logging of artifacts, metrics, and model metadata.
"""
from __future__ import annotations
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import joblib

try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception as e:
    # On macOS, xgboost may fail to load if libomp is missing; fall back gracefully.
    print(f"Warning: XGBoost unavailable ({e}); falling back to sklearn models. Tip: brew install libomp")
    xgb = None  # type: ignore
    HAS_XGB = False

def train_and_evaluate(X, y, model_type='rf'):
    """Train model with cross-validation and return trained model + detailed observability"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model configuration with observability
    model_config = {}
    
    if model_type == 'xgb' and HAS_XGB and xgb is not None:
        model_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        model = xgb.XGBClassifier(**model_config)  # type: ignore
    elif model_type == 'rf':
        model_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'random_state': 42
        }
        model = RandomForestClassifier(**model_config)
    else:  # logistic regression
        model_config = {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'lbfgs'
        }
        model = LogisticRegression(**model_config)
    
    # Train model with timing
    train_start = datetime.now()
    model.fit(X_train, y_train)
    train_duration = (datetime.now() - train_start).total_seconds()
    
    # Predictions with timing
    pred_start = datetime.now()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    pred_duration = (datetime.now() - pred_start).total_seconds()
    
    # Comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix for detailed analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Time-based cross-validation (respects temporal order)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
    cv_auc_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc')
    
    metrics = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'cv_auc_mean': float(cv_auc_scores.mean()),
        'cv_auc_std': float(cv_auc_scores.std()),
        'cv_accuracy_scores': cv_scores.tolist(),
        'cv_auc_scores': cv_auc_scores.tolist(),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'training_time_seconds': train_duration,
        'prediction_time_seconds': pred_duration,
        'samples_train': len(X_train),
        'samples_test': len(X_test)
    }
    
    observability = {
        'model_config': model_config,
        'feature_stats': {
            'n_features': X.shape[1],
            'feature_means': X.mean().to_dict(),
            'feature_stds': X.std().to_dict(),
            'feature_nulls': X.isnull().sum().to_dict()
        },
        'target_distribution': {
            'class_counts': y.value_counts().to_dict(),
            'class_proportions': (y.value_counts() / len(y)).to_dict()
        }
    }
    
    return model, metrics, observability, X_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to features.parquet')
    parser.add_argument('--outdir', required=True, help='Output directory for artifacts')
    parser.add_argument('--model_type', default='rf', choices=['xgb', 'rf', 'lr'],
                       help='Model type: xgb, rf (random forest), or lr (logistic regression)')
    parser.add_argument('--target', default='label_win', help='Target column name')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if not col.startswith(('label_', 'timestamp', 'symbol', 'tf'))]
    X = df[feature_cols].copy()
    y = df[args.target].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"Features: {feature_cols}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    if len(y.unique()) < 2:
        raise ValueError(f"Target column '{args.target}' must have at least 2 classes")
    
    # Train model
    print(f"Training {args.model_type} model...")
    model, metrics, observability, X_test, y_test = train_and_evaluate(X, y, args.model_type)
    
    # Calibrate probabilities
    print("Calibrating model...")
    calibrator = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrator.fit(X, y)
    
    # Test calibrated predictions
    cal_proba = calibrator.predict_proba(X_test)[:, 1]
    cal_pred = calibrator.predict(X_test)
    
    cal_accuracy = accuracy_score(y_test, cal_pred)
    cal_auc = roc_auc_score(y_test, cal_proba)
    
    metrics['calibrated_accuracy'] = float(cal_accuracy)
    metrics['calibrated_auc'] = float(cal_auc)
    
    # Feature importance (if available)
    feature_importance = {}
    feature_importance_details = {}
    try:
        if args.model_type in ['rf', 'xgb']:
            # Tree-based models have feature_importances_
            importances = getattr(model, 'feature_importances_', None)
            if importances is not None:
                feature_importance = dict(zip(feature_cols, importances.tolist()))
                # For Random Forest, add additional details
                if args.model_type == 'rf' and isinstance(model, RandomForestClassifier):
                    # Individual tree importances (sample from first 10 trees)
                    tree_importances = []
                    for i, tree in enumerate(model.estimators_[:10]):
                        tree_imp = dict(zip(feature_cols, tree.feature_importances_.tolist()))
                        tree_importances.append({'tree_id': i, 'importances': tree_imp})
                    
                    feature_importance_details = {
                        'individual_trees': tree_importances,
                        'n_estimators_used': len(model.estimators_),
                        'oob_score': getattr(model, 'oob_score_', None),
                        'max_features': getattr(model, 'max_features_', None),
                        'min_samples_split': getattr(model, 'min_samples_split', None),
                        'min_samples_leaf': getattr(model, 'min_samples_leaf', None)
                    }
        elif args.model_type == 'lr':
            # Linear models have coef_
            coef = getattr(model, 'coef_', None)
            if coef is not None and len(coef) > 0:
                feature_importance = dict(zip(feature_cols, coef[0].tolist()))
    except Exception as e:
        print(f"Warning: Could not extract feature importance: {e}")
        feature_importance = {}
    
    # Enhanced artifact system with timestamping and versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.outdir, exist_ok=True)
    
    # Create versioned artifact directory
    versioned_dir = os.path.join(args.outdir, f"version_{timestamp}")
    os.makedirs(versioned_dir, exist_ok=True)
    
    # Main artifacts (always overwrite these for latest version)
    model_path = os.path.join(args.outdir, 'meta_model.pkl')
    calib_path = os.path.join(args.outdir, 'calibrator.pkl')
    report_path = os.path.join(args.outdir, 'report.json')
    
    # Versioned artifacts (for historical tracking)
    versioned_model_path = os.path.join(versioned_dir, 'meta_model.pkl')
    versioned_calib_path = os.path.join(versioned_dir, 'calibrator.pkl')
    versioned_report_path = os.path.join(versioned_dir, 'report.json')
    
    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    joblib.dump(model, versioned_model_path)
    
    print(f"Saving calibrator to {calib_path}")
    joblib.dump(calibrator, calib_path)
    joblib.dump(calibrator, versioned_calib_path)
    
    # Detailed classification report
    class_report = classification_report(y_test, cal_pred, output_dict=True)
    
    # Enhanced report with comprehensive observability
    report = {
        'metadata': {
            'model_type': args.model_type,
            'training_timestamp': timestamp,
            'data_source': args.data,
            'target_column': args.target,
            'sklearn_version': joblib.__version__,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        },
        'dataset_info': {
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'feature_columns': feature_cols,
            'target_distribution': observability['target_distribution']
        },
        'model_configuration': observability['model_config'],
        'performance_metrics': {
            'base_model': metrics,
            'calibrated_model': {
                'accuracy': float(cal_accuracy),
                'auc': float(cal_auc)
            }
        },
        'feature_analysis': {
            'importance': feature_importance,
            'importance_details': feature_importance_details,
            'statistics': observability['feature_stats']
        },
        'detailed_classification_report': class_report,
        'observability': {
            'training_time_seconds': metrics['training_time_seconds'],
            'prediction_time_seconds': metrics['prediction_time_seconds'],
            'cross_validation': {
                'accuracy_scores': metrics.get('cv_accuracy_scores', []),
                'auc_scores': metrics.get('cv_auc_scores', [])
            },
            'model_artifacts': {
                'model_path': model_path,
                'calibrator_path': calib_path,
                'versioned_artifacts_dir': versioned_dir
            }
        }
    }
    
    # Save main report
    print(f"Saving comprehensive report to {report_path}")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save versioned report  
    with open(versioned_report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create observability log entry
    obs_log_path = os.path.join(args.outdir, 'training_log.jsonl')
    log_entry = {
        'timestamp': timestamp,
        'model_type': args.model_type,
        'accuracy': metrics['accuracy'],
        'auc': metrics['auc'],
        'calibrated_accuracy': float(cal_accuracy),
        'calibrated_auc': float(cal_auc),
        'training_duration': metrics['training_time_seconds'],
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'version_dir': versioned_dir
    }
    
    # Append to training log (JSONL format for easy parsing)
    with open(obs_log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"Added entry to observability log: {obs_log_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {args.model_type}")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {args.target}")
    print()
    print("PERFORMANCE:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  AUC: {metrics['auc']:.3f}")
    print(f"  CV Mean: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
    print()
    print("CALIBRATED PERFORMANCE:")
    print(f"  Accuracy: {metrics['calibrated_accuracy']:.3f}")
    print(f"  AUC: {metrics['calibrated_auc']:.3f}")
    print()
    
    if feature_importance:
        print("TOP 5 FEATURES:")
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, imp in sorted_features:
            print(f"  {feat}: {imp:.4f}")
    
    print(f"\nArtifacts saved to: {args.outdir}")
    print(f"Versioned artifacts: {versioned_dir}")
    print(f"Observability log: {obs_log_path}")
    print("="*50)

if __name__ == '__main__':
    main()
