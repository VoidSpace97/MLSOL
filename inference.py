#!/usr/bin/env python3
"""
Real-time inference module for generating trading signals every 60 seconds using a trained LightGBM model.
The script loads the latest model, retrieves the associated threshold from the database, and stores predictions.
"""
import csv
import os
import yaml
import argparse
import time
from datetime import datetime, timezone
from glob import glob
import lightgbm as lgb
import pandas as pd
import numpy as np
from feature_engineering import prepare_live_features

# ------------------------------------------------------------------------------
# CONFIGURATION AND DATABASE INITIALIZATION
# ------------------------------------------------------------------------------

# Load YAML configuration
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

SYMBOL = config['model'].get('target_symbol')

# ------------------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------------------

# Load model filename
optuna_models = sorted(glob("optuna_experiments/study_*/best_model.txt"))
if not optuna_models:
    raise FileNotFoundError("Nessun modello salvato in optuna_experiments trovato.")

model_path = optuna_models[-1]

# Load trained LightGBM model
model = lgb.Booster(model_file=model_path)
print(f"[INFO] Loaded model from {model_path} for inference.")

threshold_path = os.path.join(os.path.dirname(optuna_models[-1]), "best_trial.txt")
threshold_buy = 0.6  # default fallback
threshold_sell = 0.6
found_thresholds = False
with open(threshold_path, "r") as f:
    in_params_section = False
    for line in f:
        line = line.strip()
        if line.startswith("Params:"):
            in_params_section = True
            continue
        if in_params_section:
            if "threshold_buy" in line:
                threshold_buy = float(line.split(":")[1].strip())
                found_thresholds = True
            elif "threshold_sell" in line:
                threshold_sell = float(line.split(":")[1].strip())
                found_thresholds = True
if not found_thresholds:
    raise ValueError("Threshold non trovato in best_trial.txt")

# Extract model version
model_dir = os.path.dirname(model_path)  # e.g., "optuna_experiments/study_20250331_025219"
base_name = os.path.basename(model_dir)  # "study_20250331_025219"
model_version = base_name.replace("study_", "")  # "20250331_025219"

# Prepare output CSV path for this inference session
os.makedirs("csv", exist_ok=True)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
inference_csv_path = f"csv/inference_preds_{timestamp_str}.csv"

# ------------------------------------------------------------------------------
# INFERENCE + CSV STORAGE + CLASSIFICATION
# ------------------------------------------------------------------------------

def predict_and_store():
    """
    Perform live prediction and insert results into model_predictions table.
    """
    try:
        X_live, feature_cols, df_live = prepare_live_features(SYMBOL, return_df=True)
        features_file_path = os.path.join("csv", run_id, "inference_feature_cols.txt")
        with open(features_file_path, "w") as f:
            for col in df_live.columns:
                f.write(col + "\n")
        prediction_prob = model.predict(X_live)[0]
        pred_sell = prediction_prob[0]
        pred_hold = prediction_prob[1]
        pred_buy  = prediction_prob[2]
        predicted_class, confidence = classify_signal(prediction_prob, threshold_buy=threshold_buy, threshold_sell=threshold_sell)
        timestamp = datetime.now(timezone.utc).replace(second=0, microsecond=0)

        # Format timestamp for CSV
        row = {
            "timestamp": timestamp.isoformat(),
            "symbol": SYMBOL,
            "pred_prob_sell": pred_sell,
            "pred_prob_hold": pred_hold,
            "pred_prob_buy":  pred_buy,
            "prediction_prob": confidence,
            "threshold_sell": threshold_sell,
            "threshold_buy": threshold_buy,
            "predicted_class": predicted_class
        }

        file_exists = os.path.isfile(inference_csv_path)
        with open(inference_csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


        print(f"Prediction: {predicted_class} | Confidence: {confidence:.4f} | Buy ≥ {threshold_buy}, Sell ≥ {threshold_sell}")


    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")

def classify_signal(pred_probs, threshold_buy=0.6, threshold_sell=0.6):
    """
    Indice 0 => SELL, 1 => HOLD, 2 => BUY
    """
    class_idx = np.argmax(pred_probs)
    confidence = pred_probs[class_idx]

    if class_idx == 2 and confidence >= threshold_buy:
        return 2, confidence
    elif class_idx == 0 and confidence >= threshold_sell:
        return 0, confidence
    else:
        return 1, confidence
# ------------------------------------------------------------------------------
# MAIN EXECUTION LOOP
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="Unique run identifier")
    args = parser.parse_args()
    if args.run_id is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{model_version}_{timestamp_str}"
    else:
        run_id = args.run_id
    os.makedirs(f"csv/{run_id}", exist_ok=True)
    with open(f"csv/{run_id}/model_version.txt", "w") as f:
        f.write(model_version + "\n")
    inference_csv_path = f"csv/{run_id}/inference_preds.csv"
    print(f"[INFO] Starting inference loop for run_id={run_id} using model {model_version}...")
    try:
        while True:
            now = datetime.now(timezone.utc)
            time.sleep(60 - (now.second % 60))  # wait until next full minute
            predict_and_store()
    except KeyboardInterrupt:
        print("[INFO] Inference loop stopped by user.")
