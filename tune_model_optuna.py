#!/usr/bin/env python3

import os
import time
import optuna
import optuna.visualization as vis
import plotly
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from feature_engineering import prepare_training_data

def objective(trial):
    """
    Funzione obiettivo per Optuna:
    1) Sceglie threshold_buy, threshold_sell, target_horizon
    2) Costruisce X, y con prepare_training_data
    3) Allena un modello LightGBM
    4) Calcola F1 macro e lo ritorna
    """

    # Queste soglie servono per definire la classe di target: SELL (0), HOLD (1), BUY (2)
    threshold_buy = trial.suggest_float("threshold_buy", 0.001, 0.01)
    threshold_sell = trial.suggest_float("threshold_sell", 0.001, 0.01)
    horizon = trial.suggest_int("target_horizon", 5, 30)

    # Prepara i dati di training con threshold dinamiche
    X, y, feature_cols = prepare_training_data(
        symbol="SOL/USDC",
        threshold_buy=threshold_buy,
        threshold_sell=threshold_sell,
        horizon=horizon
    )

    # Controllo banale per evitare errori se c'è 1 sola classe
    if len(np.unique(y)) < 2:
        return 0.0

    # Train/validation split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Parametri LightGBM da ottimizzare
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 40, 120, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.8),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "verbose": -1
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        valid_sets=[dval],
        num_boost_round=300,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=0)
        ]
    )

    # Calcolo F1 su validation
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    f1 = f1_score(y_val, y_pred, average='macro')
    return f1

def main():
    import optuna
    import time
    import os

    # Crea cartella radice
    root_dir = "optuna_experiments"
    os.makedirs(root_dir, exist_ok=True)

    # Subfolder con timestamp
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root_dir, f"study_{timestamp_str}")
    os.makedirs(run_dir, exist_ok=True)
    storage_name = f"sqlite:///{os.path.join(run_dir, 'study.db')}"

    study = optuna.create_study(direction="maximize",
                                study_name="lgb_tuning_study",
                                storage=storage_name,
                                load_if_exists=False)
    # Eseguiamo l'ottimizzazione
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("==========================================")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best F1: {study.best_trial.value:.4f}")
    print("Best parameters:")
    for key, val in study.best_trial.params.items():
        print(f"  {key}: {val}")

    # Salviamo best_trial.txt
    best_trial_path = os.path.join(run_dir, "best_trial.txt")
    with open(best_trial_path, "w") as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best F1: {study.best_trial.value}\n")
        f.write("Params:\n")
        for key, val in study.best_trial.params.items():
            f.write(f"  {key}: {val}\n")

    # Salviamo tutti i trial in un CSV
    trials_df = study.trials_dataframe(attrs=("number","value","params","state"))
    trials_df.to_csv(os.path.join(run_dir, "all_trials.csv"), index=False)

    # Ora ricostruiamo e salviamo il modello con i parametri migliori
    best_params = study.best_trial.params
    threshold_buy_best = best_params["threshold_buy"]
    threshold_sell_best = best_params["threshold_sell"]
    horizon_best = best_params["target_horizon"]

    # Prepara training set completo (stesso threshold, horizon)
    X, y, feature_cols = prepare_training_data(
        symbol="SOL/USDC",
        threshold_buy=threshold_buy_best,
        threshold_sell=threshold_sell_best,
        horizon=horizon_best
    )

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    final_params = {
        k: v for k, v in best_params.items()
        if k not in ["threshold_buy", "threshold_sell", "target_horizon"]
    }
    final_params.update({
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbose": -1,
    })

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params=final_params,
        train_set=dtrain,
        valid_sets=[dval],
        num_boost_round=300,
        callbacks=[
            lgb.early_stopping(30),
            lgb.log_evaluation(0)
        ]
    )

    model_path = os.path.join(run_dir, "best_model.txt")
    model.save_model(model_path)
    print(f"[INFO] Modello salvato in {model_path}")

    importance = model.feature_importance(importance_type="gain")
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importance
    }).sort_values(by="importance_gain", ascending=False)

    # Salva CSV
    importance_df.to_csv(os.path.join(run_dir, "feature_importance.csv"), index=False)

    # Plot (facoltativo se hai plotly o matplotlib installato)
    plt.figure(figsize=(10, 6))
    importance_df.head(30).plot(kind="barh", x="feature", y="importance_gain", legend=False)
    plt.title("Top 30 Feature Importance (gain)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "feature_importance_top30.png"))

        # ========== Grafici post-tuning ==========
    print("[INFO] Generazione grafici Optuna...")

    fig1 = vis.plot_param_importances(study)
    fig1.write_image(os.path.join(run_dir, "param_importance.png"))

    fig2 = vis.plot_optimization_history(study)
    fig2.write_image(os.path.join(run_dir, "optimization_history.png"))

    fig3 = vis.plot_slice(study)
    fig3.write_image(os.path.join(run_dir, "param_slice_plot.png"))

    print(f"[INFO] 📊 Grafici salvati in: {run_dir}")

if __name__ == "__main__":
    main()
