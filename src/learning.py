"""
learning.py
===========
近赤外スペクトル分析チャレンジ — 学習モジュール（config駆動版）

使い方
------
    # グリッドサーチ → 最終モデル保存
    python src/learning.py --config configs/pls_snv_deriv2.yaml

    # LightGBM Optuna チューニング（グリッドサーチ上位N候補を最適化）
    python src/learning.py --config configs/lgbm_snv_pca.yaml \
                           --optimizer optuna --n_trials 100 --n_top 3

    # 実験結果の一覧表示
    python src/learning.py --show

設計方針
--------
- config（YAML）で前処理・モデル・探索パラメータをすべて制御する
- 前処理はすべて preprocessing.py の build_pipeline() 経由で行う
- GroupKFold の各 fold 内で OSC/PCA を fit → valid に適用（リーク防止）
- optimizer: grid  → グリッドサーチ（デフォルト）
  optimizer: optuna → グリッドサーチで前処理を絞り込んだ後 Optuna でチューニング
- 探索結果（全組み合わせの CV スコア）を JSON で保存する
- 最良パラメータで全 train を学習したモデルを joblib で保存する
- --show で models/ 以下の全実験結果を一覧表示する
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import (
    load_data, fit_msc, build_pipeline,
    apply_log_y, inverse_log_y,
)


# ============================================================
# 評価指標
# ============================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================
# 1 fold 分の学習・予測
# ============================================================

def _fit_predict_fold(X_tr, y_tr, X_val, y_val,
                      model_cfg: dict, use_log_y: bool) -> tuple[np.ndarray, object]:
    """
    1 fold 分の学習と予測を行い (予測値, 学習済みモデル) を返す。

    予測値は元のスケール・0以上にクリップ済み。
    SVR のみ (StandardScaler, SVR) の tuple として返す。
    """
    model_type = model_cfg["model_type"]
    params     = model_cfg.get("params", {})
    y_tr_fit   = apply_log_y(y_tr) if use_log_y else y_tr

    if model_type == "pls":
        model = PLSRegression(n_components=params.get("n_components", 10),
                              scale=False)
        model.fit(X_tr, y_tr_fit)
        y_pred = model.predict(X_val).flatten()

    elif model_type == "lgbm":
        import lightgbm as lgb
        lgbm_params = {
            "n_estimators":      params.get("n_estimators", 1000),
            "learning_rate":     params.get("learning_rate", 0.05),
            "num_leaves":        params.get("num_leaves", 31),
            "max_depth":         params.get("max_depth", -1),
            "min_child_samples": params.get("min_child_samples", 20),
            "subsample":         params.get("subsample", 0.8),
            "colsample_bytree":  params.get("colsample_bytree", 0.8),
            "colsample_bylevel": params.get("colsample_bylevel", 1.0),
            "reg_lambda":        params.get("reg_lambda", 1.0),
            "reg_alpha":         params.get("reg_alpha", 0.0),
            "random_state":      params.get("random_state", 42),
            "verbose":           -1,
        }
        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_tr, y_tr_fit,
                  eval_set=[(X_val, apply_log_y(y_val) if use_log_y else y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=-1)])
        y_pred = model.predict(X_val)

    elif model_type == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=params.get("alpha", 1.0))
        model.fit(X_tr, y_tr_fit)
        y_pred = model.predict(X_val)

    elif model_type == "svr":
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_tr_sc  = sc.fit_transform(X_tr)
        X_val_sc = sc.transform(X_val)
        inner = SVR(C=params.get("C", 1.0),
                    epsilon=params.get("epsilon", 0.1),
                    kernel=params.get("kernel", "rbf"),
                    gamma=params.get("gamma", "scale"))
        inner.fit(X_tr_sc, y_tr_fit)
        y_pred = inner.predict(X_val_sc)
        model  = (sc, inner)

    else:
        raise ValueError(f"未対応のモデル: {model_type}")

    if use_log_y:
        y_pred = inverse_log_y(y_pred)
    y_pred = np.clip(y_pred, 0, None)
    return y_pred, model


# ============================================================
# GroupKFold CV（汎用）
# ============================================================

def cross_validate(X_raw: np.ndarray,
                   y: np.ndarray,
                   groups: np.ndarray,
                   prep_cfg: dict,
                   model_cfg: dict,
                   ref_spectrum: np.ndarray,
                   n_splits: int = 5,
                   use_log_y: bool = False) -> tuple[float, list[dict]]:
    """
    GroupKFold でクロスバリデーションを行い CV-RMSE と詳細結果を返す。

    OSC / PCA は各 fold の train データのみで fit し、
    同 fold の valid に適用する（データリーク防止）。
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
            gkf.split(X_raw, y, groups=groups)):

        X_tr_raw, X_val_raw = X_raw[tr_idx], X_raw[val_idx]
        y_tr, y_val         = y[tr_idx], y[val_idx]
        groups_val          = groups[val_idx]

        X_tr, saved_params = build_pipeline(
            X_tr_raw, y_tr, prep_cfg, ref_spectrum, fit_mode=True)
        X_val, _ = build_pipeline(
            X_val_raw, y_val, prep_cfg, ref_spectrum,
            fit_mode=False, saved_params=saved_params)

        y_pred, model = _fit_predict_fold(
            X_tr, y_tr, X_val, y_val, model_cfg, use_log_y)

        fold_rmse    = rmse(y_val, y_pred)
        species_rmse = {
            str(int(sp)): rmse(y_val[groups_val == sp], y_pred[groups_val == sp])
            for sp in np.unique(groups_val)
        }
        extra = {}
        if hasattr(model, "best_iteration_"):
            extra["best_iteration"] = model.best_iteration_

        fold_results.append({
            "fold":         fold_idx + 1,
            "rmse":         fold_rmse,
            "n_valid":      int(len(y_val)),
            "species_rmse": species_rmse,
            **extra,
        })

    cv_rmse = float(np.mean([r["rmse"] for r in fold_results]))
    return cv_rmse, fold_results


# ============================================================
# グリッドサーチ
# ============================================================

def _expand_grid(search_params: dict) -> list[dict]:
    """search_params のリストを総当たり展開してパラメータ組み合わせリストを返す。"""
    keys_prep  = list(search_params.get("preprocessing", {}).keys())
    vals_prep  = [search_params["preprocessing"][k] for k in keys_prep]
    keys_model = list(search_params.get("model", {}).keys())
    vals_model = [search_params["model"][k] for k in keys_model]

    all_keys = keys_prep + keys_model
    all_vals = vals_prep + vals_model

    if not all_vals:
        return [{"preprocessing": {}, "model": {}}]

    grid = []
    for combo in itertools.product(*all_vals):
        flat = dict(zip(all_keys, combo))
        grid.append({
            "preprocessing": {k: flat[k] for k in keys_prep},
            "model":         {k: flat[k] for k in keys_model},
        })
    return grid


def grid_search(X_raw: np.ndarray,
                y: np.ndarray,
                groups: np.ndarray,
                base_cfg: dict,
                ref_spectrum: np.ndarray) -> tuple[dict, list[dict]]:
    """
    config の search_params に従ってグリッドサーチを行う。

    Returns
    -------
    best_cfg   : dict         最良の組み合わせ（prep + model パラメータ上書き済み）
    search_log : list[dict]   全組み合わせの結果（CV-RMSE 昇順）
    """
    n_splits  = base_cfg.get("cv", {}).get("n_splits", 5)
    use_log_y = base_cfg.get("cv", {}).get("use_log_y", False)
    base_prep  = dict(base_cfg.get("preprocessing", {}))
    base_model = dict(base_cfg.get("model", {}))

    grid  = _expand_grid(base_cfg.get("search_params", {}))
    total = len(grid)

    print(f"\n{'='*65}")
    print(f"グリッドサーチ: {total} 組み合わせ  "
          f"model={base_model.get('model_type')}  "
          f"folds={n_splits}  log_y={use_log_y}")
    print(f"{'='*65}")

    search_log = []
    best_rmse  = np.inf
    best_cfg   = None

    for i, combo in enumerate(grid, 1):
        prep_cfg  = {**base_prep,  **combo["preprocessing"]}
        model_cfg = {**base_model, "params": {
            **base_model.get("params", {}), **combo["model"]}}

        try:
            cv_rmse, fold_results = cross_validate(
                X_raw, y, groups, prep_cfg, model_cfg,
                ref_spectrum, n_splits, use_log_y)
        except Exception as e:
            print(f"  [{i:>4}/{total}] ERROR: {e}")
            continue

        marker = " ← best" if cv_rmse < best_rmse else ""
        print(f"  [{i:>4}/{total}]  CV-RMSE={cv_rmse:.4f}{marker}  "
              f"prep={combo['preprocessing']}  model={combo['model']}")

        search_log.append({
            "rank": None, "cv_rmse": cv_rmse,
            "preprocessing": prep_cfg, "model": model_cfg,
            "fold_results": fold_results,
        })

        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_cfg  = {"preprocessing": prep_cfg, "model": model_cfg,
                         "cv_rmse": cv_rmse}

    search_log.sort(key=lambda x: x["cv_rmse"])
    for rank, e in enumerate(search_log, 1):
        e["rank"] = rank

    print(f"\n最良: CV-RMSE={best_rmse:.4f}")
    print(f"  前処理: {best_cfg['preprocessing']}")
    print(f"  モデル: {best_cfg['model']}")
    return best_cfg, search_log


# ============================================================
# Optuna チューニング（LightGBM 専用）
# ============================================================

def _optuna_tune(X_raw: np.ndarray,
                 y: np.ndarray,
                 groups: np.ndarray,
                 prep_cfg: dict,
                 ref_spectrum: np.ndarray,
                 n_trials: int = 100,
                 n_splits: int = 5,
                 use_log_y: bool = False) -> tuple[dict, float]:
    """
    指定の前処理設定で LightGBM の Optuna チューニングを実行する。

    fold 内前処理を事前計算してキャッシュし、Optuna の各 trial では
    モデル学習のみ行うことで高速化している。

    Returns
    -------
    best_params : dict   最良の LightGBM パラメータ
    best_rmse   : float  最良の CV-RMSE
    """
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    gkf = GroupKFold(n_splits=n_splits)
    fold_data = []
    for tr_idx, val_idx in gkf.split(X_raw, y, groups=groups):
        X_tr_raw, X_val_raw = X_raw[tr_idx], X_raw[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        X_tr, saved_p = build_pipeline(
            X_tr_raw, y_tr, prep_cfg, ref_spectrum, fit_mode=True)
        X_val, _ = build_pipeline(
            X_val_raw, y_val, prep_cfg, ref_spectrum,
            fit_mode=False, saved_params=saved_p)
        fold_data.append((
            X_tr, X_val,
            apply_log_y(y_tr)  if use_log_y else y_tr,
            y_val,
        ))

    def objective(trial):
        params = {
            "num_leaves":        trial.suggest_int("num_leaves", 5, 100),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "reg_alpha":         trial.suggest_float("reg_alpha",  1e-3, 100.0, log=True),
            "n_estimators": 1000, "learning_rate": 0.05,
            "random_state": 42,   "verbose": -1,
        }
        fold_rmses = []
        for X_tr, X_val, y_tr_fit, y_val in fold_data:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr_fit,
                      eval_set=[(X_val, y_tr_fit)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)])
            y_pred = np.clip(model.predict(X_val), 0, None)
            if use_log_y:
                y_pred = inverse_log_y(y_pred)
            fold_rmses.append(rmse(y_val, y_pred))
        return float(np.mean(fold_rmses))

    study = optuna.create_study(direction="minimize")

    def _progress(study, trial):
        if (trial.number + 1) % 20 == 0:
            print(f"    trial {trial.number+1:>4}: "
                  f"CV-RMSE={trial.value:.4f}  best={study.best_value:.4f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[_progress])

    best_params = {**study.best_params,
                   "n_estimators": 1000, "learning_rate": 0.05,
                   "random_state": 42,   "verbose": -1}

    print(f"  → Optuna 完了: CV-RMSE={study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"     {k}: {v}")

    return best_params, study.best_value


def optuna_search(X_raw: np.ndarray,
                  y: np.ndarray,
                  groups: np.ndarray,
                  base_cfg: dict,
                  ref_spectrum: np.ndarray,
                  n_trials: int = 100,
                  n_top: int = 3) -> tuple[dict, list[dict]]:
    """
    Phase 1（グリッドサーチで前処理を絞り込み）→
    Phase 2（上位 n_top 候補を Optuna で LightGBM チューニング）

    Returns
    -------
    best_cfg   : dict         最良設定（前処理 + 最適化済み LightGBM パラメータ）
    search_log : list[dict]   Phase 2 の全候補の結果
    """
    n_splits  = base_cfg.get("cv", {}).get("n_splits", 5)
    use_log_y = base_cfg.get("cv", {}).get("use_log_y", False)

    # --- Phase 1: 前処理スクリーニング ---
    print(f"\n{'#'*65}")
    print("# Phase 1: グリッドサーチ（前処理スクリーニング）")
    print(f"{'#'*65}")
    _, phase1_log = grid_search(X_raw, y, groups, base_cfg, ref_spectrum)
    top_candidates = phase1_log[:n_top]

    print(f"\nPhase 2 対象（上位 {n_top} 候補）:")
    for i, c in enumerate(top_candidates, 1):
        print(f"  [{i}] CV-RMSE={c['cv_rmse']:.4f}  prep={c['preprocessing']}")

    # --- Phase 2: Optuna チューニング ---
    print(f"\n{'#'*65}")
    print(f"# Phase 2: Optuna チューニング (n_trials={n_trials})")
    print(f"{'#'*65}")

    search_log = []
    best_rmse  = np.inf
    best_cfg   = None

    for i, cand in enumerate(top_candidates, 1):
        prep_cfg = cand["preprocessing"]
        print(f"\n[{i}/{n_top}] 前処理: {prep_cfg}")

        best_params, cv_rmse = _optuna_tune(
            X_raw, y, groups, prep_cfg, ref_spectrum,
            n_trials=n_trials, n_splits=n_splits, use_log_y=use_log_y)

        # early_stopping の平均 best_iteration を取得するため CV を再実行
        model_cfg = {"model_type": "lgbm", "params": best_params}
        _, fold_results = cross_validate(
            X_raw, y, groups, prep_cfg, model_cfg,
            ref_spectrum, n_splits, use_log_y)
        avg_iter = int(np.mean([
            r["best_iteration"] for r in fold_results
            if "best_iteration" in r
        ] or [best_params["n_estimators"]]))

        # 最終モデルは early stopping の平均イテレーションを n_estimators に使う
        final_params = {**best_params, "n_estimators": avg_iter}
        model_cfg_final = {"model_type": "lgbm", "params": final_params}

        search_log.append({
            "rank": i, "cv_rmse": cv_rmse,
            "preprocessing": prep_cfg,
            "model": model_cfg_final,
            "avg_iter": avg_iter,
        })

        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_cfg  = {"preprocessing": prep_cfg, "model": model_cfg_final,
                         "cv_rmse": cv_rmse}

    search_log.sort(key=lambda x: x["cv_rmse"])
    for rank, e in enumerate(search_log, 1):
        e["rank"] = rank

    print(f"\nPhase 2 最良: CV-RMSE={best_rmse:.4f}")
    return best_cfg, search_log


# ============================================================
# 最終モデルの学習・保存
# ============================================================

def train_final_model(X_raw: np.ndarray,
                      y: np.ndarray,
                      prep_cfg: dict,
                      model_cfg: dict,
                      ref_spectrum: np.ndarray,
                      use_log_y: bool = False,
                      save_dir: str = "models") -> dict:
    """
    最良設定で全 train を使ってモデルを学習し保存する。

    保存ファイル:
        model.pkl        学習済みモデル
        prep_params.pkl  前処理パラメータ（OSC/PCA）
        ref_spectrum.pkl MSC 用基準スペクトル
        config.json      設定（再現用）
    """
    os.makedirs(save_dir, exist_ok=True)

    X, saved_params = build_pipeline(
        X_raw, y, prep_cfg, ref_spectrum, fit_mode=True)
    y_fit = apply_log_y(y) if use_log_y else y

    model_type = model_cfg["model_type"]
    params     = model_cfg.get("params", {})

    if model_type == "pls":
        model = PLSRegression(n_components=params.get("n_components", 10),
                              scale=False)
        model.fit(X, y_fit)

    elif model_type == "lgbm":
        import lightgbm as lgb
        p = {**params, "verbose": -1}
        p.setdefault("random_state", 42)
        model = lgb.LGBMRegressor(**p)
        model.fit(X, y_fit)

    elif model_type == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=params.get("alpha", 1.0))
        model.fit(X, y_fit)

    elif model_type == "svr":
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        inner = SVR(C=params.get("C", 1.0),
                    epsilon=params.get("epsilon", 0.1),
                    kernel=params.get("kernel", "rbf"),
                    gamma=params.get("gamma", "scale"))
        inner.fit(X_sc, y_fit)
        model = (sc, inner)

    else:
        raise ValueError(f"未対応のモデル: {model_type}")

    paths = {
        "model":        os.path.join(save_dir, "model.pkl"),
        "prep_params":  os.path.join(save_dir, "prep_params.pkl"),
        "ref_spectrum": os.path.join(save_dir, "ref_spectrum.pkl"),
        "config":       os.path.join(save_dir, "config.json"),
    }
    joblib.dump(model,        paths["model"])
    joblib.dump(saved_params, paths["prep_params"])
    joblib.dump(ref_spectrum, paths["ref_spectrum"])
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump({"preprocessing": prep_cfg, "model": model_cfg,
                   "use_log_y": use_log_y}, f, ensure_ascii=False, indent=2)

    print(f"\n[train_final_model] 保存完了: {save_dir}")
    for k, v in paths.items():
        print(f"  {k:<14}: {v}")
    return paths


# ============================================================
# CV 結果の詳細表示
# ============================================================

def print_cv_detail(fold_results: list[dict], train_df: pd.DataFrame):
    """樹種ごとの CV-RMSE を表示する。"""
    sp_name_map = dict(zip(
        train_df["species_number"].astype(str), train_df["species_name"]))

    print(f"\n{'='*55}")
    print("樹種ごとの CV-RMSE")
    print(f"{'='*55}")
    print(f"{'樹種':>6}  {'樹種名':<12}  {'RMSE':>8}  {'fold数':>6}")
    print(f"{'-'*40}")

    sp_rmse_all: dict[str, list] = {}
    for fold in fold_results:
        for sp, r in fold["species_rmse"].items():
            sp_rmse_all.setdefault(str(sp), []).append(r)

    for sp in sorted(sp_rmse_all.keys(), key=lambda x: int(x)):
        mean_r = float(np.mean(sp_rmse_all[sp]))
        print(f"{sp:>6}  {sp_name_map.get(sp, '不明'):<12}  "
              f"{mean_r:>8.4f}  {len(sp_rmse_all[sp]):>6}")


# ============================================================
# 実験結果一覧表示
# ============================================================

def show_results(model_dir: str = "models", top: int = 20):
    """models/ 以下の全実験の CV-RMSE を昇順で表示する。"""
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"[ERROR] {model_dir} が存在しません")
        return

    results = []
    for run_dir in sorted(model_path.iterdir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        prep = s.get("best_cfg", {}).get("preprocessing", {})
        results.append({
            "run_id":  s.get("run_id", run_dir.name),
            "cv_rmse": s.get("cv_rmse", float("inf")),
            "model":   s.get("best_cfg", {}).get("model", {}).get("model_type", "?"),
            "scaler":  prep.get("scaler", "-"),
            "deriv":   prep.get("deriv", "-"),
            "window":  prep.get("window", "-"),
            "osc":     "yes" if prep.get("use_osc") else "no",
            "pca":     "yes" if prep.get("use_pca") else "no",
            "log_y":   "yes" if s.get("use_log_y") else "no",
        })

    if not results:
        print("実験結果が見つかりませんでした。")
        return

    results.sort(key=lambda x: x["cv_rmse"])
    print(f"\n{'='*85}")
    print(f"実験結果サマリー（CV-RMSE 昇順、上位 {top} 件）")
    print(f"{'='*85}")
    print(f"{'run_id':<35}  {'model':<6}  {'scaler':<5}  "
          f"{'deriv':>5}  {'win':>4}  {'osc':>4}  {'pca':>4}  "
          f"{'logy':>4}  {'CV-RMSE':>10}")
    print(f"{'-'*85}")
    for r in results[:top]:
        print(f"{r['run_id']:<35}  {r['model']:<6}  {str(r['scaler']):<5}  "
              f"{str(r['deriv']):>5}  {str(r['window']):>4}  "
              f"{r['osc']:>4}  {r['pca']:>4}  {r['log_y']:>4}  "
              f"{r['cv_rmse']:>10.4f}")
    print(f"\n現時点の最良: {results[0]['run_id']}  "
          f"CV-RMSE={results[0]['cv_rmse']:.4f}")


# ============================================================
# エントリーポイント
# ============================================================

def run_from_config(config_path: str,
                    optimizer: str = "grid",
                    n_trials: int = 100,
                    n_top: int = 3):
    """
    config YAML を読み込んで実験を実行し、最終モデルを保存する。

    Parameters
    ----------
    optimizer : "grid" | "optuna"
        grid  → グリッドサーチのみ
        optuna → グリッドサーチ（前処理絞り込み）→ Optuna（LightGBM 専用）
    n_trials  : Optuna の試行回数（optimizer="optuna" のとき有効）
    n_top     : Optuna をかける上位候補数（同上）
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id   = cfg.get("run_id", Path(config_path).stem)
    save_dir = os.path.join(cfg.get("model_dir", "models"), run_id)
    os.makedirs(save_dir, exist_ok=True)

    train_df, _, _, spec_cols = load_data(cfg.get("data_dir", "data/"))
    X_raw    = train_df[spec_cols].values
    y        = train_df["moisture_content"].values
    groups   = train_df["species_number"].values
    ref_spec = fit_msc(X_raw)

    use_log_y = cfg.get("cv", {}).get("use_log_y", False)
    n_splits  = cfg.get("cv", {}).get("n_splits", 5)

    # --- 探索 ---
    if optimizer == "optuna":
        best_cfg, search_log = optuna_search(
            X_raw, y, groups, cfg, ref_spec,
            n_trials=n_trials, n_top=n_top)
    else:
        best_cfg, search_log = grid_search(
            X_raw, y, groups, cfg, ref_spec)

    # 探索ログ保存（fold_results は除外してサイズを抑える）
    log_slim = [{k: v for k, v in e.items() if k != "fold_results"}
                for e in search_log]
    search_log_path = os.path.join(save_dir, "search_log.json")
    with open(search_log_path, "w", encoding="utf-8") as f:
        json.dump(log_slim, f, ensure_ascii=False, indent=2)
    print(f"\n探索ログ保存: {search_log_path}")

    # --- 最良設定で詳細 CV ---
    print(f"\n{'='*55}")
    print("最良設定の詳細 CV 結果")
    print(f"{'='*55}")
    cv_rmse, fold_results = cross_validate(
        X_raw, y, groups,
        best_cfg["preprocessing"], best_cfg["model"],
        ref_spec, n_splits, use_log_y)
    print_cv_detail(fold_results, train_df)

    # --- 最終モデル学習・保存 ---
    print(f"\n{'='*55}")
    print("全 train データで最終モデルを学習")
    print(f"{'='*55}")
    artifacts = train_final_model(
        X_raw, y,
        best_cfg["preprocessing"], best_cfg["model"],
        ref_spec, use_log_y, save_dir)

    # サマリー保存
    summary = {
        "run_id": run_id, "config_path": str(config_path),
        "cv_rmse": cv_rmse, "best_cfg": best_cfg,
        "use_log_y": use_log_y, "artifacts": artifacts,
    }
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nサマリー保存: {summary_path}")
    print(f"\n{'='*55}")
    print(f"run_id={run_id}  最終 CV-RMSE={cv_rmse:.4f}")
    print(f"{'='*55}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="近赤外スペクトル分析チャレンジ — 学習スクリプト")
    parser.add_argument(
        "--config", type=str, default=None,
        help="実験設定 YAML ファイルのパス")
    parser.add_argument(
        "--optimizer", type=str, default="grid",
        choices=["grid", "optuna"],
        help="探索方法: grid（デフォルト）| optuna（LightGBM 専用）")
    parser.add_argument(
        "--n_trials", type=int, default=100,
        help="Optuna の試行回数（--optimizer optuna のとき有効、デフォルト 100）")
    parser.add_argument(
        "--n_top", type=int, default=3,
        help="Optuna をかける上位候補数（同上、デフォルト 3）")
    parser.add_argument(
        "--show", action="store_true",
        help="models/ 以下の実験結果を一覧表示して終了")
    parser.add_argument(
        "--model_dir", type=str, default="models",
        help="--show のときに参照するディレクトリ（デフォルト: models）")
    args = parser.parse_args()

    if args.show:
        show_results(args.model_dir)
    elif args.config:
        run_from_config(args.config, args.optimizer, args.n_trials, args.n_top)
    else:
        parser.print_help()
