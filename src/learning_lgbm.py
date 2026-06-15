"""
learning_lgbm.py
================
近赤外スペクトル分析チャレンジ — LightGBM学習モジュール

担当する処理:
  1. GroupKFoldによるCV（LightGBM版）
  2. 前処理候補の全探索
  3. 全trainデータでの最終モデル学習
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import lightgbm as lgb
import optuna
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # optunaの冗長なログを抑制
sys.path.append("src")

from preprocessing import (load_data,
                            apply_snv, apply_savgol,
                            fit_msc, apply_msc,
                            fit_osc, apply_osc)


# ============================================================
# 評価指標
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ============================================================
# 前処理の適用
# ============================================================

def apply_preprocessing(X_raw, y, preprocessing, ref_spectrum,
                         fit_mode=True, osc_params=None, pca_model=None):
    """
    前処理の設定に従ってスペクトルを変換する。

    Parameters
    ----------
    X_raw        : np.ndarray  生スペクトル
    y            : np.ndarray  含水率（OSCのfitに使用、fit_mode=Falseのとき不要）
    preprocessing: dict        前処理の設定
    ref_spectrum : np.ndarray  MSC用の基準スペクトル
    fit_mode     : bool        TrueのときOSC・PCAをfitする（train用）
                               FalseのときはパラメータをそのままApply（test用）
    osc_params   : dict        fit_mode=FalseのときのOSCパラメータ
    pca_model    : PCA         fit_mode=FalseのときのPCAモデル

    Returns
    -------
    X           : np.ndarray  前処理済みスペクトル
    osc_params  : dict or None
    pca_model   : PCA or None
    """
    X = X_raw.copy()

    # --- スケーリング ---
    scaler = preprocessing.get("scaler", None)
    if scaler == "snv":
        X = apply_snv(X)
    elif scaler == "msc":
        X = apply_msc(X, ref_spectrum)

    # --- 微分 ---
    deriv = preprocessing.get("deriv", 0)
    if deriv > 0:
        window = preprocessing.get("window", 11)
        X = apply_savgol(X, window_length=window, deriv=deriv)

    # --- OSC ---
    if preprocessing.get("use_osc", False):
        n_osc = preprocessing.get("n_components_osc", 2)
        if fit_mode:
            osc_params = fit_osc(X, y, n_components=n_osc)
        X = apply_osc(X, osc_params)

    # --- PCA ---
    if preprocessing.get("use_pca", False):
        n_pca = preprocessing.get("n_components_pca", 50)
        if fit_mode:
            pca_model = PCA(n_components=n_pca, random_state=42)
            pca_model.fit(X)
        X = pca_model.transform(X)

    return X, osc_params, pca_model


# ============================================================
# GroupKFold CV（LightGBM）
# ============================================================

def cross_validate_lgbm(X_raw, y, groups, preprocessing,
                         ref_spectrum, lgbm_params,
                         n_splits=5):
    """
    GroupKFoldでLightGBMのCVスコアを計算する。

    OSC・PCAはfold内のtrainでfitしてvalid/trainに適用する。

    Parameters
    ----------
    X_raw        : np.ndarray  生スペクトル
    y            : np.ndarray  含水率
    groups       : np.ndarray  樹種番号
    preprocessing: dict        前処理の設定
    ref_spectrum : np.ndarray  MSC用基準スペクトル
    lgbm_params  : dict        LightGBMのハイパーパラメータ
    n_splits     : int         fold数

    Returns
    -------
    cv_rmse      : float
    fold_results : list[dict]
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold_idx, (tr_idx, val_idx) in enumerate(
            gkf.split(X_raw, y, groups=groups)):

        X_tr_raw, X_val_raw = X_raw[tr_idx], X_raw[val_idx]
        y_tr, y_val         = y[tr_idx], y[val_idx]
        groups_val          = groups[val_idx]

        # fold内trainで前処理をfit
        X_tr, osc_p, pca_m = apply_preprocessing(
            X_tr_raw, y_tr, preprocessing, ref_spectrum,
            fit_mode=True
        )
        # fold内validに同じパラメータで適用
        X_val, _, _ = apply_preprocessing(
            X_val_raw, y_val, preprocessing, ref_spectrum,
            fit_mode=False, osc_params=osc_p, pca_model=pca_m
        )

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=-1)])

        y_pred     = np.clip(model.predict(X_val), 0, None)
        fold_rmse  = rmse(y_val, y_pred)

        species_rmse = {}
        for sp in np.unique(groups_val):
            mask = groups_val == sp
            species_rmse[int(sp)] = rmse(y_val[mask], y_pred[mask])

        fold_results.append({
            "fold":         fold_idx + 1,
            "rmse":         fold_rmse,
            "n_valid":      len(y_val),
            "best_iter":    model.best_iteration_,
            "species_rmse": species_rmse,
        })

    cv_rmse = np.mean([r["rmse"] for r in fold_results])
    return cv_rmse, fold_results


# ============================================================
# 前処理候補の全探索
# ============================================================

def search_preprocessing(X_raw, y, groups, ref_spectrum,
                          lgbm_params, n_splits=5):
    """
    前処理候補すべてのCVスコアを計算して比較する。

    Returns
    -------
    results : list[dict]  各候補のCVスコアと設定
    """
    # ----------------------------------------------------------------
    # 前処理候補の定義
    # スケーリング(3) × 微分(3) × OSC(2) = 18候補（PCAなし）
    # + 上記18候補 × PCA成分数(10/20/50/100/200) = 90候補
    # 合計108候補
    #
    # 微分のwindow_lengthはPLSの探索結果を初期値として使用:
    #   1次微分: window=9  （run_007・008より）
    #   2次微分: window=13 （run_003・005より）
    # ----------------------------------------------------------------
    base_candidates = [
        # --- スケーリングのみ ---
        {"name": "なし",
         "scaler": None,  "deriv": 0, "use_osc": False},
        {"name": "SNV",
         "scaler": "snv", "deriv": 0, "use_osc": False},
        {"name": "MSC",
         "scaler": "msc", "deriv": 0, "use_osc": False},
        # --- スケーリング + 1次微分 ---
        {"name": "なし + 1次微分",
         "scaler": None,  "deriv": 1, "window": 9, "use_osc": False},
        {"name": "SNV + 1次微分",
         "scaler": "snv", "deriv": 1, "window": 9, "use_osc": False},
        {"name": "MSC + 1次微分",
         "scaler": "msc", "deriv": 1, "window": 9, "use_osc": False},
        # --- スケーリング + 2次微分 ---
        {"name": "なし + 2次微分",
         "scaler": None,  "deriv": 2, "window": 13, "use_osc": False},
        {"name": "SNV + 2次微分",
         "scaler": "snv", "deriv": 2, "window": 13, "use_osc": False},
        {"name": "MSC + 2次微分",
         "scaler": "msc", "deriv": 2, "window": 13, "use_osc": False},
        # --- スケーリング + OSC ---
        {"name": "なし + OSC",
         "scaler": None,  "deriv": 0, "use_osc": True, "n_components_osc": 2},
        {"name": "SNV + OSC",
         "scaler": "snv", "deriv": 0, "use_osc": True, "n_components_osc": 2},
        {"name": "MSC + OSC",
         "scaler": "msc", "deriv": 0, "use_osc": True, "n_components_osc": 2},
        # --- スケーリング + 1次微分 + OSC ---
        {"name": "なし + 1次微分 + OSC",
         "scaler": None,  "deriv": 1, "window": 9,  "use_osc": True, "n_components_osc": 2},
        {"name": "SNV + 1次微分 + OSC",
         "scaler": "snv", "deriv": 1, "window": 9,  "use_osc": True, "n_components_osc": 2},
        {"name": "MSC + 1次微分 + OSC",
         "scaler": "msc", "deriv": 1, "window": 9,  "use_osc": True, "n_components_osc": 2},
        # --- スケーリング + 2次微分 + OSC ---
        {"name": "なし + 2次微分 + OSC",
         "scaler": None,  "deriv": 2, "window": 13, "use_osc": True, "n_components_osc": 2},
        {"name": "SNV + 2次微分 + OSC",
         "scaler": "snv", "deriv": 2, "window": 13, "use_osc": True, "n_components_osc": 2},
        {"name": "MSC + 2次微分 + OSC",
         "scaler": "msc", "deriv": 2, "window": 13, "use_osc": True, "n_components_osc": 2},
    ]

    # 上記18候補それぞれにPCAを追加（5パターン × 18 = 90候補）
    pca_n_list = [10, 20, 50, 100, 200]
    pca_candidates = []
    for n_pca in pca_n_list:
        for c in base_candidates:
            pc = dict(c)
            pc["name"]             = c["name"] + f" + PCA({n_pca})"
            pc["use_pca"]          = True
            pc["n_components_pca"] = n_pca
            pca_candidates.append(pc)

    # PCAなしの候補にuse_pca=Falseを追加
    for c in base_candidates:
        c["use_pca"] = False

    candidates = base_candidates + pca_candidates  # 合計18 + 90 = 108候補

    results = []
    print(f"\n{'='*75}")
    print(f"LightGBM 前処理候補の全探索 (GroupKFold, n_splits={n_splits})")
    print(f"候補数: {len(candidates)}")
    print(f"{'='*75}")
    print(f"{'前処理':<40}  {'CV-RMSE':>10}  {'best_iter':>10}")
    print(f"{'-'*65}")

    best_rmse = np.inf

    for prep in candidates:
        name = prep["name"]
        try:
            cv_rmse, fold_results = cross_validate_lgbm(
                X_raw, y, groups, prep,
                ref_spectrum, lgbm_params, n_splits=n_splits
            )
            avg_iter = int(np.mean([r["best_iter"] for r in fold_results
                                    if r["best_iter"]]))
            marker = " ← 現時点の最良" if cv_rmse < best_rmse else ""
            print(f"{name:<40}  {cv_rmse:>10.4f}  {avg_iter:>10}{marker}")

            if cv_rmse < best_rmse:
                best_rmse = cv_rmse

            results.append({
                "name":          name,
                "preprocessing": prep,
                "cv_rmse":       cv_rmse,
                "avg_iter":      avg_iter,
                "fold_results":  fold_results,
            })
        except Exception as e:
            print(f"{name:<40}  ERROR: {e}")

    best = min(results, key=lambda x: x["cv_rmse"])
    print(f"\n最良の前処理: {best['name']}  "
          f"(CV-RMSE={best['cv_rmse']:.4f}, best_iter={best['avg_iter']})")
    return results


# ============================================================
# Optunaによるハイパーパラメータ最適化
# ============================================================

def optimize_lgbm_params(X_raw, y, groups, preprocessing,
                          ref_spectrum, n_trials=100, n_splits=5):
    """
    OptunaでLightGBMのハイパーパラメータを最適化する。

    前処理はMSC+2次微分で固定し、A・B・Cの各パラメータを探索する。

    グリッドサーチとの違い:
        グリッドサーチ: 全組み合わせを順番に試す（膨大な時間）
        Optuna:        過去の試行結果から「良さそな領域」を推定して次の候補を選ぶ
                       → 100〜200試行で十分な精度の最適解が得られる

    Parameters
    ----------
    X_raw        : np.ndarray  生スペクトル
    y            : np.ndarray  含水率
    groups       : np.ndarray  樹種番号
    preprocessing: dict        前処理の設定（固定）
    ref_spectrum : np.ndarray  MSC用基準スペクトル
    n_trials     : int         最適化の試行回数（デフォルト100）
    n_splits     : int         GroupKFoldのfold数

    Returns
    -------
    best_params  : dict  最良のハイパーパラメータ
    study        : optuna.Study  最適化の結果（全試行の記録）
    """
    # 前処理を事前に適用しておく（毎trial同じ前処理を繰り返さないため）
    gkf = GroupKFold(n_splits=n_splits)
    fold_data = []
    for tr_idx, val_idx in gkf.split(X_raw, y, groups=groups):
        X_tr_raw, X_val_raw = X_raw[tr_idx], X_raw[val_idx]
        y_tr, y_val         = y[tr_idx], y[val_idx]
        groups_val          = groups[val_idx]

        X_tr, osc_p, pca_m = apply_preprocessing(
            X_tr_raw, y_tr, preprocessing, ref_spectrum, fit_mode=True
        )
        X_val, _, _ = apply_preprocessing(
            X_val_raw, y_val, preprocessing, ref_spectrum,
            fit_mode=False, osc_params=osc_p, pca_model=pca_m
        )
        fold_data.append((X_tr, X_val, y_tr, y_val, groups_val))

    def objective(trial):
        params = {
            # A. モデルの複雑さ
            "num_leaves":        trial.suggest_int("num_leaves", 5, 100),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            # B. サンプリング
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            # C. 正則化
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "reg_alpha":         trial.suggest_float("reg_alpha",  1e-3, 100.0, log=True),
            # 固定値
            "n_estimators":      1000,
            "learning_rate":     0.05,
            "random_state":      42,
            "verbose":           -1,
        }

        fold_rmses = []
        for X_tr, X_val, y_tr, y_val, _ in fold_data:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)])
            y_pred = np.clip(model.predict(X_val), 0, None)
            fold_rmses.append(rmse(y_val, y_pred))

        return np.mean(fold_rmses)

    print(f"\n{'='*60}")
    print(f"Optunaによるハイパーパラメータ最適化")
    print(f"前処理: {preprocessing['name']}")
    print(f"試行回数: {n_trials}")
    print(f"{'='*60}")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[_optuna_progress_callback])

    best_params = study.best_params
    best_params["n_estimators"]  = 1000
    best_params["learning_rate"] = 0.05
    best_params["random_state"]  = 42
    best_params["verbose"]       = -1

    print(f"\n{'='*60}")
    print(f"最適化完了")
    print(f"最良CV-RMSE: {study.best_value:.4f}")
    print(f"最良パラメータ:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    return best_params, study


def _optuna_progress_callback(study, trial):
    """10試行ごとに進捗を表示する"""
    if (trial.number + 1) % 10 == 0:
        print(f"  trial {trial.number+1:>4}: "
              f"CV-RMSE={trial.value:.4f}  "
              f"best={study.best_value:.4f}")


# ============================================================
# 最終モデルの学習（全trainデータ）
# ============================================================

def train_final_lgbm(X_raw, y, preprocessing, ref_spectrum,
                      lgbm_params, save_dir="models"):
    """
    最良の前処理とパラメータで全trainデータを使ってLightGBMを学習し保存する。

    Parameters
    ----------
    X_raw        : np.ndarray  生スペクトル
    y            : np.ndarray  含水率
    preprocessing: dict        前処理の設定
    ref_spectrum : np.ndarray  MSC用基準スペクトル
    lgbm_params  : dict        LightGBMのハイパーパラメータ
    save_dir     : str         保存先ディレクトリ
    """
    os.makedirs(save_dir, exist_ok=True)

    # 全trainで前処理をfit
    X, osc_params, pca_model = apply_preprocessing(
        X_raw, y, preprocessing, ref_spectrum, fit_mode=True
    )

    # LightGBMを学習（early_stoppingなし、n_estimatorsを固定）
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X, y)

    # 保存
    joblib.dump(model,        f"{save_dir}/lgbm_model.pkl")
    joblib.dump(preprocessing, f"{save_dir}/lgbm_preprocessing.pkl")
    joblib.dump(ref_spectrum,  f"{save_dir}/ref_spectrum.pkl")
    if osc_params is not None:
        joblib.dump(osc_params, f"{save_dir}/osc_params.pkl")
    if pca_model is not None:
        joblib.dump(pca_model,  f"{save_dir}/pca_model.pkl")

    print(f"\n[train_final_lgbm] 学習完了")
    print(f"[train_final_lgbm] 前処理: {preprocessing['name']}")
    print(f"[train_final_lgbm] モデルを保存: {save_dir}/lgbm_model.pkl")
    return model, osc_params, pca_model


# ============================================================
# CV結果の詳細表示
# ============================================================

def print_cv_detail(fold_results, train):
    sp_name_map = dict(zip(train["species_number"], train["species_name"]))

    print(f"\n{'='*55}")
    print("樹種ごとのCV-RMSE")
    print(f"{'='*55}")

    sp_rmse_all = {}
    for fold in fold_results:
        for sp, r in fold["species_rmse"].items():
            sp_rmse_all.setdefault(sp, []).append(r)

    print(f"{'樹種番号':>6}  {'樹種名':<12}  {'RMSE':>8}")
    print(f"{'-'*35}")
    for sp in sorted(sp_rmse_all.keys()):
        mean_rmse = np.mean(sp_rmse_all[sp])
        name = sp_name_map.get(sp, "不明")
        print(f"{sp:>6}  {name:<12}  {mean_rmse:>8.4f}")


# ============================================================
# Phase 1の探索候補定義 / Phase 2のパラメータ最適化
# ============================================================

# ============================================================
# Phase 1の探索候補定義
# ============================================================

# グループA: 最良がPCA(10)で左端 → より小さい成分数を探索
GROUP_A = [
    {"name": "なし",              "scaler": None,  "deriv": 0, "use_osc": False, "use_pca": True},
    {"name": "SNV",               "scaler": "snv", "deriv": 0, "use_osc": False, "use_pca": True},
    {"name": "SNV + OSC",         "scaler": "snv", "deriv": 0, "use_osc": True,  "use_pca": True, "n_components_osc": 2},
    {"name": "なし + OSC",         "scaler": None,  "deriv": 0, "use_osc": True,  "use_pca": True, "n_components_osc": 2},
    {"name": "MSC + OSC",         "scaler": "msc", "deriv": 0, "use_osc": True,  "use_pca": True, "n_components_osc": 2},
    {"name": "なし + 1次微分 + OSC","scaler": None,  "deriv": 1, "window": 9, "use_osc": True,  "use_pca": True, "n_components_osc": 2},
]
GROUP_A_PCA = [2, 3, 4, 5, 6, 7, 8, 9]

# グループB: 最良がPCA(200)で右端 → より大きい成分数を探索
GROUP_B = [
    {"name": "SNV + 2次微分", "scaler": "snv", "deriv": 2, "window": 13, "use_osc": False, "use_pca": True},
]
GROUP_B_PCA = [250, 300, 400, 500]

# グループC: 最良が中間だが前後が粗い → 最良付近を細かく探索
GROUP_C = [
    {"name": "MSC",               "scaler": "msc", "deriv": 0, "use_osc": False, "use_pca": True,
     "pca_candidates": [15,16,17,18,19,21,22,23,24,25]},
    {"name": "SNV + 1次微分",      "scaler": "snv", "deriv": 1, "window": 9,  "use_osc": False, "use_pca": True,
     "pca_candidates": [15,16,17,18,19,21,22,23,24,25]},
    {"name": "MSC + 2次微分",      "scaler": "msc", "deriv": 2, "window": 13, "use_osc": False, "use_pca": True,
     "pca_candidates": [30,35,40,45,55,60,70,80]},
    {"name": "なし + 1次微分",      "scaler": None,  "deriv": 1, "window": 9,  "use_osc": False, "use_pca": True,
     "pca_candidates": [25,30,35,40,45,55,60]},
    {"name": "MSC + 1次微分 + OSC","scaler": "msc", "deriv": 1, "window": 9,  "use_osc": True,  "use_pca": True, "n_components_osc": 2,
     "pca_candidates": [15,16,17,18,19,21,22,23,24,25]},
    {"name": "SNV + 2次微分 + OSC","scaler": "snv", "deriv": 2, "window": 13, "use_osc": True,  "use_pca": True, "n_components_osc": 2,
     "pca_candidates": [60,70,80,90,110,120,150]},
    {"name": "MSC + 2次微分 + OSC","scaler": "msc", "deriv": 2, "window": 13, "use_osc": True,  "use_pca": True, "n_components_osc": 2,
     "pca_candidates": [30,35,40,45,55,60,70]},
    {"name": "MSC + 1次微分",      "scaler": "msc", "deriv": 1, "window": 9,  "use_osc": False, "use_pca": True,
     "pca_candidates": [15,16,17,18,19,21,22,23,24,25]},
]

# 前回の結果（Phase 1の比較基準として使用）
PREVIOUS_RESULTS = {
    ("なし",               None): 31.3439,
    ("SNV",                None): 25.8152,
    ("MSC",                None): 25.0105,
    ("なし + 1次微分",      None): 21.9725,
    ("SNV + 1次微分",       None): 18.6635,
    ("MSC + 1次微分",       None): 18.5105,
    ("なし + 2次微分",      None): 19.4135,
    ("SNV + 2次微分",       None): 18.1753,
    ("MSC + 2次微分",       None): 17.9373,
    ("なし + OSC",          None): 35.7396,
    ("SNV + OSC",           None): 41.5599,
    ("MSC + OSC",           None): 34.5248,
    ("なし + 1次微分 + OSC", None): 25.5404,
    ("SNV + 1次微分 + OSC", None): 20.5321,
    ("MSC + 1次微分 + OSC", None): 18.8564,
    ("なし + 2次微分 + OSC", None): 19.1460,
    ("SNV + 2次微分 + OSC", None): 19.2800,
    ("MSC + 2次微分 + OSC", None): 18.7871,
    ("なし",               10): 28.6689, ("なし",               20): 31.4341, ("なし",               50): 34.5757, ("なし",               100): 35.4067, ("なし",               200): 35.2960,
    ("SNV",                10): 25.2063, ("SNV",                20): 25.9827, ("SNV",                50): 27.2527, ("SNV",                100): 27.7529, ("SNV",                200): 27.2699,
    ("MSC",                10): 25.8342, ("MSC",                20): 23.7455, ("MSC",                50): 26.1824, ("MSC",                100): 27.3133, ("MSC",                200): 27.4570,
    ("なし + 1次微分",      10): 28.4537, ("なし + 1次微分",      20): 26.5872, ("なし + 1次微分",      50): 26.2527, ("なし + 1次微分",      100): 26.5402, ("なし + 1次微分",      200): 27.0584,
    ("SNV + 1次微分",       10): 30.5790, ("SNV + 1次微分",       20): 24.0259, ("SNV + 1次微分",       50): 26.9283, ("SNV + 1次微分",       100): 27.2590, ("SNV + 1次微分",       200): 26.9579,
    ("MSC + 1次微分",       10): 29.3230, ("MSC + 1次微分",       20): 26.3949, ("MSC + 1次微分",       50): 28.2002, ("MSC + 1次微分",       100): 27.6503, ("MSC + 1次微分",       200): 28.0274,
    ("なし + 2次微分",      10): 30.6285, ("なし + 2次微分",      20): 29.1809, ("なし + 2次微分",      50): 30.2237, ("なし + 2次微分",      100): 30.5432, ("なし + 2次微分",      200): 31.3171,
    ("SNV + 2次微分",       10): 31.3460, ("SNV + 2次微分",       20): 25.3088, ("SNV + 2次微分",       50): 24.9036, ("SNV + 2次微分",       100): 24.7750, ("SNV + 2次微分",       200): 24.6887,
    ("MSC + 2次微分",       10): 35.1894, ("MSC + 2次微分",       20): 27.0624, ("MSC + 2次微分",       50): 24.7874, ("MSC + 2次微分",       100): 25.2237, ("MSC + 2次微分",       200): 25.0388,
    ("なし + OSC",          10): 31.9320, ("なし + OSC",          20): 32.6686, ("なし + OSC",          50): 33.6251, ("なし + OSC",          100): 35.2325, ("なし + OSC",          200): 34.5833,
    ("SNV + OSC",           10): 26.3049, ("SNV + OSC",           20): 26.8009, ("SNV + OSC",           50): 27.1782, ("SNV + OSC",           100): 27.9436, ("SNV + OSC",           200): 27.2511,
    ("MSC + OSC",           10): 30.6303, ("MSC + OSC",           20): 31.7041, ("MSC + OSC",           50): 33.9093, ("MSC + OSC",           100): 33.7177, ("MSC + OSC",           200): 34.5462,
    ("なし + 1次微分 + OSC", 10): 25.1092, ("なし + 1次微分 + OSC", 20): 25.9101, ("なし + 1次微分 + OSC", 50): 25.7384, ("なし + 1次微分 + OSC", 100): 26.5578, ("なし + 1次微分 + OSC", 200): 26.4932,
    ("SNV + 1次微分 + OSC", 10): 30.4043, ("SNV + 1次微分 + OSC", 20): 28.0247, ("SNV + 1次微分 + OSC", 50): 31.7463, ("SNV + 1次微分 + OSC", 100): 30.9835, ("SNV + 1次微分 + OSC", 200): 32.0975,
    ("MSC + 1次微分 + OSC", 10): 28.9367, ("MSC + 1次微分 + OSC", 20): 25.0716, ("MSC + 1次微分 + OSC", 50): 26.4255, ("MSC + 1次微分 + OSC", 100): 26.3976, ("MSC + 1次微分 + OSC", 200): 27.6607,
    ("なし + 2次微分 + OSC", 10): 30.1435, ("なし + 2次微分 + OSC", 20): 29.2115, ("なし + 2次微分 + OSC", 50): 30.5647, ("なし + 2次微分 + OSC", 100): 30.6871, ("なし + 2次微分 + OSC", 200): 31.0362,
    ("SNV + 2次微分 + OSC", 10): 30.3001, ("SNV + 2次微分 + OSC", 20): 24.2972, ("SNV + 2次微分 + OSC", 50): 24.5303, ("SNV + 2次微分 + OSC", 100): 24.0236, ("SNV + 2次微分 + OSC", 200): 24.1919,
    ("MSC + 2次微分 + OSC", 10): 33.8616, ("MSC + 2次微分 + OSC", 20): 26.6457, ("MSC + 2次微分 + OSC", 50): 25.2596, ("MSC + 2次微分 + OSC", 100): 25.2955, ("MSC + 2次微分 + OSC", 200): 25.8749,
}


# ============================================================
# Phase 1: PCA成分数の追加探索
# ============================================================

def run_phase1(X_raw, y, groups, ref_spectrum, lgbm_params, n_splits=5):
    """
    前回の結果をもとにPCA成分数の未探索領域を補完する。

    Returns
    -------
    all_results : dict  {(前処理名, PCA成分数): CV-RMSE}
    """
    # 前回の結果をベースに追加していく
    all_results = dict(PREVIOUS_RESULTS)

    print(f"\n{'='*65}")
    print("Phase 1: PCA成分数の追加探索")
    print(f"{'='*65}")

    # グループA: PCA(2〜9)
    print(f"\n--- グループA: 最良がPCA(10)→ より小さい成分数を探索 ---")
    for prep in GROUP_A:
        name = prep["name"]
        for n_pca in GROUP_A_PCA:
            p = dict(prep)
            p["n_components_pca"] = n_pca
            cv_rmse, _ = cross_validate_lgbm(
                X_raw, y, groups, p, ref_spectrum, lgbm_params, n_splits
            )
            all_results[(name, n_pca)] = cv_rmse
            print(f"  {name} + PCA({n_pca}): {cv_rmse:.4f}")

    # グループB: PCA(250〜500)
    print(f"\n--- グループB: 最良がPCA(200)→ より大きい成分数を探索 ---")
    for prep in GROUP_B:
        name = prep["name"]
        for n_pca in GROUP_B_PCA:
            p = dict(prep)
            p["n_components_pca"] = n_pca
            cv_rmse, _ = cross_validate_lgbm(
                X_raw, y, groups, p, ref_spectrum, lgbm_params, n_splits
            )
            all_results[(name, n_pca)] = cv_rmse
            print(f"  {name} + PCA({n_pca}): {cv_rmse:.4f}")

    # グループC: 最良付近を細かく
    print(f"\n--- グループC: 最良付近を細かく探索 ---")
    for prep in GROUP_C:
        name = prep["name"]
        pca_candidates = prep.pop("pca_candidates")
        for n_pca in pca_candidates:
            p = dict(prep)
            p["n_components_pca"] = n_pca
            cv_rmse, _ = cross_validate_lgbm(
                X_raw, y, groups, p, ref_spectrum, lgbm_params, n_splits
            )
            all_results[(name, n_pca)] = cv_rmse
            print(f"  {name} + PCA({n_pca}): {cv_rmse:.4f}")

    return all_results


# ============================================================
# 全結果の集計・表示
# ============================================================

def summarize_results(all_results, top_n=10):
    """
    全結果をCV-RMSE昇順で表示し、上位候補を返す。
    """
    sorted_results = sorted(all_results.items(), key=lambda x: x[1])

    print(f"\n{'='*65}")
    print(f"全結果（CV-RMSE昇順、上位{top_n}件）")
    print(f"{'='*65}")
    print(f"{'前処理':<35}  {'PCA':>6}  {'CV-RMSE':>10}")
    print(f"{'-'*55}")
    for (name, n_pca), score in sorted_results[:top_n]:
        pca_str = str(n_pca) if n_pca is not None else "なし"
        print(f"  {name:<33}  {pca_str:>6}  {score:>10.4f}")

    # 上位候補のリストを返す
    top_candidates = []
    for (name, n_pca), score in sorted_results[:top_n]:
        top_candidates.append({
            "name": name,
            "n_pca": n_pca,
            "cv_rmse": score,
        })
    return top_candidates


# ============================================================
# Phase 2: Optunaによるパラメータ最適化
# ============================================================

def run_phase2(X_raw, y, groups, ref_spectrum, top_candidates,
               all_results_lookup, n_trials=100, n_splits=5, top_k=5):
    """
    上位top_k候補それぞれについてOptunaでパラメータ最適化する。

    Parameters
    ----------
    all_results_lookup : dict  Phase 1の全結果（前処理設定を引くために使用）
    top_k              : int   最適化する候補数
    """
    # 前処理名 → 前処理設定のマッピングを作成
    prep_map = {}
    for prep in GROUP_A + GROUP_B:
        prep_map[prep["name"]] = prep
    for prep in GROUP_C:
        p = dict(prep)
        p.pop("pca_candidates", None)
        prep_map[p["name"]] = p
    # 前回結果の前処理設定も追加
    no_pca_preps = [
        {"name": "なし",               "scaler": None,  "deriv": 0, "use_osc": False, "use_pca": False},
        {"name": "SNV",                "scaler": "snv", "deriv": 0, "use_osc": False, "use_pca": False},
        {"name": "MSC",                "scaler": "msc", "deriv": 0, "use_osc": False, "use_pca": False},
        {"name": "なし + 1次微分",      "scaler": None,  "deriv": 1, "window": 9,  "use_osc": False, "use_pca": False},
        {"name": "SNV + 1次微分",       "scaler": "snv", "deriv": 1, "window": 9,  "use_osc": False, "use_pca": False},
        {"name": "MSC + 1次微分",       "scaler": "msc", "deriv": 1, "window": 9,  "use_osc": False, "use_pca": False},
        {"name": "なし + 2次微分",      "scaler": None,  "deriv": 2, "window": 13, "use_osc": False, "use_pca": False},
        {"name": "SNV + 2次微分",       "scaler": "snv", "deriv": 2, "window": 13, "use_osc": False, "use_pca": False},
        {"name": "MSC + 2次微分",       "scaler": "msc", "deriv": 2, "window": 13, "use_osc": False, "use_pca": False},
        {"name": "なし + OSC",          "scaler": None,  "deriv": 0, "use_osc": True, "use_pca": False, "n_components_osc": 2},
        {"name": "SNV + OSC",           "scaler": "snv", "deriv": 0, "use_osc": True, "use_pca": False, "n_components_osc": 2},
        {"name": "MSC + OSC",           "scaler": "msc", "deriv": 0, "use_osc": True, "use_pca": False, "n_components_osc": 2},
        {"name": "なし + 1次微分 + OSC","scaler": None,  "deriv": 1, "window": 9,  "use_osc": True, "use_pca": False, "n_components_osc": 2},
        {"name": "SNV + 1次微分 + OSC", "scaler": "snv", "deriv": 1, "window": 9,  "use_osc": True, "use_pca": False, "n_components_osc": 2},
        {"name": "MSC + 1次微分 + OSC", "scaler": "msc", "deriv": 1, "window": 9,  "use_osc": True, "use_pca": False, "n_components_osc": 2},
        {"name": "なし + 2次微分 + OSC","scaler": None,  "deriv": 2, "window": 13, "use_osc": True, "use_pca": False, "n_components_osc": 2},
        {"name": "SNV + 2次微分 + OSC", "scaler": "snv", "deriv": 2, "window": 13, "use_osc": True, "use_pca": False, "n_components_osc": 2},
        {"name": "MSC + 2次微分 + OSC", "scaler": "msc", "deriv": 2, "window": 13, "use_osc": True, "use_pca": False, "n_components_osc": 2},
    ]
    for p in no_pca_preps:
        prep_map[p["name"]] = p

    print(f"\n{'='*65}")
    print(f"Phase 2: Optunaによるパラメータ最適化（上位{top_k}候補）")
    print(f"{'='*65}")

    phase2_results = []

    for i, cand in enumerate(top_candidates[:top_k]):
        name   = cand["name"]
        n_pca  = cand["n_pca"]
        base_cv = cand["cv_rmse"]

        # 前処理設定を組み立て
        prep = dict(prep_map[name])
        if n_pca is not None:
            prep["use_pca"]          = True
            prep["n_components_pca"] = n_pca
        else:
            prep["use_pca"] = False

        pca_str = f" + PCA({n_pca})" if n_pca is not None else ""
        print(f"\n[{i+1}/{top_k}] {name}{pca_str}  (初期CV={base_cv:.4f})")

        # fold内で前処理を事前計算
        gkf = GroupKFold(n_splits=n_splits)
        fold_data = []
        for tr_idx, val_idx in gkf.split(X_raw, y, groups=groups):
            X_tr_raw, X_val_raw = X_raw[tr_idx], X_raw[val_idx]
            y_tr, y_val         = y[tr_idx], y[val_idx]
            X_tr, osc_p, pca_m = apply_preprocessing(
                X_tr_raw, y_tr, prep, ref_spectrum, fit_mode=True
            )
            X_val, _, _ = apply_preprocessing(
                X_val_raw, y_val, prep, ref_spectrum,
                fit_mode=False, osc_params=osc_p, pca_model=pca_m
            )
            fold_data.append((X_tr, X_val, y_tr, y_val))

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
                "n_estimators":      1000,
                "learning_rate":     0.05,
                "random_state":      42,
                "verbose":           -1,
            }
            fold_rmses = []
            for X_tr, X_val, y_tr, y_val in fold_data:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(period=-1)])
                y_pred = np.clip(model.predict(X_val), 0, None)
                fold_rmses.append(rmse(y_val, y_pred))
            return np.mean(fold_rmses)

        study = optuna.create_study(direction="minimize")

        def progress(study, trial):
            if (trial.number + 1) % 10 == 0:
                print(f"  trial {trial.number+1:>4}: "
                      f"CV-RMSE={trial.value:.4f}  best={study.best_value:.4f}")

        study.optimize(objective, n_trials=n_trials, callbacks=[progress])

        best_params = study.best_params
        best_params.update({"n_estimators": 1000, "learning_rate": 0.05,
                            "random_state": 42, "verbose": -1})

        print(f"\n  最適化完了: CV-RMSE={study.best_value:.4f}  "
              f"（初期値から{base_cv - study.best_value:+.4f}）")
        print(f"  最良パラメータ:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")

        phase2_results.append({
            "name":        name,
            "n_pca":       n_pca,
            "preprocessing": prep,
            "lgbm_params": best_params,
            "cv_rmse":     study.best_value,
        })

    return phase2_results



# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    # --- データ読み込み ---
    train_df, test_df, sample_sub, spec_cols = load_data()
    X_raw  = train_df[spec_cols].values
    y      = train_df["moisture_content"].values
    groups = train_df["species_number"].values
    ref_spectrum = fit_msc(X_raw)

    # LGBMの初期パラメータ（Phase 1用）
    LGBM_INIT = {
        "n_estimators":      1000,
        "learning_rate":     0.05,
        "num_leaves":        31,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_samples": 20,
        "random_state":      42,
        "verbose":           -1,
    }

    N_SPLITS  = 5
    N_TRIALS  = 100  # Phase 2のOptuna試行回数
    TOP_K     = 5    # Phase 2で最適化する上位候補数

    # ============================================================
    # Phase 1: PCA成分数の追加探索
    # ============================================================
    all_results = run_phase1(
        X_raw, y, groups, ref_spectrum, LGBM_INIT, n_splits=N_SPLITS
    )

    # 全結果を保存（途中経過の保全）
    os.makedirs("models", exist_ok=True)
    joblib.dump(all_results, "models/phase1_all_results.pkl")
    print("\nPhase 1の全結果を保存: models/phase1_all_results.pkl")

    # 上位候補を表示
    top_candidates = summarize_results(all_results, top_n=20)

    # ============================================================
    # Phase 2: Optunaによるパラメータ最適化
    # ============================================================
    phase2_results = run_phase2(
        X_raw, y, groups, ref_spectrum,
        top_candidates=top_candidates,
        all_results_lookup=all_results,
        n_trials=N_TRIALS,
        n_splits=N_SPLITS,
        top_k=TOP_K
    )

    # Phase 2の結果を表示・保存
    print(f"\n{'='*65}")
    print("Phase 2 結果まとめ（CV-RMSE昇順）")
    print(f"{'='*65}")
    print(f"{'前処理':<35}  {'PCA':>6}  {'CV-RMSE':>10}")
    print(f"{'-'*55}")
    for r in sorted(phase2_results, key=lambda x: x["cv_rmse"]):
        pca_str = str(r["n_pca"]) if r["n_pca"] is not None else "なし"
        print(f"  {r['name']:<33}  {pca_str:>6}  {r['cv_rmse']:>10.4f}")

    # 最良設定で全trainを学習・保存
    best = min(phase2_results, key=lambda x: x["cv_rmse"])
    pca_str = f" + PCA({best['n_pca']})" if best["n_pca"] is not None else ""
    print(f"\n最良: {best['name']}{pca_str}  CV-RMSE={best['cv_rmse']:.4f}")

    # best_iterを取得するためCV再実行
    cv_rmse, fold_results = cross_validate_lgbm(
        X_raw, y, groups,
        preprocessing=best["preprocessing"],
        ref_spectrum=ref_spectrum,
        lgbm_params=best["lgbm_params"],
        n_splits=N_SPLITS
    )
    avg_iter = int(np.mean([r["best_iter"] for r in fold_results if r["best_iter"]]))

    lgbm_params_final = dict(best["lgbm_params"])
    lgbm_params_final["n_estimators"] = avg_iter
    print(f"最終モデルのn_estimators: {avg_iter}")

    train_final_lgbm(
        X_raw, y,
        preprocessing=best["preprocessing"],
        ref_spectrum=ref_spectrum,
        lgbm_params=lgbm_params_final,
        save_dir="models"
    )

    joblib.dump({"phase2_results": phase2_results,
                 "best": best}, "models/phase2_results.pkl")
    print("Phase 2の結果を保存: models/phase2_results.pkl")

    # PLSとの比較
    PLS_BEST = 22.7864
    print(f"\n{'='*55}")
    print(f"PLS最良（run_017）: CV-RMSE={PLS_BEST:.4f}")
    print(f"LGBM最良（Phase2）: CV-RMSE={best['cv_rmse']:.4f}")
    if best["cv_rmse"] < PLS_BEST:
        print("→ LightGBMがPLSを上回りました")
    else:
        print("→ PLSの方が良い結果です")
    print(f"{'='*55}")

    print("\n全処理完了")