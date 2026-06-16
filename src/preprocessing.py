"""
preprocessing.py
================
近赤外スペクトル分析チャレンジ — 前処理モジュール

既存の個別前処理関数（apply_snv, apply_savgol など）を保持しつつ、
config（dict）から前処理パイプラインを組み立てる build_pipeline / apply_pipeline
を追加している。

前処理パイプラインの仕様
------------------------
config["preprocessing"] の形式（例）:
    {
        "scaler":           "snv",   # "snv" | "msc" | null
        "deriv":            2,       # 0=なし / 1=1次微分 / 2=2次微分
        "window":           13,      # Savitzky-Golayのwindow長（deriv>0のとき有効）
        "polyorder":        2,       # 多項式次数（省略可、デフォルト2）
        "use_osc":          true,    # OSC適用の有無
        "n_components_osc": 2,       # OSCの成分数（use_osc=trueのとき）
        "use_pca":          false,   # PCA次元削減の有無
        "n_components_pca": 50,      # PCA成分数（use_pca=trueのとき）
    }

ルール上の制約（再掲）
  - testの複数サンプルをまとめて使う処理は禁止
  - SNV・微分・log変換は1サンプル単独で計算できるのでルール適合
  - MSC/OSC/PCAはtrainのみからパラメータを計算してtestに適用するのでルール適合
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


# ============================================================
# 定数
# ============================================================

DATA_DIR = "data/"   # プロジェクトルートから実行することを想定


# ============================================================
# データ読み込み
# ============================================================

def load_data(data_dir: str = DATA_DIR):
    """
    train / test / sample_submit を読み込み、列名を英語に統一して返す。

    Returns
    -------
    train      : pd.DataFrame  (1322行 × 1559列)
    test       : pd.DataFrame  (550行  × 1558列)
    sample_sub : pd.DataFrame  提出フォーマットのひな形
    spec_cols  : list[str]     スペクトル列名のリスト（波数の文字列）
    """
    train = pd.read_csv(data_dir + "train.csv", encoding="shift-jis")
    test  = pd.read_csv(data_dir + "test.csv",  encoding="shift-jis")
    sample_sub = pd.read_csv(data_dir + "sample_submit.csv",
                             encoding="shift-jis", header=None)
    sample_sub.columns = ["sample_number", "moisture_content"]

    train.columns = (
        ["sample_number", "species_number", "species_name", "moisture_content"]
        + list(train.columns[4:])
    )
    test.columns = (
        ["sample_number", "species_number", "species_name"]
        + list(test.columns[3:])
    )

    spec_cols = list(train.columns[4:])

    print(f"[load_data] train: {train.shape}, test: {test.shape}")
    print(f"[load_data] スペクトル列数: {len(spec_cols)}")
    print(f"[load_data] train樹種: {sorted(train['species_number'].unique())}")
    print(f"[load_data] test樹種:  {sorted(test['species_number'].unique())}")

    return train, test, sample_sub, spec_cols


# ============================================================
# 個別前処理関数（既存コードと完全互換）
# ============================================================

def apply_snv(X: np.ndarray) -> np.ndarray:
    """SNV（Standard Normal Variate）を各サンプル独立で適用する。"""
    row_mean = X.mean(axis=1, keepdims=True)
    row_std  = X.std(axis=1, keepdims=True)
    row_std  = np.where(row_std == 0, 1e-10, row_std)
    return (X - row_mean) / row_std


def apply_savgol(X: np.ndarray,
                 window_length: int = 11,
                 polyorder: int = 2,
                 deriv: int = 2) -> np.ndarray:
    """Savitzky-Golayフィルタで微分を適用する。"""
    return savgol_filter(X,
                         window_length=window_length,
                         polyorder=polyorder,
                         deriv=deriv,
                         axis=1)


def fit_msc(X_train: np.ndarray) -> np.ndarray:
    """trainデータから基準スペクトル（全サンプルの平均）を計算して返す。"""
    return X_train.mean(axis=0)


def apply_msc(X: np.ndarray, ref_spectrum: np.ndarray) -> np.ndarray:
    """MSC（Multiplicative Scatter Correction）を適用する。"""
    X_msc = np.zeros_like(X)
    for i in range(X.shape[0]):
        A = np.column_stack([np.ones_like(ref_spectrum), ref_spectrum])
        coef, _, _, _ = np.linalg.lstsq(A, X[i], rcond=None)
        a, b = coef[0], coef[1]
        b = b if abs(b) > 1e-10 else 1e-10
        X_msc[i] = (X[i] - a) / b
    return X_msc


def fit_osc(X_train: np.ndarray, y_train: np.ndarray,
            n_components: int = 2) -> dict:
    """trainデータからOSCの補正パラメータを計算して返す。"""
    X = X_train.copy()
    y = y_train.reshape(-1, 1)
    W_list, P_list = [], []

    for _ in range(n_components):
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        w = Vt[0]
        t = (X @ w).reshape(-1, 1)
        coef = (y.T @ t) / (y.T @ y)
        t_orth = t - coef * y
        t_orth = t_orth / np.linalg.norm(t_orth)
        p = X.T @ t_orth / (t_orth.T @ t_orth)
        X = X - t_orth @ p.T
        W_list.append(w)
        P_list.append(p.flatten())

    return {
        "W": np.array(W_list),
        "P": np.array(P_list),
        "n_components": n_components,
    }


def apply_osc(X: np.ndarray, osc_params: dict) -> np.ndarray:
    """fit_osc()で計算したパラメータをスペクトルに適用する。"""
    W, P = osc_params["W"], osc_params["P"]
    X_osc = X.copy()
    for w, p in zip(W, P):
        t = X_osc @ w
        X_osc = X_osc - t[:, None] * p[None, :]
    return X_osc


def apply_log_y(y: np.ndarray) -> np.ndarray:
    """含水率にlog1p変換を適用する。"""
    return np.log1p(y)


def inverse_log_y(y_log: np.ndarray) -> np.ndarray:
    """log1p変換の逆変換（expm1）を適用する。"""
    return np.expm1(y_log)


# ============================================================
# パイプライン — config dict から前処理を組み立て・適用する
# ============================================================

def build_pipeline(X_raw: np.ndarray,
                   y: np.ndarray | None,
                   prep_cfg: dict,
                   ref_spectrum: np.ndarray,
                   fit_mode: bool = True,
                   saved_params: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    前処理設定（prep_cfg）に従ってスペクトルを変換する。

    fit_mode=True （train用）
        - OSC / PCA のパラメータをここで計算・fitする
        - 計算したパラメータを saved_params として返す
    fit_mode=False （valid / test 用）
        - saved_params に格納されたパラメータを使って変換のみ行う

    Parameters
    ----------
    X_raw        : np.ndarray           生スペクトル (n_samples, n_wavelengths)
    y            : np.ndarray | None    含水率（OSC の fit に使用。fit_mode=False は不要）
    prep_cfg     : dict                 前処理設定（詳細はモジュール docstring 参照）
    ref_spectrum : np.ndarray           MSC用の基準スペクトル（trainから計算済み）
    fit_mode     : bool                 True=train用 / False=valid・test用
    saved_params : dict | None          fit_mode=False のとき必須

    Returns
    -------
    X           : np.ndarray  前処理済みスペクトル
    params_out  : dict        {
                                  "osc_params": dict | None,
                                  "pca_model":  PCA | None,
                              }
    """
    X = X_raw.copy()
    params_out = {"osc_params": None, "pca_model": None}

    if not fit_mode:
        if saved_params is None:
            raise ValueError("fit_mode=False のとき saved_params が必要です")
        params_out = saved_params  # valに適用するパラメータを引き継ぐ

    # --- 1. スケーリング ---
    scaler = prep_cfg.get("scaler", None)
    if scaler == "snv":
        X = apply_snv(X)
    elif scaler == "msc":
        X = apply_msc(X, ref_spectrum)
    # None（スケーリングなし）のときはそのまま

    # --- 2. Savitzky-Golay 微分 ---
    deriv = prep_cfg.get("deriv", 0)
    if deriv > 0:
        window    = prep_cfg.get("window", 11)
        polyorder = prep_cfg.get("polyorder", 2)
        X = apply_savgol(X, window_length=window, polyorder=polyorder, deriv=deriv)

    # --- 3. OSC ---
    if prep_cfg.get("use_osc", False):
        n_osc = prep_cfg.get("n_components_osc", 2)
        if fit_mode:
            osc_params = fit_osc(X, y, n_components=n_osc)
            params_out["osc_params"] = osc_params
        else:
            osc_params = params_out["osc_params"]
        X = apply_osc(X, osc_params)

    # --- 4. PCA ---
    if prep_cfg.get("use_pca", False):
        n_pca = prep_cfg.get("n_components_pca", 50)
        if fit_mode:
            pca_model = PCA(n_components=n_pca, random_state=42)
            pca_model.fit(X)
            params_out["pca_model"] = pca_model
        else:
            pca_model = params_out["pca_model"]
        X = pca_model.transform(X)

    return X, params_out


# ============================================================
# 後方互換ラッパー（既存コードから呼び出せるよう残す）
# ============================================================

def get_train_data(train: pd.DataFrame, spec_cols: list,
                   use_savgol: bool = True,
                   window_length: int = 11):
    """【後方互換】trainデータから X, y, groups を返す。"""
    X_raw = train[spec_cols].values
    X = apply_snv(X_raw)
    if use_savgol:
        X = apply_savgol(X, window_length=window_length)
        print(f"[get_train_data] 前処理: SNV + 2次微分(window={window_length})")
    else:
        print("[get_train_data] 前処理: SNVのみ")

    y      = train["moisture_content"].values
    groups = train["species_number"].values

    print(f"[get_train_data] X: {X.shape}, y: {y.shape}")
    return X, y, groups


def get_test_data(test: pd.DataFrame, spec_cols: list,
                  use_savgol: bool = True,
                  window_length: int = 11):
    """【後方互換】testデータから X を返す。"""
    X_raw = test[spec_cols].values
    X = apply_snv(X_raw)
    if use_savgol:
        X = apply_savgol(X, window_length=window_length)
    print(f"[get_test_data] X: {X.shape}")
    return X
