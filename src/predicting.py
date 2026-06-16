"""
predicting.py
=============
近赤外スペクトル分析チャレンジ — 予測・提出ファイル生成

使い方
------
    # 単一モデルで予測
    python src/predicting.py --run_id pls_snv_deriv2

    # 複数モデルのアンサンブル（単純平均）
    python src/predicting.py --run_id pls_snv_deriv2 lgbm_snv_pca50 --ensemble mean

処理の流れ
----------
1. models/<run_id>/config.json を読み込む
2. models/<run_id>/model.pkl, prep_params.pkl, ref_spectrum.pkl を読み込む
3. test.csv を読み込んで前処理 → 予測
4. submission.csv を出力する
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import load_data, build_pipeline, inverse_log_y


# ============================================================
# 1 run_id 分の予測
# ============================================================

def predict_one(run_id: str,
                test_df: pd.DataFrame,
                spec_cols: list[str],
                model_dir: str = "models") -> np.ndarray:
    """
    1 つの学習済みモデルで test を予測して含水率の予測値を返す。

    Parameters
    ----------
    run_id    : str             学習時に指定した run_id
    test_df   : pd.DataFrame    test データ
    spec_cols : list[str]       スペクトル列名
    model_dir : str             models/ ディレクトリのパス

    Returns
    -------
    y_pred : np.ndarray  含水率の予測値（元のスケール、0以上）
    """
    run_dir = os.path.join(model_dir, run_id)

    # --- 設定の読み込み ---
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    prep_cfg   = cfg["preprocessing"]
    model_cfg  = cfg["model"]
    use_log_y  = cfg.get("use_log_y", False)

    # --- モデルとパラメータの読み込み ---
    model        = joblib.load(os.path.join(run_dir, "model.pkl"))
    saved_params = joblib.load(os.path.join(run_dir, "prep_params.pkl"))
    ref_spectrum = joblib.load(os.path.join(run_dir, "ref_spectrum.pkl"))

    # --- test への前処理適用 ---
    X_raw = test_df[spec_cols].values
    X_test, _ = build_pipeline(
        X_raw, y=None, prep_cfg=prep_cfg,
        ref_spectrum=ref_spectrum,
        fit_mode=False, saved_params=saved_params
    )

    print(f"[predict_one] run_id={run_id}  X_test: {X_test.shape}")

    # --- 予測 ---
    model_type = model_cfg["model_type"]

    if model_type == "pls":
        y_pred = model.predict(X_test).flatten()
    elif model_type == "lgbm":
        y_pred = model.predict(X_test)
    elif model_type == "ridge":
        y_pred = model.predict(X_test)
    elif model_type == "svr":
        sc, svr = model
        X_sc   = sc.transform(X_test)
        y_pred  = svr.predict(X_sc)
    else:
        raise ValueError(f"未対応のモデル: {model_type}")

    if use_log_y:
        y_pred = inverse_log_y(y_pred)

    y_pred = np.clip(y_pred, 0, None)

    print(f"[predict_one] 予測値: mean={y_pred.mean():.2f}, "
          f"std={y_pred.std():.2f}, "
          f"min={y_pred.min():.2f}, max={y_pred.max():.2f}")

    return y_pred


# ============================================================
# アンサンブル
# ============================================================

def ensemble_predictions(preds: list[np.ndarray],
                         method: str = "mean") -> np.ndarray:
    """
    複数の予測値をアンサンブルする。

    Parameters
    ----------
    preds  : list[np.ndarray]  各モデルの予測値
    method : str               "mean" | "median"

    Returns
    -------
    np.ndarray  アンサンブル後の予測値
    """
    stacked = np.stack(preds, axis=0)  # (n_models, n_samples)
    if method == "mean":
        return stacked.mean(axis=0)
    elif method == "median":
        return np.median(stacked, axis=0)
    else:
        raise ValueError(f"未対応のアンサンブル方法: {method}")


# ============================================================
# 提出ファイルの作成
# ============================================================

def make_submission(y_pred: np.ndarray,
                    test_df: pd.DataFrame,
                    sample_sub: pd.DataFrame,
                    output_path: str = "submission.csv"):
    """
    予測値を提出フォーマットに整形して CSV に保存する。

    Parameters
    ----------
    y_pred      : np.ndarray    含水率の予測値
    test_df     : pd.DataFrame  test データ（sample_number 列を使用）
    sample_sub  : pd.DataFrame  提出サンプル（行数確認用）
    output_path : str           出力先パス
    """
    assert len(y_pred) == len(sample_sub), (
        f"行数不一致: 予測={len(y_pred)}, サンプル={len(sample_sub)}"
    )

    submission = pd.DataFrame({
        "sample_number":    test_df["sample_number"].values,
        "moisture_content": y_pred,
    })

    submission.to_csv(output_path, index=False, header=False)
    print(f"\n[make_submission] 提出ファイル保存: {output_path}")
    print(submission.head(5).to_string(index=False))


# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="近赤外スペクトル分析チャレンジ — 予測スクリプト"
    )
    parser.add_argument(
        "--run_id", type=str, nargs="+", required=True,
        help="使用する run_id（複数指定でアンサンブル）"
    )
    parser.add_argument(
        "--ensemble", type=str, default="mean",
        choices=["mean", "median"],
        help="アンサンブル方法（デフォルト: mean）"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models",
        help="models/ ディレクトリのパス（デフォルト: models）"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/",
        help="data/ ディレクトリのパス（デフォルト: data/）"
    )
    parser.add_argument(
        "--output", type=str, default="submission.csv",
        help="出力ファイル名（デフォルト: submission.csv）"
    )
    args = parser.parse_args()

    # データ読み込み
    train_df, test_df, sample_sub, spec_cols = load_data(args.data_dir)

    # 各 run_id で予測
    all_preds = []
    for rid in args.run_id:
        print(f"\n{'='*50}")
        print(f"run_id: {rid}")
        print(f"{'='*50}")
        y_pred = predict_one(rid, test_df, spec_cols, args.model_dir)
        all_preds.append(y_pred)

    # アンサンブル（1 モデルのときはそのまま）
    if len(all_preds) == 1:
        final_pred = all_preds[0]
    else:
        final_pred = ensemble_predictions(all_preds, method=args.ensemble)
        print(f"\n[ensemble] {args.ensemble}  "
              f"mean={final_pred.mean():.2f}, std={final_pred.std():.2f}")

    # 提出ファイル作成
    make_submission(final_pred, test_df, sample_sub, args.output)
    print("\n予測モジュール: 正常終了")
