"""
predicting_lgbm.py
==================
近赤外スペクトル分析チャレンジ — LightGBM予測モジュール

担当する処理:
  1. 学習済みLightGBMモデルと前処理パラメータの読み込み
  2. testデータへの前処理・予測適用
  3. 提出ファイル（submission.csv）の作成
"""

import numpy as np
import pandas as pd
import joblib
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
sys.path.append("src")

from preprocessing import load_data
from learning_lgbm import apply_preprocessing


if __name__ == "__main__":
    # --- 学習済みモデルと前処理設定の読み込み ---
    model        = joblib.load("models/lgbm_model.pkl")
    preprocessing = joblib.load("models/lgbm_preprocessing.pkl")
    ref_spectrum = joblib.load("models/ref_spectrum.pkl")

    # OSC・PCAパラメータが存在すれば読み込む
    try:
        osc_params = joblib.load("models/osc_params.pkl")
    except FileNotFoundError:
        osc_params = None
    try:
        pca_model = joblib.load("models/pca_model.pkl")
    except FileNotFoundError:
        pca_model = None

    print(f"[main] モデル読み込み完了")
    print(f"[main] 前処理: {preprocessing['name']}")

    # --- データ読み込み ---
    train_df, test_df, sample_sub, spec_cols = load_data()
    X_raw_test = test_df[spec_cols].values

    # --- testへの前処理適用（trainと同じ設定）---
    X_test, _, _ = apply_preprocessing(
        X_raw_test, y=None, preprocessing=preprocessing,
        ref_spectrum=ref_spectrum,
        fit_mode=False,
        osc_params=osc_params,
        pca_model=pca_model
    )
    print(f"[main] X_test: {X_test.shape}")

    # --- 予測 ---
    y_pred = np.clip(model.predict(X_test), 0, None)

    print(f"[main] 予測値の統計:")
    print(f"  mean={y_pred.mean():.2f}, std={y_pred.std():.2f}, "
          f"min={y_pred.min():.2f}, max={y_pred.max():.2f}")

    # --- 提出ファイルの作成 ---
    submission = pd.DataFrame({
        "sample_number":    test_df["sample_number"].values,
        "moisture_content": y_pred
    })

    assert len(submission) == len(sample_sub), \
        f"行数不一致: {len(submission)} vs {len(sample_sub)}"

    submission.to_csv("submission.csv", index=False, header=False)
    print(f"[main] 提出ファイル保存: submission.csv")

    print("\n提出ファイルの先頭5行:")
    print(submission.head())
    print("\nLightGBM予測モジュール: 正常終了")
