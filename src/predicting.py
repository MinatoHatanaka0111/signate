"""
predicting.py
=============
近赤外スペクトル分析チャレンジ — 予測モジュール

担当する処理:
  1. 学習済みモデルとパラメータの読み込み
  2. testデータへの前処理・予測適用
  3. 提出ファイル（submission.csv）の作成
"""

import numpy as np
import pandas as pd
import joblib
import sys
sys.path.append("src")

from preprocessing import (load_data, apply_snv, apply_savgol,
                            apply_msc, apply_osc, inverse_log_y)


# ============================================================
# 予測と提出ファイル作成
# ============================================================

def predict_and_submit(model, test, X_test, sample_sub,
                       use_log_y=False,
                       output_path="submission.csv"):
    """
    学習済みモデルでtestを予測し、提出ファイルを作成する。

    Parameters
    ----------
    model       : PLSRegression  学習済みモデル
    test        : pd.DataFrame   testデータ
    X_test      : np.ndarray     前処理済みtestスペクトル
    sample_sub  : pd.DataFrame   提出フォーマットのひな形
    use_log_y   : bool           Trueのとき予測値をexp逆変換する
    output_path : str            出力先のパス

    Returns
    -------
    submission  : pd.DataFrame   提出ファイルの内容
    """
    y_pred = model.predict(X_test).flatten()

    # log変換して学習した場合は逆変換で元のスケールに戻す
    if use_log_y:
        y_pred = inverse_log_y(y_pred)

    # 含水率は0以上のため負の予測値を0にクリップ
    y_pred = np.clip(y_pred, 0, None)

    submission = pd.DataFrame({
        "sample_number":    test["sample_number"].values,
        "moisture_content": y_pred
    })

    assert len(submission) == len(sample_sub), \
        f"行数不一致: submission={len(submission)}, sample_sub={len(sample_sub)}"

    submission.to_csv(output_path, index=False, header=False)

    print(f"[predict_and_submit] 予測完了")
    print(f"[predict_and_submit] 予測値の統計:")
    print(f"  mean={y_pred.mean():.2f}, std={y_pred.std():.2f}, "
          f"min={y_pred.min():.2f}, max={y_pred.max():.2f}")
    print(f"[predict_and_submit] 提出ファイル保存: {output_path}")

    return submission


# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    # --- 最良設定の読み込み（learning.pyが自動保存）---
    best_params   = joblib.load("models/best_params.pkl")
    scaler        = best_params["scaler"]
    window_length = best_params["window_length"]
    use_log_y     = best_params.get("use_log_y", False)
    deriv         = best_params.get("deriv", 2)
    use_osc       = best_params.get("use_osc", False)
    print(f"[main] 最良設定: scaler={scaler}, window={window_length}, "
          f"deriv={deriv}, osc={use_osc}, log_y={use_log_y}")

    # --- データ読み込み ---
    train, test, sample_sub, spec_cols = load_data()
    X_raw_test = test[spec_cols].values

    # --- testへの前処理適用（trainと同じ設定） ---
    if scaler == "snv":
        X_scaled = apply_snv(X_raw_test)
    else:  # msc
        ref_spectrum = joblib.load("models/ref_spectrum.pkl")
        X_scaled = apply_msc(X_raw_test, ref_spectrum)

    X_test = apply_savgol(X_scaled, window_length=window_length, deriv=deriv) \
        if deriv > 0 else X_scaled

    # OSCが有効な場合はtrainで計算したパラメータを適用
    if use_osc:
        osc_params = joblib.load("models/osc_params.pkl")
        X_test = apply_osc(X_test, osc_params)

    print(f"[main] X_test: {X_test.shape}")

    # --- 学習済みモデルの読み込み ---
    model = joblib.load("models/pls_model.pkl")
    print(f"[main] モデル読み込み完了: 成分数={model.n_components}")

    # --- 予測と提出ファイル作成 ---
    submission = predict_and_submit(
        model, test, X_test, sample_sub,
        use_log_y=use_log_y,
        output_path="submission.csv"
    )

    print("\n予測モジュール: 正常終了")
    print("\n提出ファイルの先頭5行:")
    print(submission.head())