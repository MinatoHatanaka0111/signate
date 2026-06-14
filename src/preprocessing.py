"""
preprocessing.py
================
近赤外スペクトル分析チャレンジ — 前処理モジュール

担当する処理:
  1. データの読み込みと列名の整理
  2. SNV（標準正規変量変換）の適用
  3. MSC（乗法的散乱補正）の適用
  4. Savitzky-Golayフィルタによる2次微分の適用
  5. 目的変数（含水率）のlog変換・逆変換
  6. train用・test用それぞれの前処理関数を提供

ルール上の制約:
  - testの複数サンプルをまとめて使う処理は禁止
  - SNV・2次微分・log変換は1サンプル単独で計算できるのでルール適合
  - MSCはtrainの平均スペクトルを基準として使うが、
    基準はtrainのみから計算してtestに適用するのでルール適合
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


# ============================================================
# 定数
# ============================================================

DATA_DIR = "data/"  # データの場所（プロジェクトルートから実行することを想定）


# ============================================================
# データ読み込み
# ============================================================

def load_data():
    """
    train / test / sample_submit を読み込み、列名を英語に統一して返す。

    Returns
    -------
    train : pd.DataFrame  (1322行 × 1559列)
    test  : pd.DataFrame  (550行  × 1558列)
    sample_sub : pd.DataFrame  提出フォーマットのひな形
    spec_cols  : list[str]     スペクトル列名のリスト（波数の文字列）
    """
    train = pd.read_csv(DATA_DIR + "train.csv", encoding="shift-jis")
    test  = pd.read_csv(DATA_DIR + "test.csv",  encoding="shift-jis")
    # sample_submitは先頭行が実データ（95,50）のためheader=Noneで読み直す
    sample_sub = pd.read_csv(DATA_DIR + "sample_submit.csv",
                             encoding="shift-jis", header=None)
    sample_sub.columns = ["sample_number", "moisture_content"]

    # 列名を英語に統一
    train.columns = (
        ["sample_number", "species_number", "species_name", "moisture_content"]
        + list(train.columns[4:])
    )
    test.columns = (
        ["sample_number", "species_number", "species_name"]
        + list(test.columns[3:])
    )

    # スペクトル列名のリスト（波数の文字列）
    spec_cols = list(train.columns[4:])

    print(f"[load_data] train: {train.shape}, test: {test.shape}")
    print(f"[load_data] スペクトル列数: {len(spec_cols)}")
    print(f"[load_data] train樹種: {sorted(train['species_number'].unique())}")
    print(f"[load_data] test樹種:  {sorted(test['species_number'].unique())}")

    return train, test, sample_sub, spec_cols


# ============================================================
# SNV（標準正規変量変換）
# ============================================================

def apply_snv(X: np.ndarray) -> np.ndarray:
    """
    SNV（Standard Normal Variate）を適用する。

    各サンプル（行）ごとに独立して計算するため、
    testの1サンプルだけでも正しく動作する。

    処理内容:
        SNV後の値 = (元の値 - そのサンプルの全波数平均) / そのサンプルの全波数標準偏差

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wavelengths)

    Returns
    -------
    X_snv : np.ndarray, shape (n_samples, n_wavelengths)
    """
    # 各行（サンプル）の平均と標準偏差を計算
    row_mean = X.mean(axis=1, keepdims=True)   # shape: (n_samples, 1)
    row_std  = X.std(axis=1, keepdims=True)    # shape: (n_samples, 1)

    # 標準偏差が0のサンプルは除算エラーになるので回避（実際はほぼ起きない）
    row_std = np.where(row_std == 0, 1e-10, row_std)

    X_snv = (X - row_mean) / row_std
    return X_snv


# ============================================================
# Savitzky-Golayフィルタ（2次微分）
# ============================================================

def apply_savgol(X: np.ndarray,
                 window_length: int = 11,
                 polyorder: int = 2,
                 deriv: int = 2) -> np.ndarray:
    """
    Savitzky-Golayフィルタで2次微分を適用する。

    単純な差分ではなく周辺N点を多項式でフィッティングしてから
    微分するため、ノイズを増幅させずにベースラインを除去できる。

    SNVの後に適用することで:
        SNV  → スケールのずれを除去
        2次微分 → 残ったベースラインの傾き・曲がりを除去 + ピーク鮮明化

    各サンプル（行）ごとに独立して計算するため、
    testの1サンプルだけでも正しく動作する。

    Parameters
    ----------
    X             : np.ndarray, shape (n_samples, n_wavelengths)
    window_length : int  平滑化に使う点数（奇数、大きいほど平滑化が強い）
                         初期値11は近赤外分析の文献でよく使われる値
    polyorder     : int  フィッティングする多項式の次数（2か3が一般的）
    deriv         : int  微分の次数（2 = 2次微分）

    Returns
    -------
    X_sg : np.ndarray, shape (n_samples, n_wavelengths)
    """
    # savgol_filterはaxis=1で各行（サンプル）ごとに適用
    X_sg = savgol_filter(X,
                         window_length=window_length,
                         polyorder=polyorder,
                         deriv=deriv,
                         axis=1)
    return X_sg


# ============================================================
# MSC（乗法的散乱補正）
# ============================================================

def fit_msc(X_train: np.ndarray) -> np.ndarray:
    """
    trainデータから基準スペクトル（全サンプルの平均）を計算して返す。

    MSCはSNVと同様に散乱の影響を補正するが、
    「理想的な基準スペクトルへの線形回帰」という物理的根拠がある。

    基準スペクトルはtrainのみから計算し、
    joblib等で保存してtestにも同じ基準を適用する。

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_wavelengths)  生スペクトル

    Returns
    -------
    ref_spectrum : np.ndarray, shape (n_wavelengths,)  基準スペクトル
    """
    ref_spectrum = X_train.mean(axis=0)  # 全サンプルの波数ごとの平均
    return ref_spectrum


def apply_msc(X: np.ndarray, ref_spectrum: np.ndarray) -> np.ndarray:
    """
    MSC（Multiplicative Scatter Correction）を適用する。

    各サンプルを基準スペクトルに線形回帰し、
    その切片・傾きのずれを補正する。

    処理内容（サンプルiについて）:
        1. X[i] を ref_spectrum に回帰: X[i] ≈ a + b × ref_spectrum
        2. 補正: X_msc[i] = (X[i] - a) / b

    1サンプルずつ独立して処理できるのでルール適合。

    Parameters
    ----------
    X            : np.ndarray, shape (n_samples, n_wavelengths)
    ref_spectrum : np.ndarray, shape (n_wavelengths,)  fit_msc()で計算した基準

    Returns
    -------
    X_msc : np.ndarray, shape (n_samples, n_wavelengths)
    """
    X_msc = np.zeros_like(X)
    for i in range(X.shape[0]):
        # ref_spectrumにX[i]を線形回帰: [1, ref] × [a, b]^T = X[i]
        A = np.column_stack([np.ones_like(ref_spectrum), ref_spectrum])
        coef, _, _, _ = np.linalg.lstsq(A, X[i], rcond=None)
        a, b = coef[0], coef[1]
        # b=0は理論上起きないが念のため回避
        b = b if abs(b) > 1e-10 else 1e-10
        X_msc[i] = (X[i] - a) / b
    return X_msc


# ============================================================
# OSC（直交信号補正）
# ============================================================

def fit_osc(X_train: np.ndarray, y_train: np.ndarray,
            n_components: int = 2) -> dict:
    """
    trainデータからOSCの補正パラメータを計算して返す。

    OSCは「yと直交するスペクトルの変動成分」を除去する。
    「直交する」= 含水率yと無関係な変動 = 樹種間の違いなど。

    アルゴリズム（Wold 1998の単純OSC）:
        1. Xの第1主成分tを計算する（PCAの要領）
        2. tからyへの回帰成分を除去して「yと直交するt_orth」を作る
        3. t_orthでXを回帰してローディングpを求める
        4. X -= t_orth × p.T  （直交成分を除去）
        5. 1〜4をn_components回繰り返す

    補正パラメータ（W, P）をtrainから計算し、
    保存してtestにも同じパラメータで適用する。

    Parameters
    ----------
    X_train    : np.ndarray, shape (n_train, n_wavelengths)  前処理済みスペクトル
    y_train    : np.ndarray, shape (n_train,)                含水率
    n_components: int  除去する直交成分の数（デフォルト2）

    Returns
    -------
    osc_params : dict  {"W": np.ndarray, "P": np.ndarray}
                 testへの適用に必要なパラメータ
    """
    from sklearn.cross_decomposition import PLSRegression

    X = X_train.copy()
    y = y_train.reshape(-1, 1)

    W_list = []
    P_list = []

    for _ in range(n_components):
        # Step1: Xの第1主成分の方向（重みベクトルw）を求める
        # SVDの第1右特異ベクトルを使う
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        w = Vt[0]  # shape: (n_wavelengths,)

        # Step2: スコアt = X @ w
        t = X @ w  # shape: (n_train,)
        t = t.reshape(-1, 1)

        # Step3: tからyへの成分を除去（yと直交化）
        # t_orth = t - t @ (y.T @ t)^{-1} @ y.T @ t ... を簡略化
        # = t - (y^T t / y^T y) * y
        coef = (y.T @ t) / (y.T @ y)  # スカラー
        t_orth = t - coef * y          # shape: (n_train, 1)

        # Step4: t_orthを正規化
        t_orth = t_orth / np.linalg.norm(t_orth)

        # Step5: ローディングp = X.T @ t_orth / (t_orth.T @ t_orth)
        p = X.T @ t_orth / (t_orth.T @ t_orth)  # shape: (n_wavelengths, 1)

        # Step6: 直交成分をXから除去
        X = X - t_orth @ p.T

        W_list.append(w)
        P_list.append(p.flatten())

    osc_params = {
        "W": np.array(W_list),  # shape: (n_components, n_wavelengths)
        "P": np.array(P_list),  # shape: (n_components, n_wavelengths)
        "n_components": n_components,
    }
    return osc_params


def apply_osc(X: np.ndarray, osc_params: dict) -> np.ndarray:
    """
    fit_osc()で計算したパラメータをスペクトルに適用する。

    trainで計算したW・Pを使ってtestにも同じ補正を適用できる。
    1サンプルずつ独立して処理できるのでルール適合。

    Parameters
    ----------
    X          : np.ndarray, shape (n_samples, n_wavelengths)
    osc_params : dict  fit_osc()の返り値

    Returns
    -------
    X_osc : np.ndarray, shape (n_samples, n_wavelengths)
    """
    W = osc_params["W"]  # shape: (n_components, n_wavelengths)
    P = osc_params["P"]  # shape: (n_components, n_wavelengths)

    X_osc = X.copy()
    for w, p in zip(W, P):
        # スコア t = X @ w
        t = X_osc @ w  # shape: (n_samples,)
        # 直交成分を除去: X -= t[:, None] * p[None, :]
        X_osc = X_osc - t[:, None] * p[None, :]

    return X_osc


# ============================================================
# 目的変数のlog変換・逆変換
# ============================================================

def apply_log_y(y: np.ndarray) -> np.ndarray:
    """
    含水率にlog1p変換（log(1 + y)）を適用する。

    なぜlog1pを使うか:
        含水率の分布は右に強く偏っている（歪度1.65）。
        log変換することで分布が正規分布に近づき、
        大きな含水率の外れ値がモデルに与える影響を抑えられる。
        log1p = log(1 + y) は y=0 のときも定義される（log(0)を回避）。

    Parameters
    ----------
    y : np.ndarray  含水率（0以上）

    Returns
    -------
    np.ndarray  log1p変換後の値
    """
    return np.log1p(y)


def inverse_log_y(y_log: np.ndarray) -> np.ndarray:
    """
    log1p変換の逆変換（expm1 = exp(y) - 1）を適用する。

    予測時にlog空間で予測した値を元のスケールに戻すために使う。

    Parameters
    ----------
    y_log : np.ndarray  log1p変換済みの予測値

    Returns
    -------
    np.ndarray  元のスケールに戻した含水率
    """
    return np.expm1(y_log)


# ============================================================
# train用前処理
# ============================================================

def get_train_data(train: pd.DataFrame, spec_cols: list,
                   use_savgol: bool = True,
                   window_length: int = 11):
    """
    trainデータから特徴量行列Xと目的変数yを作成する。

    処理の流れ:
        生スペクトル → SNV → (2次微分) → X（特徴量行列）
        含水率 → y（目的変数）

    Parameters
    ----------
    train         : pd.DataFrame  load_data()で読み込んだtrainデータ
    spec_cols     : list[str]     スペクトル列名のリスト
    use_savgol    : bool          Trueのとき2次微分を適用する（デフォルトTrue）
    window_length : int           Savitzky-Golayの平滑化点数（デフォルト11）

    Returns
    -------
    X         : np.ndarray, shape (1322, 1555)  前処理済みスペクトル
    y         : np.ndarray, shape (1322,)        含水率
    groups    : np.ndarray, shape (1322,)        樹種番号（CV用）
    """
    # 生スペクトルを行列として取り出す
    X_raw = train[spec_cols].values  # shape: (1322, 1555)

    # SNV適用
    X = apply_snv(X_raw)

    # 2次微分（SNVの後に適用）
    if use_savgol:
        X = apply_savgol(X, window_length=window_length)
        print(f"[get_train_data] 前処理: SNV + 2次微分(window={window_length})")
    else:
        print(f"[get_train_data] 前処理: SNVのみ")

    # 目的変数（含水率）
    y = train["moisture_content"].values  # shape: (1322,)

    # グループ（樹種番号）— GroupKFold用
    groups = train["species_number"].values  # shape: (1322,)

    print(f"[get_train_data] X: {X.shape}, y: {y.shape}")
    print(f"[get_train_data] 含水率 mean={y.mean():.1f}, std={y.std():.1f}, "
          f"min={y.min():.1f}, max={y.max():.1f}")

    return X, y, groups


# ============================================================
# test用前処理
# ============================================================

def get_test_data(test: pd.DataFrame, spec_cols: list,
                  use_savgol: bool = True,
                  window_length: int = 11):
    """
    testデータから特徴量行列Xを作成する。

    trainと全く同じ前処理を適用する。
    SNV・2次微分はどちらもサンプル単独で計算できるため
    trainの情報は不要でルール適合。

    Parameters
    ----------
    test          : pd.DataFrame  load_data()で読み込んだtestデータ
    spec_cols     : list[str]     スペクトル列名のリスト
    use_savgol    : bool          Trueのとき2次微分を適用する（デフォルトTrue）
    window_length : int           Savitzky-Golayの平滑化点数（デフォルト11）

    Returns
    -------
    X : np.ndarray, shape (550, 1555)  前処理済みスペクトル
    """
    X_raw = test[spec_cols].values  # shape: (550, 1555)

    # SNV適用
    X = apply_snv(X_raw)

    # 2次微分（trainと同じ設定を使う）
    if use_savgol:
        X = apply_savgol(X, window_length=window_length)

    print(f"[get_test_data] X: {X.shape}")

    return X


# ============================================================
# 動作確認（このファイルを直接実行したとき）
# ============================================================

if __name__ == "__main__":
    train, test, sample_sub, spec_cols = load_data()

    print("\n=== SNVのみ ===")
    X_train_snv, y_train, groups = get_train_data(
        train, spec_cols, use_savgol=False)

    print("\n=== SNV + 2次微分(window=11) ===")
    X_train_sg, _, _ = get_train_data(
        train, spec_cols, use_savgol=True, window_length=11)

    print("\n=== 2次微分適用後の変化確認 ===")
    print(f"SNVのみ      — 平均: {X_train_snv.mean():.4f}, "
          f"標準偏差: {X_train_snv.std():.4f}")
    print(f"SNV+2次微分  — 平均: {X_train_sg.mean():.6f}, "
          f"標準偏差: {X_train_sg.std():.6f}")
    print("（2次微分後は値のスケールが小さくなるが問題なし）")

    X_test = get_test_data(test, spec_cols, use_savgol=True, window_length=11)
    print("\n前処理モジュール: 正常終了")