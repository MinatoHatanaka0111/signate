"""
learning.py
===========
近赤外スペクトル分析チャレンジ — 学習モジュール

担当する処理:
  1. GroupKFoldによるクロスバリデーション（CV）
  2. 最適な成分数の探索
  3. 全trainデータでの最終モデル学習
  4. CV結果の詳細表示（樹種ごとのRMSEなど）
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import joblib
import os

# 前処理モジュールを読み込む
import sys
sys.path.append("src")
from preprocessing import (load_data, get_train_data,
                           apply_snv, apply_savgol,
                           fit_msc, apply_msc,
                           fit_osc, apply_osc,
                           apply_log_y, inverse_log_y)


# ============================================================
# 評価指標
# ============================================================

def rmse(y_true, y_pred):
    """RMSE（平均二乗誤差の平方根）を計算する。小さいほど良い。"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ============================================================
# GroupKFold クロスバリデーション
# ============================================================

def cross_validate_pls(X, y, groups, n_components,
                       n_splits=5, use_log_y=False):
    """
    GroupKFoldでPLSのCVスコアを計算する。

    GroupKFoldを使う理由:
        trainとtestで樹種が完全に分かれているため、
        validationにも学習していない樹種を使わないと
        実際の提出スコアとの乖離が大きくなる。

    Parameters
    ----------
    X            : np.ndarray  前処理済みスペクトル
    y            : np.ndarray  含水率（元のスケール）
    groups       : np.ndarray  樹種番号（fold分割の基準）
    n_components : int         PLSの成分数
    n_splits     : int         fold数（樹種数13に対して5が妥当）
    use_log_y    : bool        Trueのとき含水率をlog変換してから学習する

    Returns
    -------
    cv_rmse      : float       全foldのRMSE平均（元のスケール）
    fold_results : list[dict]  各foldの詳細結果
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, valid_idx) in enumerate(
            gkf.split(X, y, groups=groups)):

        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]
        groups_val  = groups[valid_idx]

        # log変換（学習時のみ適用）
        y_tr_fit = apply_log_y(y_tr) if use_log_y else y_tr

        # PLSモデルの学習
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_tr, y_tr_fit)

        # 予測
        y_pred = pls.predict(X_val).flatten()

        # log変換した場合は逆変換で元のスケールに戻す
        if use_log_y:
            y_pred = inverse_log_y(y_pred)

        # 含水率は0以上のため負の予測値を0にクリップ
        y_pred = np.clip(y_pred, 0, None)

        # RMSEは元のスケールで計算（提出スコアと同じ基準）
        fold_rmse = rmse(y_val, y_pred)

        # このfoldに含まれる樹種ごとのRMSEも計算
        species_rmse = {}
        for sp in np.unique(groups_val):
            mask = groups_val == sp
            species_rmse[int(sp)] = rmse(y_val[mask], y_pred[mask])

        fold_results.append({
            "fold":        fold_idx + 1,
            "rmse":        fold_rmse,
            "n_valid":     len(y_val),
            "species_rmse": species_rmse,
        })

    cv_rmse = np.mean([r["rmse"] for r in fold_results])
    return cv_rmse, fold_results


# ============================================================
# 最適成分数の探索
# ============================================================

def search_best_n_components(X, y, groups,
                              candidates=None, n_splits=5):
    """
    成分数を変えながらCVを繰り返し、最良の成分数を探す。

    Parameters
    ----------
    candidates : list[int]  試す成分数のリスト
                            Noneのとき [2,4,6,8,10,12,15,20,25,30] を使う

    Returns
    -------
    best_n     : int        最良の成分数
    search_log : list[dict] 各成分数のCVスコア
    """
    if candidates is None:
        candidates = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

    print(f"\n{'='*50}")
    print(f"成分数の探索 (GroupKFold, n_splits={n_splits})")
    print(f"{'='*50}")
    print(f"{'成分数':>6}  {'CV-RMSE':>10}")
    print(f"{'-'*20}")

    search_log = []
    best_rmse = np.inf
    best_n    = candidates[0]

    for n in candidates:
        cv_rmse, _ = cross_validate_pls(X, y, groups,
                                         n_components=n,
                                         n_splits=n_splits)
        search_log.append({"n_components": n, "cv_rmse": cv_rmse})
        marker = " ← 現時点の最良" if cv_rmse < best_rmse else ""
        print(f"{n:>6}  {cv_rmse:>10.4f}{marker}")

        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_n    = n

    print(f"\n最良の成分数: {best_n}  (CV-RMSE: {best_rmse:.4f})")
    return best_n, search_log


# ============================================================
# window_length × n_components の総当たり探索
# ============================================================

def search_best_params(train_raw, y, groups, spec_cols,
                       window_candidates=None,
                       n_comp_candidates=None,
                       n_splits=5,
                       scaler="snv",
                       use_log_y=False,
                       ref_spectrum=None,
                       deriv=2):
    """
    window_length × n_components を総当たりで探索し最良の組み合わせを返す。

    Parameters
    ----------
    train_raw        : np.ndarray  生スペクトル (n_samples, n_wavelengths)
    y                : np.ndarray  含水率（元のスケール）
    groups           : np.ndarray  樹種番号
    spec_cols        : list[str]   波数列名（表示用）
    window_candidates: list[int]   試すwindow_lengthのリスト
    n_comp_candidates: list[int]   試すn_componentsのリスト
    n_splits         : int         GroupKFoldのfold数
    scaler           : str         "snv" または "msc"
    use_log_y        : bool        Trueのとき含水率をlog変換して学習
    ref_spectrum     : np.ndarray  MSC用の基準スペクトル（scaler="msc"のとき必須）
    deriv            : int         微分の次数（0=微分なし、1=1次微分、2=2次微分）
                                   deriv=0のときwindow_candidatesは無視される

    Returns
    -------
    best_params : dict  {"window_length": int or None, "n_components": int}
    search_log  : list[dict]  全組み合わせのCVスコア
    """
    if n_comp_candidates is None:
        n_comp_candidates = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

    # deriv=0（微分なし）のときはwindowループが不要
    # window_candidatesをNoneにして分岐で処理する
    if deriv == 0:
        window_loop = [None]  # windowなしで1回だけループ
    else:
        if window_candidates is None:
            window_candidates = [5, 7, 11, 15, 21]
        window_loop = window_candidates

    total = len(window_loop) * len(n_comp_candidates)
    log_label   = " + log(y)" if use_log_y else ""
    deriv_label = "微分なし" if deriv == 0 else f"{deriv}次微分"
    print(f"\n{'='*60}")
    print(f"パラメータ探索: {scaler.upper()} + {deriv_label}{log_label} "
          f"(GroupKFold, n_splits={n_splits})")
    if deriv == 0:
        print(f"組み合わせ数: {len(n_comp_candidates)}通り（windowなし）")
        print(f"{'n_comp':>8}  {'CV-RMSE':>10}")
        print(f"{'-'*20}")
    else:
        print(f"組み合わせ数: {len(window_loop)} × "
              f"{len(n_comp_candidates)} = {total}通り")
        print(f"{'window':>8}  {'n_comp':>8}  {'CV-RMSE':>10}")
        print(f"{'-'*30}")

    search_log  = []
    best_rmse   = np.inf
    best_params = {"window_length": window_loop[0],
                   "n_components":  n_comp_candidates[0]}

    for window in window_loop:
        # スケーリング（SNV or MSC）を適用
        if scaler == "snv":
            X_scaled = apply_snv(train_raw)
        elif scaler == "msc":
            if ref_spectrum is None:
                raise ValueError("scaler='msc'のときref_spectrumが必要です")
            X_scaled = apply_msc(train_raw, ref_spectrum)
        else:
            raise ValueError(f"scalerは'snv'か'msc'を指定してください: {scaler}")

        # 微分を適用（deriv=0のときはスキップ）
        X = apply_savgol(X_scaled, window_length=window, deriv=deriv) \
            if deriv > 0 else X_scaled

        for n_comp in n_comp_candidates:
            cv_rmse, _ = cross_validate_pls(
                X, y, groups,
                n_components=n_comp,
                n_splits=n_splits,
                use_log_y=use_log_y
            )
            search_log.append({
                "window_length": window,
                "n_components":  n_comp,
                "cv_rmse":       cv_rmse,
            })

            marker = " ← 現時点の最良" if cv_rmse < best_rmse else ""
            if deriv == 0:
                print(f"{n_comp:>8}  {cv_rmse:>10.4f}{marker}")
            else:
                print(f"{window:>8}  {n_comp:>8}  {cv_rmse:>10.4f}{marker}")

            if cv_rmse < best_rmse:
                best_rmse = cv_rmse
                best_params = {"window_length": window,
                               "n_components":  n_comp}

    if deriv == 0:
        print(f"\n最良の成分数: {best_params['n_components']}  "
              f"(CV-RMSE: {best_rmse:.4f})")
    else:
        print(f"\n最良の組み合わせ: window={best_params['window_length']}, "
              f"n_components={best_params['n_components']}  "
              f"(CV-RMSE: {best_rmse:.4f})")
    return best_params, search_log


# ============================================================
# 最終モデルの学習（全trainデータを使う）
# ============================================================

def train_final_model(X, y, n_components,
                      save_path="pls_model.pkl",
                      use_log_y=False):
    """
    最適な成分数で全trainデータを使ってPLSを学習し、モデルを保存する。

    Parameters
    ----------
    X            : np.ndarray  全trainスペクトル
    y            : np.ndarray  全train含水率（元のスケール）
    n_components : int         最適成分数
    save_path    : str         モデルの保存先
    use_log_y    : bool        Trueのとき含水率をlog変換して学習する

    Returns
    -------
    model : PLSRegression  学習済みモデル
    """
    y_fit = apply_log_y(y) if use_log_y else y

    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X, y_fit)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"\n[train_final_model] 学習完了: 成分数={n_components}, "
          f"log変換={'あり' if use_log_y else 'なし'}")
    print(f"[train_final_model] モデルを保存: {save_path}")

    return model


# ============================================================
# CV結果の詳細表示
# ============================================================

def print_cv_detail(fold_results, train):
    """
    CVの詳細結果を表示する。
    樹種ごとのRMSEを見ることで、どの樹種が難しいかがわかる。
    """
    # 樹種番号 → 樹種名のマッピング
    sp_name_map = dict(zip(train["species_number"], train["species_name"]))

    print(f"\n{'='*55}")
    print("樹種ごとのCV-RMSE（validationに出現したfoldの平均）")
    print(f"{'='*55}")

    # 樹種ごとにRMSEをまとめる
    sp_rmse_all = {}
    for fold in fold_results:
        for sp, r in fold["species_rmse"].items():
            sp_rmse_all.setdefault(sp, []).append(r)

    print(f"{'樹種番号':>6}  {'樹種名':<12}  {'RMSE':>8}  {'出現fold数':>6}")
    print(f"{'-'*40}")
    for sp in sorted(sp_rmse_all.keys()):
        rmse_list = sp_rmse_all[sp]
        mean_rmse = np.mean(rmse_list)
        name = sp_name_map.get(sp, "不明")
        print(f"{sp:>6}  {name:<12}  {mean_rmse:>8.4f}  {len(rmse_list):>6}")


# ============================================================
# 第2段階探索用の候補リスト生成
# ============================================================

def make_fine_candidates(best_value, margin, step=1, min_val=1):
    """
    best_valueを中心にmargin分前後の候補リストを生成する。

    第2段階探索で「第1段階の最良値の周辺を細かく探索する」ために使う。

    Parameters
    ----------
    best_value : int  第1段階で得られた最良値
    margin     : int  前後いくつ探索するか
    step       : int  刻み幅（デフォルト1）
    min_val    : int  候補の最小値（デフォルト1）

    Returns
    -------
    candidates : list[int]

    例: best_value=15, margin=3, step=1
        → [12, 13, 14, 15, 16, 17, 18]
    """
    start = max(min_val, best_value - margin)
    stop  = best_value + margin + 1
    return list(range(start, stop, step))


# ============================================================
# OSC + window_length × n_components の総当たり探索
# ============================================================

def search_best_params_with_osc(train_raw, y, groups, spec_cols,
                                 window_candidates=None,
                                 n_comp_candidates=None,
                                 n_comp_osc_candidates=None,
                                 n_splits=5,
                                 scaler="snv",
                                 ref_spectrum=None,
                                 deriv=2):
    """
    n_comp_osc × window_length × n_components を総当たり探索する。

    OSCのfit（補正パラメータ計算）はtrainデータ全体を使うため、
    GroupKFoldの各foldで再計算する必要がある点に注意。

    具体的には:
        各fold内のtrainデータでfit_osc → そのfoldのtrain/validに適用
        → PLSを学習 → validで評価

    Parameters
    ----------
    train_raw           : np.ndarray  生スペクトル
    y                   : np.ndarray  含水率
    groups              : np.ndarray  樹種番号
    spec_cols           : list[str]   波数列名（表示用）
    window_candidates   : list[int]   試すwindow_lengthのリスト
    n_comp_candidates   : list[int]   試すPLSのn_componentsのリスト
    n_comp_osc_candidates: list[int]  試すOSCのn_componentsのリスト
    n_splits            : int         GroupKFoldのfold数
    scaler              : str         "snv" または "msc"
    ref_spectrum        : np.ndarray  MSC用の基準スペクトル
    deriv               : int         微分の次数（1または2）

    Returns
    -------
    best_params : dict  {"window_length", "n_components", "n_components_osc"}
    search_log  : list[dict]
    """
    if window_candidates is None:
        window_candidates = [5, 7, 11, 15, 21]
    if n_comp_candidates is None:
        n_comp_candidates = [2, 5, 10, 15, 20, 25, 30]
    if n_comp_osc_candidates is None:
        n_comp_osc_candidates = [1, 2, 3, 4, 5]

    # deriv=0（微分なし）のときwindowループは不要
    window_loop = [None] if deriv == 0 else window_candidates

    total = (len(window_loop) * len(n_comp_osc_candidates)
             * len(n_comp_candidates))
    deriv_label = "微分なし" if deriv == 0 else f"{deriv}次微分"
    print(f"\n{'='*65}")
    print(f"OSCパラメータ探索: {scaler.upper()} + {deriv_label} + OSC "
          f"(GroupKFold, n_splits={n_splits})")
    if deriv == 0:
        print(f"組み合わせ数: {len(n_comp_osc_candidates)} × "
              f"{len(n_comp_candidates)} = {total}通り（windowなし）")
        print(f"{'osc':>5}  {'n_comp':>8}  {'CV-RMSE':>10}")
    else:
        print(f"組み合わせ数: {len(window_loop)} × "
              f"{len(n_comp_osc_candidates)} × "
              f"{len(n_comp_candidates)} = {total}通り")
        print(f"{'window':>8}  {'osc':>5}  {'n_comp':>8}  {'CV-RMSE':>10}")
    print(f"{'-'*35}")

    search_log  = []
    best_rmse   = np.inf
    best_params = {
        "window_length":    window_loop[0],
        "n_components_osc": n_comp_osc_candidates[0],
        "n_components":     n_comp_candidates[0],
    }

    gkf = GroupKFold(n_splits=n_splits)

    for window in window_loop:
        # スケーリング（SNV or MSC）を適用
        if scaler == "snv":
            X_scaled = apply_snv(train_raw)
        else:
            X_scaled = apply_msc(train_raw, ref_spectrum)

        # 微分を適用（deriv=0のときはスキップ）
        X_diff = apply_savgol(X_scaled, window_length=window, deriv=deriv) \
            if deriv > 0 else X_scaled

        for n_osc in n_comp_osc_candidates:
            # OSCはfold内のtrainで毎回fit → valid/trainに適用
            fold_rmse_list = []
            for train_idx, valid_idx in gkf.split(X_diff, y, groups=groups):
                X_tr, X_val = X_diff[train_idx], X_diff[valid_idx]
                y_tr, y_val = y[train_idx], y[valid_idx]

                # fold内trainでOSCをfit
                osc_params = fit_osc(X_tr, y_tr, n_components=n_osc)

                # fold内train/validにOSCを適用
                X_tr_osc  = apply_osc(X_tr,  osc_params)
                X_val_osc = apply_osc(X_val, osc_params)

                fold_rmse_list.append((y_tr, y_val, X_tr_osc, X_val_osc,
                                       groups[valid_idx]))

            for n_comp in n_comp_candidates:
                fold_rmses = []
                for y_tr, y_val, X_tr_osc, X_val_osc, groups_val \
                        in fold_rmse_list:
                    pls = PLSRegression(n_components=n_comp, scale=False)
                    pls.fit(X_tr_osc, y_tr)
                    y_pred = np.clip(pls.predict(X_val_osc).flatten(), 0, None)
                    fold_rmses.append(rmse(y_val, y_pred))

                cv_rmse = np.mean(fold_rmses)
                search_log.append({
                    "window_length":    window,
                    "n_components_osc": n_osc,
                    "n_components":     n_comp,
                    "cv_rmse":          cv_rmse,
                })

                marker = " ← 現時点の最良" if cv_rmse < best_rmse else ""
                if deriv == 0:
                    print(f"{n_osc:>5}  {n_comp:>8}  "
                          f"{cv_rmse:>10.4f}{marker}")
                else:
                    print(f"{window:>8}  {n_osc:>5}  {n_comp:>8}  "
                          f"{cv_rmse:>10.4f}{marker}")

                if cv_rmse < best_rmse:
                    best_rmse = cv_rmse
                    best_params = {
                        "window_length":    window,
                        "n_components_osc": n_osc,
                        "n_components":     n_comp,
                    }

    print(f"\n最良: window={best_params['window_length']}, "
          f"osc={best_params['n_components_osc']}, "
          f"n_comp={best_params['n_components']}  "
          f"(CV-RMSE: {best_rmse:.4f})")
    return best_params, search_log


# ============================================================
# メイン実行
# ============================================================

if __name__ == "__main__":
    # --- データ読み込み ---
    train, test, sample_sub, spec_cols = load_data()

    X_raw  = train[spec_cols].values
    y      = train["moisture_content"].values
    groups = train["species_number"].values

    # MSC用の基準スペクトルをtrainから計算
    ref_spectrum = fit_msc(X_raw)

    # 探索候補
    N_COMP_COARSE    = [2, 5, 8, 10, 15, 20, 25, 30]
    N_OSC_COARSE     = [1, 2, 3, 4, 5]

    results_summary = []

    # ============================================================
    # run_017: MSCのみ（微分なし・OSCなし）
    # ============================================================
    print(f"\n{'#'*65}")
    print("# run_017: MSC のみ（微分なし・OSCなし）")
    print(f"{'#'*65}")
    best_1, _ = search_best_params(
        X_raw, y, groups, spec_cols,
        n_comp_candidates=N_COMP_COARSE,
        n_splits=5, scaler="msc",
        ref_spectrum=ref_spectrum, deriv=0
    )
    n_cands = make_fine_candidates(best_1["n_components"], margin=4, min_val=2)
    best_2, _ = search_best_params(
        X_raw, y, groups, spec_cols,
        n_comp_candidates=n_cands,
        n_splits=5, scaler="msc",
        ref_spectrum=ref_spectrum, deriv=0
    )
    X_msc = apply_msc(X_raw, ref_spectrum)
    cv_rmse, fold_results = cross_validate_pls(
        X_msc, y, groups, n_components=best_2["n_components"], n_splits=5
    )
    print(f"\nrun_017 最終結果: MSCのみ, n_comp={best_2['n_components']}, CV-RMSE={cv_rmse:.4f}")
    print_cv_detail(fold_results, train)
    results_summary.append({
        "run_id": "017", "scaler": "msc", "deriv": 0,
        "use_osc": False, "use_log_y": False,
        "window_length": None, "n_components_osc": None,
        "n_components": best_2["n_components"], "cv_rmse": cv_rmse,
    })

    # ============================================================
    # run_018・019: 微分なし + OSC
    # ============================================================
    for run_id, scaler in [("018", "snv"), ("019", "msc")]:
        ref = ref_spectrum if scaler == "msc" else None
        print(f"\n{'#'*65}")
        print(f"# run_{run_id}: {scaler.upper()} + OSC（微分なし）")
        print(f"{'#'*65}")

        # 第1段階
        print("\n【第1段階】粗い探索")
        best_1, _ = search_best_params_with_osc(
            X_raw, y, groups, spec_cols,
            n_comp_candidates=N_COMP_COARSE,
            n_comp_osc_candidates=N_OSC_COARSE,
            n_splits=5, scaler=scaler,
            ref_spectrum=ref, deriv=0
        )

        # 第2段階
        print("\n【第2段階】細かい探索")
        osc_cands = make_fine_candidates(best_1["n_components_osc"],
                                         margin=2, min_val=1)
        n_cands   = make_fine_candidates(best_1["n_components"],
                                         margin=4, min_val=2)
        print(f"  osc候補:    {osc_cands}")
        print(f"  n_comp候補: {n_cands}")
        best_2, _ = search_best_params_with_osc(
            X_raw, y, groups, spec_cols,
            n_comp_candidates=n_cands,
            n_comp_osc_candidates=osc_cands,
            n_splits=5, scaler=scaler,
            ref_spectrum=ref, deriv=0
        )

        best_n_osc = best_2["n_components_osc"]
        best_n     = best_2["n_components"]

        # 詳細CV
        if scaler == "snv":
            X_scaled = apply_snv(X_raw)
        else:
            X_scaled = apply_msc(X_raw, ref_spectrum)

        gkf = GroupKFold(n_splits=5)
        fold_results = []
        for fold_idx, (tr_idx, val_idx) in enumerate(
                gkf.split(X_scaled, y, groups=groups)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            groups_val  = groups[val_idx]

            osc_params_fold = fit_osc(X_tr, y_tr, n_components=best_n_osc)
            X_tr_osc  = apply_osc(X_tr,  osc_params_fold)
            X_val_osc = apply_osc(X_val, osc_params_fold)

            pls = PLSRegression(n_components=best_n, scale=False)
            pls.fit(X_tr_osc, y_tr)
            y_pred = np.clip(pls.predict(X_val_osc).flatten(), 0, None)

            fold_rmse    = rmse(y_val, y_pred)
            species_rmse = {}
            for sp in np.unique(groups_val):
                mask = groups_val == sp
                species_rmse[int(sp)] = rmse(y_val[mask], y_pred[mask])
            fold_results.append({
                "fold": fold_idx + 1, "rmse": fold_rmse,
                "n_valid": len(y_val), "species_rmse": species_rmse,
            })

        cv_rmse = np.mean([r["rmse"] for r in fold_results])
        print(f"\nrun_{run_id}: {scaler.upper()} + OSC, "
              f"osc={best_n_osc}, n_comp={best_n}, CV-RMSE={cv_rmse:.4f}")
        print_cv_detail(fold_results, train)

        results_summary.append({
            "run_id": run_id, "scaler": scaler, "deriv": 0,
            "use_osc": True, "use_log_y": False,
            "window_length": None, "n_components_osc": best_n_osc,
            "n_components": best_n, "cv_rmse": cv_rmse,
        })

    # --- 全実験の比較 ---
    CURRENT_BEST_RMSE = 24.5375
    CURRENT_BEST_RUN  = "014"

    print(f"\n{'='*75}")
    print("全実験の比較（既存結果も含む）")
    print(f"{'='*75}")
    print(f"{'run_id':>8}  {'scaler':>6}  {'deriv':>6}  {'osc':>5}  "
          f"{'window':>8}  {'n_osc':>6}  {'n_comp':>8}  {'CV-RMSE':>10}")
    print(f"{'-'*70}")
    print(f"{'014':>8}  {'snv':>6}  {'2':>6}  {'yes':>5}  "
          f"{'13':>8}  {'2':>6}  {'23':>8}  "
          f"{CURRENT_BEST_RMSE:>10.4f}  <- 既存ベスト")
    for r in results_summary:
        w_str   = str(r["window_length"])   if r["window_length"]   else "-"
        osc_str = str(r["n_components_osc"]) if r["n_components_osc"] else "-"
        osc_label = "yes" if r["use_osc"] else "no"
        print(f"{r['run_id']:>8}  {r['scaler']:>6}  {r['deriv']:>6}  "
              f"{osc_label:>5}  {w_str:>8}  {osc_str:>6}  "
              f"{r['n_components']:>8}  {r['cv_rmse']:>10.4f}")

    # --- 最良の設定でモデルを保存 ---
    best_run = min(results_summary, key=lambda x: x["cv_rmse"])
    if best_run["cv_rmse"] < CURRENT_BEST_RMSE:
        print(f"\n-> run_{best_run['run_id']}が最良 "
              f"(CV-RMSE={best_run['cv_rmse']:.4f})")

        if best_run["scaler"] == "snv":
            X_scaled = apply_snv(X_raw)
        else:
            X_scaled = apply_msc(X_raw, ref_spectrum)

        X_diff = apply_savgol(X_scaled,
                              window_length=best_run["window_length"],
                              deriv=best_run["deriv"])             if best_run["deriv"] > 0 else X_scaled

        if best_run["use_osc"]:
            osc_params = fit_osc(X_diff, y,
                                 n_components=best_run["n_components_osc"])
            X_save = apply_osc(X_diff, osc_params)
            joblib.dump(osc_params, "models/osc_params.pkl")
            print("[main] OSCパラメータを保存: models/osc_params.pkl")
        else:
            X_save = X_diff

        train_final_model(
            X_save, y, best_run["n_components"],
            save_path="models/pls_model.pkl"
        )
        if best_run["scaler"] == "msc":
            joblib.dump(ref_spectrum, "models/ref_spectrum.pkl")
            print("[main] MSC基準スペクトルを保存: models/ref_spectrum.pkl")
        joblib.dump(best_run, "models/best_params.pkl")
        print(f"[main] 最良設定を保存: models/best_params.pkl")
    else:
        print(f"\n-> run_{CURRENT_BEST_RUN}が依然として最良 "
              f"(CV-RMSE={CURRENT_BEST_RMSE})")
        print("  モデルの更新は不要です")

    print("\n学習モジュール: 正常終了")