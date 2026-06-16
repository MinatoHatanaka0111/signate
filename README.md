# 近赤外研究会 スペクトル分析チャレンジ — 実験ガイド

## プロジェクト構成

```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submit.csv
├── models/
│   └── <run_id>/               # 実験ごとに自動生成
│       ├── model.pkl           # 学習済みモデル
│       ├── prep_params.pkl     # 前処理パラメータ（OSC/PCA）
│       ├── ref_spectrum.pkl    # MSC 用基準スペクトル
│       ├── config.json         # この実験の設定（再現用）
│       ├── search_log.json     # グリッドサーチの全結果
│       └── summary.json        # 最良パラメータと CV-RMSE
├── configs/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # 前処理（SNV/MSC/OSC/PCA/SavGol）
│   ├── learning.py        # 学習・グリッドサーチ・Optuna・結果一覧
│   └── predicting.py      # 予測・提出ファイル生成
├── pyproject.toml
└── submission.csv         # 最新の提出ファイル
```

---

## セットアップ

```bash
uv sync
```

スクリプトの実行はすべて `uv run` を使う。

---

## 基本的な実験の流れ

### 1. グリッドサーチ → 最終モデル保存

```bash
# ベースライン（PLS × SNV + 2次微分）
uv run python src/learning.py --config configs/pls_snv_deriv2.yaml

# PLS × MSC + 2次微分 + OSC
uv run python src/learning.py --config configs/pls_msc_deriv2_osc.yaml

# LightGBM の前処理スクリーニング
uv run python src/learning.py --config configs/lgbm_snv_pca.yaml
```

実行すると `models/<run_id>/` 以下に以下が生成される：

| ファイル | 内容 |
|---|---|
| `search_log.json` | 全組み合わせの CV-RMSE |
| `summary.json` | 最良設定と CV-RMSE |
| `model.pkl` | 最良設定で学習したモデル |
| `prep_params.pkl` | 前処理パラメータ（OSC/PCA） |
| `ref_spectrum.pkl` | MSC 用基準スペクトル |
| `config.json` | 設定（再現用） |

### 2. LightGBM の Optuna チューニング

グリッドサーチで前処理を絞り込んだ後、上位 N 候補に対して Optuna でハイパーパラメータを最適化する。

```bash
uv run python src/learning.py --config configs/lgbm_snv_pca.yaml \
                               --optimizer optuna --n_trials 100 --n_top 3
```

### 3. 実験結果の比較

```bash
uv run python src/learning.py --show
```

### 4. 提出ファイルの生成

```bash
# 単一モデル
uv run python src/predicting.py --run_id pls_snv_deriv2

# アンサンブル（複数モデルの平均）
uv run python src/predicting.py --run_id pls_snv_deriv2 lgbm_snv_pca --ensemble mean
```

---

## config YAML の書き方

```yaml
run_id: my_experiment    # models/<run_id>/ に結果を保存
model_dir: models
data_dir: data/

cv:
  n_splits: 5
  use_log_y: false       # true にすると含水率を log1p 変換して学習

preprocessing:           # 固定設定（search_params で上書きされる）
  scaler: snv            # "snv" | "msc" | null
  deriv: 2               # 0=なし / 1=1次微分 / 2=2次微分
  window: 13             # Savitzky-Golay の window 長
  polyorder: 2
  use_osc: false
  n_components_osc: 2    # use_osc: true のとき有効
  use_pca: false
  n_components_pca: 50   # use_pca: true のとき有効

model:
  model_type: pls        # "pls" | "lgbm" | "ridge" | "svr"
  params:
    n_components: 20

search_params:           # ここのリストが総当たりで探索される
  preprocessing:
    window: [9, 11, 13, 15]
  model:
    n_components: [10, 15, 20, 25, 30]
```

**`search_params` のポイント：**
- `preprocessing` / `model` 以下のキーにリストを書くと総当たりで展開される
- リストのないキーは固定設定として維持される
- `{}` を指定するとグリッドサーチなし（CV を 1 回だけ実行）

### Claude に config を書かせる場合

実験アイデアを伝えれば YAML を生成してもらえる。

> 「MSC + 1次微分 + OSCなし で PLS を試したい。
> window を [5, 7, 9, 11]、成分数を [10, 15, 20, 25] で探索してほしい。」

---

## 実験戦略メモ（提出回数: チーム全体で 5 回/日）

### フェーズ 1: PLS の基本探索
- [ ] `pls_snv_deriv2.yaml`
- [ ] `pls_msc_deriv2_osc.yaml`

### フェーズ 2: LightGBM の前処理スクリーニング
- [ ] `lgbm_snv_pca.yaml`

### フェーズ 3: Optuna チューニング
- [ ] `--optimizer optuna` で上位候補をチューニング

### フェーズ 4: アンサンブル
- [ ] PLS 最良 × LightGBM 最良のアンサンブルを試す

---

## 注意事項

- CV-RMSE が改善した場合のみ提出する（`--show` で比較してから判断）
- test.csv の樹種は train.csv と異なるため GroupKFold（樹種単位の分割）を必ず使う