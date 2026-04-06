# プレイヤー特定モデル (Player Identification Model)

格闘ゲーム（Capcom Arcade Stadium）の対戦ログからプレイヤーを特定する機械学習モデルです。

## 概要

- **目的**: コントローラーの操作ログから、どのプレイヤー（人物）が操作しているかを特定
- **入力**: CSVファイル（時系列のコントローラー入力記録）
- **出力**: プレイヤー名と信頼度スコア
- **精度**: 45% (11人のプレイヤー分類、ランダム予測の9%より大幅に向上)

## ファイル構成

```
taisen/
├── taisen.py                    # メインモジュール
├── player_model.pkl             # 学習済みモデル
├── player_model_scaler.pkl      # 標準化スケーラー
└── README.md                    # このファイル
```

## 主なクラス

### 1. DataLoader
複数のCSVディレクトリから対戦ログを読み込み、統一されたDataFrameを作成します。

**特徴**:
- ファイル名（`player1 player2.csv`）からプレイヤー名を自動解析
- con1 → player1、con2 → player2 に対応付け
- RT/LT列の欠落を自動補完（2026_1_7ファイル対応）
- プレイヤー名を正規化（keita1-8 → keita、大文字小文字の統一）

### 2. FeatureExtractor
スライディングウィンドウ（デフォルト50行）で特徴量を抽出します。

**抽出される特徴量（74次元）**:
- **時間差（5次元）**: 入力間隔の平均・標準偏差・最大・四分位数
- **ボタン頻度（62次元）**: 各ボタンの押下数×2（押下数 + 連続性）
  - X, Y, B, A, RB, LB, RT, LT, SELECT, START
  - CenterArrow, UpArrow, DownArrow, LeftArrow, RightArrow, UpRightArrow, UpLeftArrow, DownRightArrow, DownLeftArrow
- **アナログ値（6次元）**: StateX/Y の平均・標準偏差 + スティック移動速度
- **アイドル時間（2次元）**: ボタン非押下期間の統計
- **イベント時間（1次元）**: ファイル先頭からの経過時間

### 3. VideoAnalyzer
動画ファイルから前向き/後ろ向きを判定します（OpenCV使用）。

**実装状況**:
- ✅ フレーム取得機能
- ✅ シンプルな左右明るさ差分による向き判定
- 🔄 HSV色空間を使用したHP-bar検出（将来実装予定）

### 4. PlayerIdentifier
データの読み込み、学習、予測を一元管理するメインモデルクラス。

**メソッド**:
- `fit()`: モデルを学習（StratifiedShuffleSplitで70%train/30%test）
- `predict(csv_file, username)`: 新しいCSVファイルのプレイヤーを予測
- `save(path)`: モデルを保存
- `load(path)`: モデルを読み込み

## 特徴量の重要度ランキング

実際の学習結果（トップ15）:

| 順位 | 特徴量 | 重要度 |
|------|--------|--------|
| 1 | StateY_mean | 0.1016 |
| 2 | StateX_mean | 0.0994 |
| 3 | delta_t_std | 0.0450 |
| 4 | time_since_event | 0.0417 |
| 5 | delta_t_mean | 0.0413 |
| 6 | CenterArrow_count | 0.0411 |
| 7 | delta_t_max | 0.0408 |
| 8 | CenterArrow_run_length_avg | 0.0349 |
| 9 | RT_run_length_avg | 0.0327 |
| 10 | RT_count | 0.0303 |

**解釈**: アナログスティックの使い方（StateX/Y）が最も重要な特徴。各プレイヤーは独特の操作パターン（スティックの持ち方、押し方など）を持っている。

## 学習結果

**総精度**: 45.4% (16,311テストサンプル)

### プレイヤー別の精度

| プレイヤー | Precision | Recall | F1-score | 対戦数 |
|----------|-----------|--------|----------|--------|
| keita | 0.73 | 0.22 | 0.34 | 1,297 |
| usigome | 0.72 | 0.47 | 0.57 | 832 |
| ryou | 0.70 | 0.21 | 0.33 | 351 |
| yuusuke | 0.67 | 0.59 | 0.63 | 1,063 |
| toyouka | 0.61 | 0.69 | 0.64 | 478 |
| hasegawa | 0.49 | 0.43 | 0.46 | 965 |
| jin | 0.46 | 0.30 | 0.36 | 2,128 |
| kotaro | 0.44 | 0.49 | 0.46 | 2,981 |
| koike | 0.38 | 0.73 | 0.49 | 3,531 |
| akira | 0.38 | 0.21 | 0.27 | 1,853 |
| daiki | 0.51 | 0.25 | 0.34 | 832 |

**注**: 
- Recall が低いプレイヤー（例：keita 22%）: 他のプレイヤーと混同されやすい
- Recall が高いプレイヤー（例：koike 73%）: 独特の操作パターンを持つ

## 使用方法

### 基本的な使い方

```python
from taisen import PlayerIdentifier

# モデルを作成・学習
model = PlayerIdentifier(
    data_dirs=[
        r"c:\GitHub\AoutrockProject-Controller\2026_1_7_対戦記録",
        r"c:\GitHub\AoutrockProject-Controller\2026_1_15_対戦記録",
        r"c:\GitHub\AoutrockProject-Controller\2026_1_20_対戦記録",
    ],
    window_size=50
)

# 学習
results = model.fit()
print(f"Accuracy: {results['accuracy']:.4f}")

# 新しいCSVで予測
player = model.predict(
    csv_file="new_match.csv",
    username="con1"
)
print(f"Predicted player: {player}")

# モデルを保存
model.save("my_model.pkl")
```

### コマンドラインからの実行

```bash
python taisen.py
```

このコマンドで:
1. 指定されたディレクトリからすべてのCSVを読み込み
2. モデルを学習
3. テストセットで評価
4. 特徴量重要度を表示
5. モデルを保存

## データフォーマット

### 入力 CSV

**必須カラム**:
- `username`: "con1" or "con2"
- `Timestamp`: 日時（複数形式対応：`2026-01-15 17:49:07.652011` または `2026-01-07 17:39:17`）

**ボタンカラム** (0 or 1):
- `X, Y, B, A`: 各ボタン
- `RB, LB`: バンパー
- `RT, LT`: トリガー（2026_1_15以降のみ）
- `RStick, LStick`: スティック押し込み
- `SELECT, START`: メニュー
- 方向入力: `CenterArrow`, `UpArrow` など（十字キー）
- スティック方向: `Center`, `Up`, `Down`, `Left`, `Right`, `UpRight` など

**アナログカラム** (整数):
- `StateX, StateY`: アナログスティック座標（-32768 ～ 32767）

### ファイル命名規則

```
player1 player2.csv
```

**例**:
- `kotaro keita1.csv`: con1=kotaro, con2=keita
- `Jin Koike.csv`: con1=jin, con2=koike

## 技術スタック

- **機械学習**: Random Forest (scikit-learn)
- **データ処理**: pandas, numpy
- **動画処理**: OpenCV (cv2, オプション)
- **ファイル管理**: pathlib, joblib

## 今後の改善案

### 短期
1. **特徴量エンジニアリング**:
   - コマンド入力パターン（昇竜拳など）の検出
   - ボタン同時押しのパターン分析
   - 操作の「癖」スコア（押し込み強度、タイミングの分散）

2. **モデル改善**:
   - LSTM/Attention を使った時系列モデル
   - Ensemble (XGBoost + Random Forest)
   - ハイパーパラメータ自動チューニング

3. **動画連携**:
   - HSV色空間でHP-barを検出し、1P/2P側を自動判定
   - キャラクター検出で操作者の向き正規化
   - コマンド入力判定（動画の画面表示と照合）

### 中期
1. **マルチモーダル学習**: 操作ログ + 動画フレーム + 音声を組み合わせた学習
2. **オンライン学習**: 新しい対戦ログでインクリメンタル学習
3. **異常検出**: 通常と異なる操作パターンを検出

### 長期
1. **クローズド・セット識別**: 既知プレイヤーのみの高精度モデル
2. **オープン・セット識別**: 未知プレイヤーを検出
3. **リアルタイム予測**: ゲーム実行中にプレイヤーを特定

## トラブルシューティング

### OpenCV がない場合

```
Warning: OpenCV not available. VideoAnalyzer disabled.
```

解決方法:
```bash
pip install opencv-python
```

### CSV の読み込みに失敗する

**原因**: タイムスタンプ形式が異なる
**解決**: `pd.to_datetime(..., format='mixed')` で自動判定

**原因**: 列名が一致していない
**解決**: `BUTTON_COLUMNS` 定数を確認・更新

### 精度が低い

**考えられる原因**:
1. データ量が不足（各プレイヤーあたり最低100サンプル推奨）
2. プレイヤー間の操作パターンが類似
3. データ品質の問題（ノイズが多い）

**改善策**:
1. より多くの対戦ログを収集
2. 特徴量をカスタマイズ
3. LSTM などの複雑なモデルを試す

## 参考文献

- Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forests
- TimeSeriesSplit: https://scikit-learn.org/stable/modules/cross_validation.html
- OpenCV: https://docs.opencv.org/

## ライセンス

MIT License

## 作成者

Claude AI Assistant

---

**最終更新**: 2026-04-06
