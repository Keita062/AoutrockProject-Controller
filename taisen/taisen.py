"""
プレイヤー特定モデル (Player Identification Model)

格闘ゲームの対戦ログ（コントローラー操作CSV）から、プレイヤー（人物）を特定するモデル。
各CSVファイルのファイル名（`player1 player2.csv`）がcon1/con2の対応関係を示す。

依存ライブラリ:
    - pandas: CSVデータ処理
    - numpy: 数値計算
    - scikit-learn: 機械学習モデル
    - opencv-python: 動画分析（オプション）
    - pathlib: ファイルパス操作

使用方法:
    model = PlayerIdentifier(data_dirs=[...])
    model.fit()
    player_name = model.predict(csv_path, username="con1")
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# オプション: OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    warnings.warn("OpenCV not available. VideoAnalyzer will not work.", ImportWarning)


# ============================================================
# 定数
# ============================================================

# 34列のボタン/方向カラム（2026_1_15以降の完全形式）
BUTTON_COLUMNS = [
    "X", "Y", "B", "A", "RB", "LB", "RT", "LT", "RStick", "LStick",
    "SELECT", "START",
    "CenterArrow", "UpArrow", "DownArrow", "LeftArrow", "RightArrow",
    "UpRightArrow", "UpLeftArrow", "DownRightArrow", "DownLeftArrow",
    "Center", "Up", "Down", "Right", "Left",
    "UpRight", "UpLeft", "DownRight", "DownLeft",
    "StateX", "StateY"
]

# 方向カラム（Stateを除く）
DIRECTION_COLUMNS = [
    "CenterArrow", "UpArrow", "DownArrow", "LeftArrow", "RightArrow",
    "UpRightArrow", "UpLeftArrow", "DownRightArrow", "DownLeftArrow",
    "Center", "Up", "Down", "Right", "Left",
    "UpRight", "UpLeft", "DownRight", "DownLeft",
]

# 除外するCSVファイル名パターン
EXCLUDE_FILES = {"shouryuuken.csv", "zikken.csv", "experiment.csv"}

# スライディングウィンドウサイズ（行数）
WINDOW_SIZE = 50

# ビデオフレームレート（判定用）
VIDEO_FPS = 60

# イベント定義（試合開始 = ファイル先頭）
EVENT_DEFINITION = "file_start"


# ============================================================
# DataLoader クラス
# ============================================================

class DataLoader:
    """複数のディレクトリからCSVを読み込み、統一されたDataFrameを作成する。"""

    def __init__(self, data_dirs: List[str]):
        """
        Args:
            data_dirs: 対象ディレクトリのパス一覧
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.df = None
        self.file_info = []  # 各ファイルの情報を記録
        self.file_indices = []  # 各行がどのファイルから来たかを記録
        self.pair_info = {}  # 対戦ペア情報（ペア -> ファイルリスト）
        self.video_mapping_success = {}  # ビデオマッピング成功状況

    def load(self) -> pd.DataFrame:
        """
        すべてのCSVファイルを読み込み、統一されたDataFrameを作成する。

        Returns:
            統一されたDataFrame（username, Timestamp, ボタン列, player_name列を含む）
        """
        dfs = []
        file_idx = 0

        for data_dir in self.data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                print(f"Warning: {data_dir} does not exist")
                continue

            for csv_file in sorted(data_dir.glob("*.csv")):
                if csv_file.name in EXCLUDE_FILES:
                    continue

                # ファイル名からプレイヤー名を解析
                player1, player2 = self._parse_filename(csv_file.name)
                if player1 is None or player2 is None:
                    print(f"Skipping {csv_file.name} (name parse failed)")
                    continue

                df = self._load_single_csv(csv_file, player1, player2, file_idx)
                if df is not None and len(df) > 0:
                    dfs.append(df)
                    print(f"[File {file_idx+1}] Loaded {csv_file.name}: {len(df)} rows")
                    file_idx += 1

        if not dfs:
            raise ValueError("No CSV files loaded")

        self.df = pd.concat(dfs, ignore_index=True)

        # Timestamp を datetime に変換（複数形式に対応）
        self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"], format='mixed')

        print(f"\nTotal loaded: {len(self.df)} rows from {len(self.file_info)} files")
        print(f"Players: {self.df['player_name'].unique()}")

        # 対戦ペア情報を構築
        self._build_pair_info()
        return self.df

    def _build_pair_info(self):
        """
        ファイル情報から対戦ペア（match pair）情報を構築。
        複数ファイルを持つペアと1ファイルのみのペアを分類。
        """
        from collections import defaultdict

        pairs = defaultdict(list)

        for file_idx, info in enumerate(self.file_info):
            player1 = info["player1"]
            player2 = info["player2"]

            # ペアを正規化（アルファベット順）
            pair_key = tuple(sorted([player1, player2]))
            pairs[pair_key].append(file_idx)

        self.pair_info = pairs

        # ペア情報をログ出力
        print("\n対戦ペア分析:")
        print("=" * 80)
        multi_file_pairs = {k: v for k, v in pairs.items() if len(v) > 1}
        single_file_pairs = {k: v for k, v in pairs.items() if len(v) == 1}

        print(f"複数ファイルペア: {len(multi_file_pairs)}")
        for pair, file_indices in sorted(multi_file_pairs.items()):
            print(f"  {pair[0]} vs {pair[1]}: {len(file_indices)}ファイル (indices: {file_indices})")

        print(f"\n1ファイルのみペア（訓練に含める）: {len(single_file_pairs)}")
        for pair, file_indices in sorted(single_file_pairs.items()):
            print(f"  {pair[0]} vs {pair[1]}: indices {file_indices}")

    def _parse_filename(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """
        ファイル名 `player1 player2.csv` からプレイヤー名を抽出。

        Returns:
            (player1, player2) or (None, None)
        """
        name = filename.replace(".csv", "").strip()

        # スペース区切りの場合
        if " " in name:
            parts = name.split()
            if len(parts) >= 2:
                # プレイヤー名を正規化
                p1 = self._normalize_player_name(parts[0])
                p2 = self._normalize_player_name(parts[1])
                return p1, p2

        return None, None

    def _normalize_player_name(self, name: str) -> str:
        """
        プレイヤー名を正規化（ベース名に統一）。
        例: keita1, keita2 → keita; usigome, usigome2 → usigome
        例: Jin, JIN → jin; Kotaro, kotaro → kotaro
        """
        # 数字のサフィックスを削除
        base_name = re.sub(r'\d+$', '', name)
        # 小文字に統一
        base_name = base_name.lower()
        return base_name if base_name else name

    def _load_single_csv(
        self, csv_file: Path, player1: str, player2: str, file_idx: int
    ) -> Optional[pd.DataFrame]:
        """
        単一のCSVファイルを読み込み、スキーマを統一する。

        Args:
            csv_file: ファイルパス
            player1: con1に対応するプレイヤー名
            player2: con2に対応するプレイヤー名
            file_idx: ファイルインデックス

        Returns:
            統一されたDataFrame or None
        """
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            return None

        # スキーマ統一：RT/LT が無い場合は追加
        for col in ["RT", "LT"]:
            if col not in df.columns:
                df[col] = 0

        # 必要なカラムのみを抽出
        required_cols = ["username", "Timestamp"] + BUTTON_COLUMNS
        available_cols = [c for c in required_cols if c in df.columns]

        if "username" not in df.columns or "Timestamp" not in df.columns:
            print(f"Skipping {csv_file.name} (missing username or Timestamp)")
            return None

        df = df[available_cols]

        # player_name 列を追加（username に応じて）
        df["player_name"] = df["username"].apply(
            lambda x: player1 if x == "con1" else (player2 if x == "con2" else None)
        )

        # player_name が解決できなかった行は削除
        df = df[df["player_name"].notna()]

        if len(df) == 0:
            return None

        # ファイルインデックスを列として追加（train/test分割用）
        df["file_idx"] = file_idx

        # ファイル情報を記録（デバッグ用）
        self.file_info.append({
            "file_idx": file_idx,
            "file": csv_file.name,
            "player1": player1,
            "player2": player2,
            "rows": len(df),
        })

        return df


# ============================================================
# FeatureExtractor クラス
# ============================================================

class FeatureExtractor:
    """スライディングウィンドウで特徴量を抽出する。"""

    def __init__(self, window_size: int = WINDOW_SIZE):
        """
        Args:
            window_size: ウィンドウサイズ（行数）
        """
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.feature_names = []

    def extract_features(
        self, df: pd.DataFrame, video_analyzer: Optional["VideoAnalyzer"] = None, loader = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        DataFrameをウィンドウに分割し、各ウィンドウから特徴量を抽出。

        Args:
            df: ロード済みのDataFrame
            video_analyzer: 動画解析器（オプション）
            loader: DataLoader インスタンス（CSV情報取得用）

        Returns:
            (X, y, feature_names, file_indices)
            - X: 特徴量行列 (n_samples, n_features)
            - y: 教師ラベル（プレイヤー名）
            - feature_names: 特徴名一覧
            - file_indices: 各ウィンドウが属するファイルインデックス
        """
        features_list = []
        labels_list = []
        file_indices_list = []

        # ファイル/プレイヤーごとにウィンドウを作成
        for player_name in df["player_name"].unique():
            player_df = df[df["player_name"] == player_name].reset_index(drop=True)
            features = self._extract_player_features(
                player_df, player_name, video_analyzer, loader
            )
            features_list.extend(features["X"])
            labels_list.extend(features["y"])
            file_indices_list.extend(features["file_idx"])

        X = np.array(features_list)
        y = np.array(labels_list)
        file_indices = np.array(file_indices_list)

        self.feature_names = self._get_feature_names()

        print(f"Extracted {len(X)} windows from {len(df)} rows")
        print(f"Feature dimension: {X.shape[1]}")

        return X, y, self.feature_names, file_indices

    def _extract_player_features(
        self, df: pd.DataFrame, player_name: str, video_analyzer = None, loader = None
    ) -> Dict[str, List]:
        """
        単一プレイヤーのデータからウィンドウごとに特徴量を抽出。
        """
        features_X = []
        features_y = []
        features_file_idx = []

        # スライディングウィンドウ
        for start_idx in range(0, len(df) - self.window_size + 1, self.window_size // 2):
            end_idx = start_idx + self.window_size
            window = df.iloc[start_idx:end_idx]

            feat = self._compute_window_features(
                window, video_analyzer=video_analyzer, loader=loader
            )
            features_X.append(feat)
            features_y.append(player_name)
            # ウィンドウのファイルインデックス（最初の行から取得）
            features_file_idx.append(window["file_idx"].iloc[0])

        return {"X": features_X, "y": features_y, "file_idx": features_file_idx}

    def _compute_window_features(self, window: pd.DataFrame, video_analyzer = None, loader = None) -> np.ndarray:
        """
        ウィンドウから特徴量ベクトルを計算。
        """
        features = []

        # 1. 時間差系（delta_t）
        timestamps = pd.to_datetime(window["Timestamp"])
        if len(timestamps) > 1:
            deltas = timestamps.diff().dt.total_seconds().dropna().values
            features.extend([
                np.mean(deltas) if len(deltas) > 0 else 0,
                np.std(deltas) if len(deltas) > 1 else 0,
                np.max(deltas) if len(deltas) > 0 else 0,
                np.percentile(deltas, 25) if len(deltas) > 0 else 0,
                np.percentile(deltas, 75) if len(deltas) > 0 else 0,
            ])
        else:
            features.extend([0, 0, 0, 0, 0])

        # 2. ボタン頻度と連続性（各ボタンの合計＋連続押下パターン）
        # SELECT, START を除外（ゲーム中に不使用）
        button_cols = [c for c in BUTTON_COLUMNS if c not in ["StateX", "StateY", "SELECT", "START"] and c in window.columns]
        for col in button_cols:
            count = window[col].sum()
            features.append(count)

            # 連続性を計算：同じボタンが連続で1になる長さの平均
            if count > 0:
                presses = window[col].values
                run_lengths = []
                current_length = 0
                for val in presses:
                    if val == 1:
                        current_length += 1
                    elif current_length > 0:
                        run_lengths.append(current_length)
                        current_length = 0
                if current_length > 0:
                    run_lengths.append(current_length)
                features.append(np.mean(run_lengths) if run_lengths else 0)
            else:
                features.append(0)

        # 3. スティック移動速度（連続フレーム間の距離）のみを使用
        # StateX_mean, StateY_mean, StateX_std, StateY_std は定常的な特徴量のため削除
        if "StateX" in window.columns and "StateY" in window.columns:
            state_x = window["StateX"].values
            state_y = window["StateY"].values

            # スティック移動速度（連続フレーム間の距離）
            if len(state_x) > 1 and len(state_y) > 1:
                distances = np.sqrt(np.diff(state_x)**2 + np.diff(state_y)**2)
                features.extend([
                    np.mean(distances) if len(distances) > 0 else 0,
                    np.std(distances) if len(distances) > 1 else 0,
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])

        # 4. アイドル時間（何もボタンを押さない期間の統計）
        # idle_max は定常的なため削除。idle_mean のみ使用
        # SELECT, START を除外（ゲーム中に不使用）
        all_button_cols = [c for c in BUTTON_COLUMNS if c not in ["StateX", "StateY", "SELECT", "START"] and c in window.columns]
        any_button = window[all_button_cols].sum(axis=1) > 0
        idle_periods = []
        idle_count = 0
        for pressed in any_button.values:
            if not pressed:
                idle_count += 1
            elif idle_count > 0:
                idle_periods.append(idle_count)
                idle_count = 0
        if idle_count > 0:
            idle_periods.append(idle_count)

        features.append(np.mean(idle_periods) if idle_periods else 0)

        # 5. イベント（試合開始）からの経過時間
        if EVENT_DEFINITION == "file_start":
            time_since_event = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
            features.append(time_since_event)
        else:
            features.append(0)

        # facing_direction は削除（ビデオマッピング失敗ファイルが多い）

        return np.array(features)

    def _get_feature_names(self) -> List[str]:
        """特徴量の名前一覧を返す。"""
        names = [
            "delta_t_mean", "delta_t_std", "delta_t_max", "delta_t_q25", "delta_t_q75",
        ]

        # 除外するボタン：SELECT, START（ゲーム中に不使用）
        button_cols = [c for c in BUTTON_COLUMNS if c not in ["StateX", "StateY", "SELECT", "START"]]
        for col in button_cols:
            names.append(f"{col}_count")
            names.append(f"{col}_run_length_avg")

        names.extend([
            # StateX_mean, StateY_mean を削除（定常的な特徴量）
            # StateX_std, StateY_std を削除（定常的な特徴量）
            "stick_distance_mean", "stick_distance_std",
            "idle_mean",  # idle_max を削除（定常的な特徴量）
            "time_since_event",
        ])

        return names


# ============================================================
# VideoAnalyzer クラス
# ============================================================

class VideoAnalyzer:
    """
    動画ファイルから前向き/後ろ向きを推定する。
    シンプルな実装：フレームの左右の明るさ差分を使う。
    CSVと動画ファイルの自動マッピング機能付き。
    """

    def __init__(self, data_dirs: List[str] = None):
        if not HAS_OPENCV:
            print("Warning: OpenCV not available. VideoAnalyzer disabled.")
        self.data_dirs = [Path(d) for d in (data_dirs or [])]
        self.csv_video_mapping = {}  # {csv_filename: (video_path, time_offset_seconds)}

    def get_facing_direction(
        self, video_path: str, timestamp_sec: float
    ) -> str:
        """
        指定されたタイムスタンプでのキャラクター向きを推定。

        Args:
            video_path: MP4ファイルパス
            timestamp_sec: 秒単位のタイムスタンプ

        Returns:
            "right" | "left" | "unknown"
        """
        if not HAS_OPENCV:
            return "unknown"

        if not os.path.exists(video_path):
            return "unknown"

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(timestamp_sec * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return "unknown"

            # フレームを灰色に変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 左右半分に分割
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:]

            left_brightness = np.mean(left_half)
            right_brightness = np.mean(right_half)

            # シンプルなヒューリスティック：
            # 左側がより明るい → キャラクターは右向き（1P側）
            # 右側がより明るい → キャラクターは左向き（2P側）
            if abs(left_brightness - right_brightness) > 10:
                return "right" if left_brightness > right_brightness else "left"
            else:
                return "unknown"

        except Exception as e:
            print(f"VideoAnalyzer error: {e}")
            return "unknown"

    def create_csv_video_mapping(self, loader) -> Dict[str, tuple]:
        """
        CSVファイルと動画ファイルを日付・時刻で自動マッピング。

        Args:
            loader: DataLoader インスタンス（file_info を含む）

        Returns:
            {csv_filename: (video_path, time_offset_seconds), ...}

        また、loader.video_mapping_success を更新：
            {csv_filename: True/False}
        """
        from collections import defaultdict

        # データディレクトリから動画ファイルをスキャン
        video_files_by_date = defaultdict(list)

        for data_dir in self.data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                continue

            for video_file in data_dir.rglob("*.mp4"):
                # ファイル名から日付を抽出: YYYY-MM-DD形式
                match = re.search(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2})-(\d{2})-(\d{2})', video_file.name)
                if match:
                    date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                    time_str = f"{match.group(4)}:{match.group(5)}:{match.group(6)}"
                    video_files_by_date[date_str].append((video_file, time_str))

        # CSVファイルのタイムスタンプから動画を探す
        mapping = {}
        print("\nCSV-動画マッピング:")
        print("=" * 100)

        for file_idx, file_info in enumerate(loader.file_info):
            csv_filename = file_info["file"]
            csv_path = None

            # ファイルを見つける
            for data_dir in self.data_dirs:
                data_dir = Path(data_dir)
                potential_csv = data_dir / csv_filename
                if potential_csv.exists():
                    csv_path = potential_csv
                    break

            if csv_path is None:
                continue

            try:
                # CSVの最初のタイムスタンプを取得
                df = pd.read_csv(csv_path, nrows=1)
                if 'Timestamp' not in df.columns:
                    continue

                first_ts = pd.to_datetime(df['Timestamp'].iloc[0])
                csv_date = first_ts.strftime('%Y-%m-%d')
                csv_time = first_ts.strftime('%H:%M:%S')

                # その日付の動画ファイルを探す
                if csv_date in video_files_by_date:
                    videos_on_date = video_files_by_date[csv_date]

                    # 時刻が最も近い動画を選択
                    best_video = None
                    best_offset = float('inf')

                    for video_path, video_time in videos_on_date:
                        # 時刻の差分を計算
                        csv_h, csv_m, csv_s = map(int, csv_time.split(':'))
                        vid_h, vid_m, vid_s = map(int, video_time.split(':'))

                        csv_seconds = csv_h * 3600 + csv_m * 60 + csv_s
                        vid_seconds = vid_h * 3600 + vid_m * 60 + vid_s
                        offset = abs(csv_seconds - vid_seconds)

                        if offset < best_offset:
                            best_offset = offset
                            best_video = (video_path, vid_seconds - csv_seconds)

                    if best_video and best_offset < 300:  # 5分以内なら対応と判定
                        mapping[csv_filename] = best_video
                        print(f"OK {csv_filename:<40} → {best_video[0].name:<50} (offset: {best_video[1]:+.0f}秒)")
                    else:
                        print(f"NG {csv_filename:<40} → 対応動画なし（日付: {csv_date}）")
                else:
                    print(f"NG {csv_filename:<40} → 対応動画なし（日付: {csv_date}）")

            except Exception as e:
                print(f"ER {csv_filename:<40} → エラー: {e}")

        print("=" * 100)
        self.csv_video_mapping = mapping

        # loader.video_mapping_success を更新
        for file_idx, file_info in enumerate(loader.file_info):
            csv_filename = file_info["file"]
            loader.video_mapping_success[csv_filename] = (csv_filename in mapping)

        return mapping

    def get_facing_direction_from_csv(self, csv_filename: str, timestamp: str) -> str:
        """
        CSVのタイムスタンプに対応する動画フレームから前向き/後ろ向きを判定。

        Args:
            csv_filename: CSVファイル名
            timestamp: タイムスタンプ（ISO形式）

        Returns:
            "right" | "left" | "unknown"
        """
        if csv_filename not in self.csv_video_mapping:
            return "unknown"

        video_path, time_offset = self.csv_video_mapping[csv_filename]

        try:
            # CSVのタイムスタンプをビデオ内の秒数に変換
            ts = pd.to_datetime(timestamp)
            # ビデオの最初のフレームを基準に計算
            # （time_offset は CSV開始時刻とビデオ開始時刻の差）
            video_seconds = (ts - ts).total_seconds() - time_offset  # 0秒から始まる相対時間

            if video_seconds < 0:
                return "unknown"

            return self.get_facing_direction(str(video_path), video_seconds)

        except Exception as e:
            return "unknown"


# ============================================================
# PlayerIdentifier クラス
# ============================================================

class PlayerIdentifier:
    """プレイヤー特定モデル。"""

    def __init__(self, data_dirs: List[str], window_size: int = WINDOW_SIZE):
        """
        Args:
            data_dirs: データディレクトリ一覧
            window_size: 特徴量抽出のウィンドウサイズ
        """
        self.data_dirs = data_dirs
        self.window_size = window_size
        self.loader = DataLoader(data_dirs)
        self.extractor = FeatureExtractor(window_size)
        self.analyzer = VideoAnalyzer(data_dirs)  # ビデオアナライザーにデータディレクトリを渡す
        self.model = None
        self.scaler = StandardScaler()
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def _get_pair_based_split(self, pair_info: Dict = None) -> tuple:
        """
        対戦ペア単位での訓練・テスト分割を自動生成。

        訓練用: 複数ファイルペアの最初のN-1ファイル + 1ファイルのみのペア
        テスト用: 複数ファイルペアの最後の1ファイル

        Args:
            pair_info: 対戦ペア情報（Noneの場合は self.loader.pair_info を使用）

        Returns:
            (train_file_indices, test_file_indices)
        """
        from collections import defaultdict

        if pair_info is None:
            pair_info = self.loader.pair_info

        train_indices = []
        test_indices = []

        for pair, file_indices in pair_info.items():
            if len(file_indices) > 1:
                # 複数ファイルペア: 最初のN-1を訓練、最後の1をテスト
                train_indices.extend(file_indices[:-1])
                test_indices.append(file_indices[-1])
            else:
                # 1ファイルのみ: 訓練に含める
                train_indices.extend(file_indices)

        return sorted(train_indices), sorted(test_indices)

    def fit(self, train_file_indices: Optional[List[int]] = None,
            test_file_indices: Optional[List[int]] = None,
            save_data_csv: bool = True,
            auto_pair_split: bool = True) -> Dict:
        """
        モデルを学習する（対戦ペア単位での分割）。

        Args:
            train_file_indices: 訓練用ファイルのインデックス（Noneの場合は自動）
            test_file_indices: テスト用ファイルのインデックス（Noneの場合は自動）
            save_data_csv: 訓練/テストデータをCSVとして保存するか（デフォルト True）
            auto_pair_split: 対戦ペア単位での自動分割（デフォルト True）

        Returns:
            {accuracy, report, confusion_matrix}
        """
        print("Loading data...")
        self.df = self.loader.load()
        original_row_count = len(self.df)

        # CSV-動画マッピングを作成
        print("\nCreating CSV-Video mapping...")
        self.analyzer.create_csv_video_mapping(self.loader)

        # ビデオマッピング失敗ファイルのデータを除外
        print("\nFiltering out files with failed video mapping...")
        failed_files = [fname for fname, success in self.loader.video_mapping_success.items() if not success]
        failed_file_indices = [i for i, info in enumerate(self.loader.file_info) if info["file"] in failed_files]
        successful_file_indices = set(range(len(self.loader.file_info))) - set(failed_file_indices)

        if failed_files:
            print(f"Excluding {len(failed_files)} files with failed video mapping:")
            for fname in failed_files[:5]:
                print(f"  - {fname}")
            if len(failed_files) > 5:
                print(f"  ... and {len(failed_files) - 5} more files")

            # failed_file_indices を含まないデータのみを保持
            self.df = self.df[~self.df["file_idx"].isin(failed_file_indices)].reset_index(drop=True)
            print(f"Data reduced from {original_row_count} to {len(self.df)} rows")
        else:
            print(f"All {len(self.loader.file_info)} files have successful video mapping")
            successful_file_indices = set(range(len(self.loader.file_info)))

        # ファイルインデックスの自動設定
        if train_file_indices is None or test_file_indices is None:
            if auto_pair_split:
                # 対戦ペア単位での自動分割（成功ファイルのみを使用）
                # 成功ファイルのみからペア情報を再構築
                successful_list = sorted(successful_file_indices)

                # 成功ファイルのみでペア情報を再構築
                from collections import defaultdict
                rebuilt_pair_info = defaultdict(list)
                for orig_idx in successful_list:
                    info = self.loader.file_info[orig_idx]
                    player1 = info["player1"]
                    player2 = info["player2"]
                    pair_key = tuple(sorted([player1, player2]))
                    rebuilt_pair_info[pair_key].append(orig_idx)

                if rebuilt_pair_info:
                    print(f"\nRebuilt pair info from successful files: {len(rebuilt_pair_info)} pairs")
                    for pair, indices in sorted(rebuilt_pair_info.items()):
                        print(f"  {pair[0]} vs {pair[1]}: {len(indices)} files (indices: {indices})")

                    train_file_indices, test_file_indices = self._get_pair_based_split(pair_info=dict(rebuilt_pair_info))
                else:
                    print(f"Warning: No pairs found in successful files. Using default split.")
                    # フォールバック
                    split_point = max(1, len(successful_list) * 7 // 10)
                    train_file_indices = successful_list[:split_point]
                    test_file_indices = successful_list[split_point:]
            else:
                # デフォルト: ファイル 1-8 を訓練用、9+ をテスト用
                successful_list = sorted(successful_file_indices)
                split_point = min(8, len(successful_list))
                train_file_indices = successful_list[:split_point]
                test_file_indices = successful_list[split_point:]

        print(f"\nFile-based splitting:")
        print(f"  Training files (indices {train_file_indices}): {[self.loader.file_info[i]['file'] for i in train_file_indices if i < len(self.loader.file_info)]}")
        print(f"  Test files (indices {test_file_indices}): {[self.loader.file_info[i]['file'] for i in test_file_indices if i < len(self.loader.file_info)]}")

        print("\nExtracting features...")
        X, y, self.feature_names, window_file_indices = self.extractor.extract_features(
            self.df, video_analyzer=self.analyzer, loader=self.loader
        )

        # ファイル単位でのトレーニング/テスト分割
        train_mask = np.isin(window_file_indices, train_file_indices)
        test_mask = np.isin(window_file_indices, test_file_indices)

        self.X_train, self.X_test = X[train_mask], X[test_mask]
        self.y_train, self.y_test = y[train_mask], y[test_mask]

        print(f"\nFitting model...")
        print(f"  Train set size: {len(self.X_train)} samples from files {train_file_indices}")
        print(f"  Test set size: {len(self.X_test)} samples from files {test_file_indices}")

        # 標準化
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(self.X_train, self.y_train)

        # 評価
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"\n{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Train players: {np.unique(self.y_train)}")
        print(f"Test players: {np.unique(self.y_test)}")
        print(f"{'='*60}")

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)

        print("\nTop 15 Feature Importances:")
        importances = self.model.feature_importances_
        print(f"  (Feature dimension: {len(self.feature_names)}, Importances dimension: {len(importances)})")
        top_indices = np.argsort(importances)[::-1][:15]
        for idx in top_indices:
            if idx < len(self.feature_names):
                print(f"  {self.feature_names[idx]}: {importances[idx]:.4f}")

        # 訓練/テストデータをCSVとして保存
        if save_data_csv:
            self._save_training_data(train_file_indices, test_file_indices)

        return {
            "accuracy": accuracy,
            "y_pred": y_pred,
            "confusion_matrix": cm,
        }

    def _save_training_data(self, train_file_indices: List[int], test_file_indices: List[int]):
        """
        訓練/テストデータを CSV として保存。

        Args:
            train_file_indices: 訓練ファイルのインデックス
            test_file_indices: テストファイルのインデックス
        """
        # model_data フォルダを作成
        model_data_dir = Path(r"c:\GitHub\AoutrockProject-Controller\taisen\model_data")
        model_data_dir.mkdir(parents=True, exist_ok=True)

        # 訓練データを DataFrame に変換
        train_df = pd.DataFrame(
            self.X_train,
            columns=self.feature_names
        )
        train_df["player_name"] = self.y_train
        train_df["train_file_indices"] = str(train_file_indices)
        train_df["data_type"] = "train"

        # テストデータを DataFrame に変換
        test_df = pd.DataFrame(
            self.X_test,
            columns=self.feature_names
        )
        test_df["player_name"] = self.y_test
        test_df["test_file_indices"] = str(test_file_indices)
        test_df["data_type"] = "test"

        # CSV として保存
        train_csv_path = model_data_dir / "train_data.csv"
        test_csv_path = model_data_dir / "test_data.csv"

        train_df.to_csv(train_csv_path, index=False, encoding="utf-8")
        test_df.to_csv(test_csv_path, index=False, encoding="utf-8")

        print(f"\n[SAVED] Training data: {train_csv_path}")
        print(f"  Shape: {train_df.shape} (rows: {len(train_df)}, cols: {len(train_df.columns)})")
        print(f"  Players: {sorted(train_df['player_name'].unique())}")
        print(f"  File indices: {train_file_indices}")

        print(f"\n[SAVED] Test data: {test_csv_path}")
        print(f"  Shape: {test_df.shape} (rows: {len(test_df)}, cols: {len(test_df.columns)})")
        print(f"  Players: {sorted(test_df['player_name'].unique())}")
        print(f"  File indices: {test_file_indices}")

    def predict(self, csv_file: str, username: str) -> str:
        """
        新しいCSVファイルのプレイヤーを予測する。

        Args:
            csv_file: 予測対象のCSVファイルパス
            username: "con1" or "con2"

        Returns:
            プレイヤー名
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        df = pd.read_csv(csv_file)

        # username でフィルタ
        df = df[df["username"] == username].reset_index(drop=True)

        if len(df) == 0:
            return "unknown"

        # 特徴量抽出
        feat = self.extractor._compute_window_features(df)
        feat = self.scaler.transform([feat])

        # 予測
        player = self.model.predict(feat)[0]
        probability = self.model.predict_proba(feat)[0]

        print(f"Predicted player: {player}")
        print(f"Confidence: {np.max(probability):.4f}")

        return player

    def save(self, model_path: str):
        """モデルを保存。"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, model_path.replace(".pkl", "_scaler.pkl"))
        print(f"Model saved to {model_path}")

    def load(self, model_path: str):
        """モデルを読み込み。"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(model_path.replace(".pkl", "_scaler.pkl"))
        print(f"Model loaded from {model_path}")


# ============================================================
# メイン関数（テスト用）
# ============================================================

def main():
    """テスト実行。"""
    # データディレクトリを指定（より多くのデータを使用）
    data_dirs = [
        r"c:\GitHub\AoutrockProject-Controller\2026_1_7_対戦記録",
        r"c:\GitHub\AoutrockProject-Controller\2026_1_15_対戦記録",
        r"c:\GitHub\AoutrockProject-Controller\2026_1_20_対戦記録",
    ]

    # モデルを作成・学習
    model = PlayerIdentifier(data_dirs=data_dirs, window_size=WINDOW_SIZE)

    print("=" * 60)
    print("Player Identification Model")
    print("=" * 60)

    # 対戦ペア単位での自動分割
    # 訓練用: 複数ファイルペアの最初のN-1ファイル + 1ファイルのみのペア
    # テスト用: 複数ファイルペアの最後の1ファイル
    results = model.fit(auto_pair_split=True)

    # モデルを保存
    model.save(r"c:\GitHub\AoutrockProject-Controller\taisen\player_model.pkl")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
