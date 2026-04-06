"""
プレイヤー特定モデルの詳細分析スクリプト

以下を算出：
- 全体の特徴量重要度
- 全体の正解率
- ユーザーごとの正解率
- ユーザーごとの特徴量重要度（プレイヤー別に重要な特徴量を抽出）
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# taisen.py をインポート
sys.path.insert(0, str(Path(__file__).parent))
from taisen import PlayerIdentifier

def load_model_and_data():
    """保存されたモデルとデータを読み込む"""
    model_path = Path(__file__).parent / "player_model.pkl"
    scaler_path = Path(__file__).parent / "player_model_scaler.pkl"

    train_data_path = Path(__file__).parent / "model_data" / "train_data.csv"
    test_data_path = Path(__file__).parent / "model_data" / "test_data.csv"

    # モデルとスケーラーを読み込む
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # データを読み込む
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    return model, scaler, train_df, test_df

def analyze_feature_importance(model, feature_names):
    """全体の特徴量重要度を分析"""
    importances = model.feature_importances_

    # 特徴量重要度でソート
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_percent': importances * 100
    }).sort_values('importance', ascending=False)

    return feature_importance_df

def analyze_player_accuracy(test_df, y_test, y_pred):
    """ユーザーごとの正解率を分析"""
    players = sorted(test_df['player_name'].unique())

    player_accuracy = {}
    for player in players:
        mask = test_df['player_name'] == player
        if mask.sum() > 0:
            player_acc = accuracy_score(y_test[mask], y_pred[mask])
            player_precision = precision_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
            player_recall = recall_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)
            player_f1 = f1_score(y_test[mask], y_pred[mask], average='weighted', zero_division=0)

            player_accuracy[player] = {
                'accuracy': player_acc,
                'precision': player_precision,
                'recall': player_recall,
                'f1': player_f1,
                'samples': mask.sum()
            }

    return player_accuracy

def analyze_player_specific_features(train_df, model, feature_names):
    """プレイヤーごとに重要な特徴量を分析"""

    players = sorted(train_df['player_name'].unique())

    # 各プレイヤーの平均特徴量値を計算
    player_feature_means = {}
    for player in players:
        player_data = train_df[train_df['player_name'] == player]
        feature_cols = [col for col in train_df.columns if col in feature_names]
        player_feature_means[player] = player_data[feature_cols].mean()

    # プレイヤーごとに異なる特徴量を特定
    # （他のプレイヤーと比較して値が高い特徴量）
    player_specific_features = {}

    for player in players:
        this_player_means = player_feature_means[player]
        other_players_means = pd.concat([player_feature_means[p] for p in players if p != player], axis=1).mean(axis=1)

        # 差分を計算
        diff = this_player_means - other_players_means

        # 上位10個の特徴量を抽出
        top_features = diff.nlargest(10)

        player_specific_features[player] = top_features

    return player_feature_means, player_specific_features

def print_overall_analysis(model, feature_names, test_df, y_test, y_pred):
    """全体の分析を出力"""

    print("=" * 80)
    print("【全体分析】")
    print("=" * 80)

    # 全体の正解率
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n全体正解率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

    # クラス別の正解率
    print("\n各クラスの正解率:")
    print(classification_report(y_test, y_pred, digits=4))

    # 特徴量重要度
    print("\n【特徴量重要度（全体）】")
    print("=" * 80)
    feature_importance_df = analyze_feature_importance(model, feature_names)

    # 上位20を表示
    print("\nTop 20 特徴量:")
    for idx, row in feature_importance_df.head(20).iterrows():
        print(f"  {row['feature']:40s}: {row['importance_percent']:6.2f}%")

    # 下位10を表示
    print("\nBottom 10 特徴量:")
    for idx, row in feature_importance_df.tail(10).iterrows():
        print(f"  {row['feature']:40s}: {row['importance_percent']:6.2f}%")

    return feature_importance_df

def print_player_analysis(test_df, y_test, y_pred, player_accuracy, player_specific_features):
    """プレイヤーごとの分析を出力"""

    print("\n" + "=" * 80)
    print("【ユーザーごとの分析】")
    print("=" * 80)

    # ユーザーごとの正解率
    print("\nユーザーごとの正解率:")
    print("-" * 80)

    for player in sorted(player_accuracy.keys()):
        stats = player_accuracy[player]
        print(f"\n{player}:")
        print(f"  サンプル数:  {stats['samples']:4d}")
        print(f"  正解率:     {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
        print(f"  精度(P):    {stats['precision']:.4f}")
        print(f"  再現率(R):  {stats['recall']:.4f}")
        print(f"  F1スコア:   {stats['f1']:.4f}")

    # プレイヤーごとの特徴量重要度
    print("\n" + "=" * 80)
    print("【ユーザーごとに重要な特徴量】")
    print("=" * 80)

    for player in sorted(player_specific_features.keys()):
        print(f"\n{player}に特徴的な特徴量（他のプレイヤーとの差）:")
        print("-" * 80)
        features = player_specific_features[player]
        for idx, (feat_name, value) in enumerate(features.items(), 1):
            print(f"  {idx:2d}. {feat_name:40s}: {value:+.4f}")

def save_analysis_reports(feature_importance_df, player_accuracy, player_specific_features):
    """分析結果をファイルに保存"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 特徴量重要度をCSVに保存
    feature_importance_df.to_csv(
        output_dir / "feature_importance_all.csv",
        index=False,
        encoding="utf-8"
    )
    print(f"\n[保存] 特徴量重要度（全体）: {output_dir / 'feature_importance_all.csv'}")

    # プレイヤーごとの精度をCSVに保存
    player_accuracy_df = pd.DataFrame(player_accuracy).T
    player_accuracy_df.to_csv(
        output_dir / "player_accuracy.csv",
        encoding="utf-8"
    )
    print(f"[保存] ユーザーごとの正解率: {output_dir / 'player_accuracy.csv'}")

    # プレイヤーごとの特徴量をCSVに保存
    for player, features in player_specific_features.items():
        features_df = pd.DataFrame({
            'feature': features.index,
            'diff_from_others': features.values
        })
        features_df.to_csv(
            output_dir / f"player_specific_features_{player}.csv",
            index=False,
            encoding="utf-8"
        )
    print(f"[保存] ユーザーごとの特徴量: {output_dir / 'player_specific_features_*.csv'}")

def plot_analysis(feature_importance_df, player_accuracy):
    """分析結果をプロット"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 特徴量重要度のプロット（全体）
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 30
    top_features = feature_importance_df.head(top_n)
    ax.barh(range(len(top_features)), top_features['importance_percent'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('重要度 (%)')
    ax.set_title('特徴量重要度 (Top 30)')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_overall_top30.png', dpi=150, bbox_inches='tight')
    print(f"\n[保存] グラフ: {output_dir / 'feature_importance_overall_top30.png'}")
    plt.close()

    # プレイヤーごとの正解率プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    players = sorted(player_accuracy.keys())
    accuracies = [player_accuracy[p]['accuracy'] * 100 for p in players]
    colors = ['green' if acc >= 50 else 'orange' if acc >= 25 else 'red' for acc in accuracies]
    ax.bar(players, accuracies, color=colors)
    ax.set_ylabel('正解率 (%)')
    ax.set_title('ユーザーごとの正解率')
    ax.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='25% (平均的)')
    ax.set_ylim(0, 100)
    for i, (player, acc) in enumerate(zip(players, accuracies)):
        ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', va='bottom')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'player_accuracy.png', dpi=150, bbox_inches='tight')
    print(f"[保存] グラフ: {output_dir / 'player_accuracy.png'}")
    plt.close()

def main():
    print("=" * 80)
    print("プレイヤー特定モデル - 詳細分析")
    print("=" * 80)

    # モデルとデータを読み込む
    print("\nモデルとデータを読み込み中...")
    model, scaler, train_df, test_df = load_model_and_data()

    # 特徴量名を取得
    feature_cols = [col for col in train_df.columns
                   if col not in ['player_name', 'train_file_indices', 'test_file_indices', 'data_type']]
    feature_names = feature_cols

    # テストデータで予測
    X_test = test_df[feature_names].values
    y_test = test_df['player_name'].values
    y_pred = model.predict(X_test)

    # 全体分析を実施
    feature_importance_df = print_overall_analysis(model, feature_names, test_df, y_test, y_pred)

    # プレイヤーごとの分析を実施
    player_accuracy = analyze_player_accuracy(test_df, y_test, y_pred)
    player_feature_means, player_specific_features = analyze_player_specific_features(train_df, model, feature_names)

    # プレイヤーごとの分析を出力
    print_player_analysis(test_df, y_test, y_pred, player_accuracy, player_specific_features)

    # 結果を保存
    save_analysis_reports(feature_importance_df, player_accuracy, player_specific_features)

    # グラフを生成
    plot_analysis(feature_importance_df, player_accuracy)

    print("\n" + "=" * 80)
    print("分析完了！")
    print("=" * 80)

if __name__ == "__main__":
    main()
