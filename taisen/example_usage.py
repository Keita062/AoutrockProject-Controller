"""
プレイヤー特定モデルの使用例

このスクリプトは以下の操作例を示します:
1. モデルの学習
2. 新しいデータでの予測
3. モデルの保存・読み込み
"""

from pathlib import Path
from taisen import PlayerIdentifier
import pandas as pd


def example_1_train_model():
    """例1: モデルを学習する"""
    print("=" * 60)
    print("例1: モデルを学習する")
    print("=" * 60)

    # データディレクトリを指定
    data_dirs = [
        r"c:\GitHub\AoutrockProject-Controller\2026_1_7_対戦記録",
        r"c:\GitHub\AoutrockProject-Controller\2026_1_15_対戦記録",
        r"c:\GitHub\AoutrockProject-Controller\2026_1_20_対戦記録",
    ]

    # モデルを作成
    model = PlayerIdentifier(
        data_dirs=data_dirs,
        window_size=50
    )

    # 学習
    results = model.fit()

    print(f"\n学習完了！")
    print(f"精度: {results['accuracy']:.4f}")

    return model


def example_2_predict_player(model):
    """例2: 新しいCSVファイルで予測する"""
    print("\n" + "=" * 60)
    print("例2: 新しいCSVファイルで予測する")
    print("=" * 60)

    # テスト用のCSVファイル
    csv_file = r"c:\GitHub\AoutrockProject-Controller\2026_1_15_対戦記録\kotaro usigome.csv"

    # con1 (kotaro) のプレイヤーを予測
    player = model.predict(csv_file, username="con1")
    print(f"\nCSV: {Path(csv_file).name}")
    print(f"Username: con1")
    print(f"予測されたプレイヤー: {player}")


def example_3_save_and_load(model):
    """例3: モデルを保存・読み込みする"""
    print("\n" + "=" * 60)
    print("例3: モデルを保存・読み込みする")
    print("=" * 60)

    # モデルを保存
    model_path = r"c:\GitHub\AoutrockProject-Controller\taisen\custom_model.pkl"
    model.save(model_path)
    print(f"モデルを保存しました: {model_path}")

    # 新しいモデルインスタンスを作成
    new_model = PlayerIdentifier(data_dirs=[])

    # 保存したモデルを読み込み
    new_model.load(model_path)
    print(f"モデルを読み込みました")

    # 読み込んだモデルで予測
    csv_file = r"c:\GitHub\AoutrockProject-Controller\2026_1_15_対戦記録\yuusuke kotaro.csv"
    player = new_model.predict(csv_file, username="con2")
    print(f"\n読み込んだモデルで予測: {player}")


def example_4_batch_predict():
    """例4: 複数のCSVファイルをまとめて予測する"""
    print("\n" + "=" * 60)
    print("例4: 複数のCSVファイルをまとめて予測する")
    print("=" * 60)

    from taisen import PlayerIdentifier

    # モデルを読み込み
    model = PlayerIdentifier(data_dirs=[])
    model.load(r"c:\GitHub\AoutrockProject-Controller\taisen\player_model.pkl")

    # 対象ディレクトリ
    target_dir = Path(r"c:\GitHub\AoutrockProject-Controller\2026_1_15_対戦記録")

    results = []

    # すべてのCSVファイルを処理
    for csv_file in sorted(target_dir.glob("*.csv")):
        try:
            # ファイル名からプレイヤー名を抽出
            file_name = csv_file.name.replace(".csv", "")
            parts = file_name.split()

            if len(parts) >= 2:
                player1_name = parts[0]
                player2_name = parts[1]

                # con1 と con2 の予測
                con1_pred = model.predict(str(csv_file), username="con1")
                con2_pred = model.predict(str(csv_file), username="con2")

                results.append({
                    "ファイル": file_name,
                    "con1_実際": player1_name,
                    "con1_予測": con1_pred,
                    "con2_実際": player2_name,
                    "con2_予測": con2_pred,
                })

                print(f"✓ {file_name}")
        except Exception as e:
            print(f"✗ {csv_file.name}: {e}")

    # 結果をDataFrameで表示
    if results:
        df = pd.DataFrame(results)
        print("\n予測結果:")
        print(df.to_string(index=False))

        # 精度を計算
        con1_correct = (df["con1_実際"] == df["con1_予測"]).sum()
        con2_correct = (df["con2_実際"] == df["con2_予測"]).sum()
        total = len(df) * 2

        print(f"\n精度: {(con1_correct + con2_correct) / total:.2%}")


def main():
    """すべての例を実行する"""
    print("\nプレイヤー特定モデル - 使用例")
    print("=" * 60)

    # 例1: モデルの学習
    model = example_1_train_model()

    # 例2: 予測
    example_2_predict_player(model)

    # 例3: 保存・読み込み
    example_3_save_and_load(model)

    # 例4: バッチ予測（時間がかかるためコメントアウト）
    # example_4_batch_predict()

    print("\n" + "=" * 60)
    print("すべての例が完了しました！")
    print("=" * 60)


if __name__ == "__main__":
    main()
