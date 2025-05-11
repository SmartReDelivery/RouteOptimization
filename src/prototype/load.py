from typing import List

import tsplib95

from . import utils


def load_location_tsplib(file_path) -> List[utils.Location]:
    """
    TSPLIB形式のファイルを読み込み、ノードの座標を取得する関数
    :param file_path: TSPLIB形式のファイルパス
    :return: ノードの座標を含む辞書 {ノードID: (x, y)}
    """
    problem = tsplib95.load(file_path)
    if problem.node_coords:
        locations = []
        for node_id, coords in problem.node_coords.items():
            locations.append((coords[0], coords[1]))
        return locations
    else:
        raise ValueError("No node coordinates found in the TSPLIB file.")


if __name__ == "__main__":

    # TSPLIBファイルのパスを指定
    # 例: 'att48.tsp' のようにファイル名を指定
    # ファイルは事前にダウンロードして、実行コードと同じディレクトリか指定したパスに置いてください
    file_path = "src/prototype/asset/att48.tsp"  # 例としてファイル名を入れています

    locations = load_location_tsplib(file_path)
    print("Loaded locations:", locations)

    # try:
    #     # 問題ファイルを読み込む
    #     problem = tsplib95.load(file_path)

    #     # 問題の情報を表示
    #     print(f"Problem Name: {problem.name}")
    #     print(f"Problem Type: {problem.type}")
    #     print(f"Dimension: {problem.dimension}")
    #     print(f"Edge Weight Type: {problem.edge_weight_type}")

    #     # ノードの座標を取得 (NODE_COORD_SECTIONがある場合)
    #     if problem.node_coords:
    #         print("\nNode Coordinates (first 5):")
    #         # problem.node_coords はディクショナリ {node_id: (x, y)}
    #         for node_id, coords in list(problem.node_coords.items())[:5]:
    #             print(f"  Node {node_id}: {coords}")

    #     # 距離行列を取得 (EDGE_WEIGHT_SECTION または計算される場合)
    #     # problem.get_graph() で NetworkX のグラフオブジェクトを取得できる
    #     graph = problem.get_graph()
    #     print(f"\nGraph Info: {graph}")

    #     # 距離を計算する例 (EUC_2Dの場合など)
    #     # problem.get_weight(i, j) でノードiとノードj間の距離を取得できる
    #     # ただし、距離はノードID (通常は1から始まる整数) で指定する
    #     # TSPLIBのノードIDは1から始まることが多いので注意
    #     if problem.dimension >= 2:
    #         # 例としてノード1とノード2の距離を取得
    #         try:
    #             distance_1_2 = problem.get_weight(1, 2)
    #             print(f"\nDistance between node 1 and node 2: {distance_1_2}")
    #         except Exception as e:
    #             print(f"Could not get weight between node 1 and 2: {e}")

    # except FileNotFoundError:
    #     print(f"Error: File not found at {file_path}")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
