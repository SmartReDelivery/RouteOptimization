from typing import Tuple
import numpy as np


def maximize_availability(A: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    最大の在宅確率を得る配送スケジュールを求める

    配送順序が決まっているとき、その順序において最大の在宅確率を得る配送時刻を求める。

    Parameters
    ----------
    A : np.ndarray
        `A[i, j]` は時刻`i`に荷物`j`を配送した場合の報酬 (在宅確率).
        ただし荷物は `j = 0, 1, 2, ...` の順に配送されるものとする

    Returns
    -------
    max_availability : float
        最大の在宅確率の合計
    schedule : np.ndarray
        `schedule[k]` は荷物`k`の配送時刻(`A`のインデックス)を示す.
        もし要素に`-1`が含まれていたら, それはバグである

    >>> A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    >>> maximize_availability(A)
    (3.0, array([0, 1, 2, 2]))
    """

    # 各時間iにおいて、次に配送する荷物jを決める際、次の2つの選択肢がある:
    # 1. 時刻iはスキップし、まだ荷物jを配送せずに次の時刻へ進む
    # 2. 時刻iに荷物jを配送して報酬A[i][j-1]を得て、次は荷物j＋1の配送に移る

    (T, N) = A.shape  # 荷物の個数, 時間スロット数
    # dp[i, j]: 時刻iまでにj個配送したときの最大報酬
    dp = np.full((T, N + 1), -np.inf)
    dp[:, 0] = 0
    # steps[i, j]: 時刻iでの配送決定 (0: スキップ, 1: 配送)
    steps = np.zeros((T, N + 1), dtype=int)

    for i in range(T):
        for j in range(N + 1):
            # 1. 時刻iをスキップする場合
            if i > 0:
                if dp[i - 1, j] > dp[i, j]:
                    dp[i, j] = dp[i - 1, j]
                    steps[i, j] = 0
            # 2. 時刻iに配送する場合
            if j > 0:
                if dp[i, j - 1] + A[i, j - 1] > dp[i, j]:
                    dp[i, j] = dp[i, j - 1] + A[i, j - 1]
                    steps[i, j] = 1

    # 最終的な最適値はdp[T-1, N-1]
    max_availability: float = dp[T - 1, N]

    # schedule[k] には荷物kの配送時刻(Aのインデックス)を記録
    schedule: np.ndarray = np.full(N, -1, dtype=int)
    i, j = T - 1, N
    while i >= 0 and j > 0:
        if steps[i, j] == 1:
            schedule[j - 1] = i
            j -= 1
        else:
            i -= 1

    return max_availability, schedule


if __name__ == "__main__":
    # 例: 3時間、荷物は3個とする
    # A[i, j] は時刻iに荷物jを配送した場合の報酬
    test_set = [
        (
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                ]
            ),
            np.array([0, 1, 2, 2]),
        ),
        (
            np.array(
                [
                    [1, 0, 5, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                ]
            ),
            np.array([0, 0, 0, 2]),
        ),
        (
            np.array(
                [
                    [5, 3, 2],
                    [14, 6, 13],
                    [7, 2, 8],
                    [1, 12, 3],
                ]
            ),
            np.array([1, 1, 1]),
        ),
        (
            np.array(
                [
                    [5, 3, 2],
                    [14, 1, 13],
                    [7, 2, 8],
                    [1, 12, 3],
                ]
            ),
            np.array([1, 3, 3]),
        ),
    ]

    for A, expected_schedule in test_set:
        print("A", A, sep="\n")
        max_availability, schedule = maximize_availability(A)
        print("最大報酬:", max_availability)
        # schedule[k] は、k番目の荷物を配送する時刻(Aのインデックス)を示す
        print("配送スケジュール")
        load_row = "荷物: " + " ".join(f"{i:2}" for i in range(A.shape[1]))
        time_row = "時刻: " + " ".join(f"{i:2}" for i in schedule)
        print(load_row)
        print(time_row)
        if np.array_equal(schedule, expected_schedule):
            print("Test passed")
        else:
            print("Test failed")
            print("got :", schedule)
            print("want:", expected_schedule)
        print()
