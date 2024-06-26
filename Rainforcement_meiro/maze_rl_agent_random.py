"""
迷路探索問題で強化学習を学ぶ
"""
from maze_rl_map import MAZE      # 作成した迷路をモジュール化しているためインポート
import numpy as np

# 迷路の作成
maze = MAZE()
line = maze.set_start()     # 動き回るエージェントの座標を変更できる変数を取得
# line.set_data([1, 2])
# maze.show()


### エージェントの実装 ####################
class Agent():
    def __init__(self) -> None:
        # 進めるルールを定義
        # 行：状態S0~S7（S8はゴールであるから方策不要）、列：選択（↑, →, ↓, ←）
        '''
        self.theta_0 = np.array([
                            [np.nan,    1,      1,      np.nan],    # S0: ↑ 不可    → 可    ↓ 可    ← 不可
                            [np.nan,    1,      np.nan, 1],         # S1: ↑ 不可    → 可    ↓ 不可  ← 可
                            [np.nan,    np.nan, 1,      1],         # S2: ↑ 不可    → 不可  ↓ 可    ← 可
                            [1 ,        1,      1,      np.nan],    # S3: ↑ 可      → 可    ↓ 可    ← 不可
                            [np.nan,    np.nan, 1,      1],         # S4: ↑ 不可    → 不可  ↓ 可    ← 可
                            [1,         np.nan, np.nan, np.nan],    # S5: ↑ 可      → 不可  ↓ 不可  ← 不可
                            [1,         np.nan, np.nan, np.nan],    # S6: ↑ 可      → 不可  ↓ 不可  ← 不可
                            [1,         1,      np.nan, np.nan],    # S7: ↑ 可      → 可    ↓ 不可  ← 不可
                            ])
        '''
        self.theta_0 = np.array([
                    [np.nan,    1,      1,      np.nan, 1, np.nan],    # U0: ↑ 不可    → 可    ↓ 可    ← 不可　　F↑ 可　　F↓ 不可
                    [np.nan,    1,      np.nan, 1,      1, np.nan],    # U1: ↑ 不可    → 可    ↓ 不可  ← 可　　　F↑ 可　　F↓ 不可
                    [np.nan,    np.nan, 1,      1,      1, np.nan],    # U2: ↑ 不可    → 不可  ↓ 可    ← 可　　　F↑ 可　　F↓ 不可
                    [1 ,        np.nan, 1,      1,      1, np.nan],    # U3: ↑ 可      → 不可    ↓ 可    ← 不可　F↑ 可　　F↓ 不可
                    [np.nan,    1,      1,      np.nan, 1, np.nan],    # U4: ↑ 不可    → 可  ↓ 可    ← 不可　　　F↑ 可　　F↓ 不可
                    [1,         np.nan, np.nan, 1,      1, np.nan],    # U5: ↑ 可      → 不可  ↓ 不可  ← 可　　　F↑ 可　　F↓ 不可
                    [1,         np.nan, np.nan, np.nan, 1, np.nan],    # U6: ↑ 可      → 不可  ↓ 不可  ← 不可　　F↑ 可　　F↓ 不可
                    [1,         1,      np.nan, np.nan, 1, np.nan],    # U7: ↑ 可      → 可    ↓ 不可  ← 不可　　F↑ 可　　F↓ 不可
                    [1,         np.nan, np.nan, 1,      1, np.nan],    # U8: ↑ 可      → 不可    ↓ 不可  ← 可　　F↑ 可　　F↓ 不可
                    
                    [np.nan,    1,      1,      np.nan, 1, 1],    # T0: ↑ 不可    → 可    ↓ 可    ← 不可　　F↑ 可　　F↓ 可
                    [np.nan,    1,      np.nan, 1,      1, 1],    # T1: ↑ 不可    → 可    ↓ 不可  ← 可　　　F↑ 可　　F↓ 可
                    [np.nan,    np.nan, 1,      1,      1, 1],    # T2: ↑ 不可    → 不可  ↓ 可    ← 可　　　F↑ 可　　F↓ 可
                    [1 ,        np.nan, 1,      1,      1, 1],    # T3: ↑ 可      → 不可    ↓ 可    ← 不可　F↑ 可　　F↓ 可
                    [np.nan,    1,      1,      np.nan, 1, 1],    # T4: ↑ 不可    → 可  ↓ 可    ← 不可　　　F↑ 可　　F↓ 可
                    [1,         np.nan, np.nan, 1,      1, 1],    # T5: ↑ 可      → 不可  ↓ 不可  ← 可　　　F↑ 可　　F↓ 可
                    [1,         np.nan, np.nan, np.nan, 1, 1],    # T6: ↑ 可      → 不可  ↓ 不可  ← 不可　　F↑ 可　　F↓ 可
                    [1,         1,      np.nan, np.nan, 1, 1],    # T7: ↑ 可      → 可    ↓ 不可  ← 不可　　F↑ 可　　F↓ 可
                    [1,         np.nan, np.nan, 1,      1, 1],    # T8: ↑ 可      → 不可    ↓ 不可  ← 可　　F↑ 可　　F↓ 可

                    [np.nan,    1,      1,      np.nan, np.nan, 1],    # S0: ↑ 不可    → 可    ↓ 可    ← 不可　　F↑ 不可　　F↓ 可
                    [np.nan,    1,      np.nan, 1,      np.nan, 1],    # S1: ↑ 不可    → 可    ↓ 不可  ← 可　　　F↑ 不可　　F↓ 可
                    [np.nan,    np.nan, 1,      1,      np.nan, 1],    # S2: ↑ 不可    → 不可  ↓ 可    ← 可　　　F↑ 不可　　F↓ 可
                    [1 ,        np.nan, 1,      1,      np.nan, 1],    # S3: ↑ 可      → 不可    ↓ 可    ← 不可
                    [np.nan,    1,      1,      np.nan, np.nan, 1],    # S4: ↑ 不可    → 可  ↓ 可    ← 不可　　　F↑ 不可　　F↓ 可
                    [1,         np.nan, np.nan, 1,      np.nan, 1],    # S5: ↑ 可      → 不可  ↓ 不可  ← 可　　　F↑ 不可　　F↓ 可
                    [1,         np.nan, np.nan, np.nan, np.nan, 1],    # S6: ↑ 可      → 不可  ↓ 不可  ← 不可　　F↑ 不可　　F↓ 可
                    [1,         1,      np.nan, np.nan, np.nan, 1],    # S7: ↑ 可      → 可    ↓ 不可  ← 不可　　F↑ 不可　　F↓ 可
                    ])

    # 方策パラメータ（ルール）から行動方策piを導く
    def simple_convert_into_pi_from_theta(self, theta):
        """単純に割合（その行動をとる確率）を計算する"""

        [m, n] = theta.shape    # thetaの行列サイズを取得
        print(theta.shape)
        pi = np.zeros((m, n))

        for i in range(m):
            pi[i, :] = theta[i, :] / np.nansum(theta[i, :]) # 割合計算（各箇所をその要素合計で割る）
        
        pi = np.nan_to_num(pi)

        return pi

    # 1 step移動後の状態sを求める
    def get_next_s(self, pi, s):
        direction = ['up', 'right', 'down', 'left', 'floor up', 'floor down']

        next_direction = np.random.choice(direction, p=pi[s, :])    # pi[s,:]の確率に従ってdirectionが選択される

        if next_direction == 'up':
            s_next = s - 3
        elif next_direction == 'right':
            s_next = s + 1
        elif next_direction == 'down':
            s_next = s + 3
        elif next_direction == 'left':
            s_next = s - 1
        elif next_direction == 'floor up':
            s_next = s + 9
        elif next_direction == 'floor down':
            s_next = s - 9
        return s_next


    # 迷路内をエージェントがゴールするまで移動させる
    def goal_maze(self, pi):
        s = 0       # スタート状態S0
        state_history = [0] # エージェントの移動した道を記録

        while True:
            next_s = self.get_next_s(pi, s)
            state_history.append(next_s)

            if next_s == 26:
                break
            else:
                s = next_s
        
        return state_history

if __name__ == '__main__':
    agent = Agent()
    pi_0 = agent.simple_convert_into_pi_from_theta(theta=agent.theta_0)     # 初期の方策
    state_history = agent.goal_maze(pi_0)                                   # ゴールするまで1つの方策でランダム動き回る
    print(state_history)
    print(pi_0)

