"""
迷路探索問題で強化学習を学ぶ
"""
from maze_rl_map import MAZE      # 作成した迷路をモジュール化しているためインポート
import numpy as np

# 迷路の作成
maze = MAZE()
line = maze.set_start()     # 動き回るエージェントの座標を変更できる変数を取得


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

    # 方策パラメータ（ルール）から行動方策piをソフトマックス関数で導く
    def softmax_convert_into_pi_from_theta(self, theta):
        """単純に割合（その行動をとる確率）を計算する"""

        beta = 1.0
        [m, n] = theta.shape    # thetaの行列サイズを取得
        pi = np.zeros((m, n))

        exp_theta = np.exp(theta * beta)

        for i in range(m):

            pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
        pi = np.nan_to_num(pi)

        return pi

    # 1 step移動後の状態sを求める
    '''
    def get_next_s_and_action(self, pi, s):
        direction = ['up', 'right', 'down', 'left']

        next_direction = np.random.choice(direction, p=pi[s, :])    # pi[s,:]の確率に従ってdirectionが選択される

        if next_direction == 'up':
            action = 0
            s_next = s - 3
        elif next_direction == 'right':
            action = 1
            s_next = s + 1
        elif next_direction == 'down':
            action = 2
            s_next = s + 3
        elif next_direction == 'left':
            action = 3
            s_next = s - 1
        
        return s_next, action
        '''
    def get_next_s_and_action(self, pi, s):
        direction = ['up', 'right', 'down', 'left', 'floor up', 'floor down']

        next_direction = np.random.choice(direction, p=pi[s, :])    # pi[s,:]の確率に従ってdirectionが選択される

        if next_direction == 'up':
            action = 0
            s_next = s - 3
        elif next_direction == 'right':
            action = 1
            s_next = s + 1
        elif next_direction == 'down':
            action = 2
            s_next = s + 3
        elif next_direction == 'left':
            action = 3
            s_next = s - 1
        elif next_direction == 'floor up':
            action = 4
            s_next = s + 9
        elif next_direction == 'floor down':
            action = 5
            s_next = s - 9
        
        return s_next, action


    # 迷路内をエージェントがゴールするまで移動させる
    def goal_maze(self, pi):
        s = 0       # スタート状態S0
        s_a_history = [[0, np.nan]] # エージェントの移動した道を記録

        while True:
            next_s, action = self.get_next_s_and_action(pi, s)
            s_a_history[-1][1] = action
            s_a_history.append([next_s, np.nan])

            if next_s == 26:
                break
            else:
                s = next_s
        
        return s_a_history
    
    # 方策パラメータの更新
    def update_theta(self, theta, pi, s_a_history):
        eta = 0.1   # 学習率
        T = len(s_a_history) - 1    # ゴールまでの総ステップ数

        [m, n] = theta.shape
        delta_theta = theta.copy()

        # delta_thetaを要素ごとに求める
        for i in range(m):
            for j in range(n):
                if not(np.isnan(theta[i, j])):    # thetaがnanでないとき

                    SA_i = [SA for SA in s_a_history if SA[0] == i]     # 状態がiのものだけを抽出

                    SA_ij = [SA for SA in s_a_history if SA == [i, j]]  # 状態がiで行動jをしたものだけを抽出

                    N_i = len(SA_i)
                    N_ij = len(SA_ij)
                    
                    delta_theta[i, j] = (N_ij - pi[i, j] * N_i) / T
        
        new_theta = theta + eta * delta_theta
    
        return new_theta


if __name__ == '__main__':
    agent = Agent()
    pi_0 = agent.softmax_convert_into_pi_from_theta(theta=agent.theta_0)     # 初期の方策
    s_a_history = agent.goal_maze(pi_0)                                   # ゴールするまで1つの方策でランダム動き回る
    new_theta = agent.update_theta(agent.theta_0, pi_0, s_a_history)
    pi = agent.softmax_convert_into_pi_from_theta(new_theta)
    print(pi)
