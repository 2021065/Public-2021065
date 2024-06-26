"""
迷路探索問題で強化学習を学ぶ
"""
from maze_rl_map import MAZE      # 作成した迷路をモジュール化しているためインポート
from maze_rl_agent_softmax import Agent      # 作成した迷路をモジュール化しているためインポート
import numpy as np


### 方策勾配法で迷路を解く ###########
stop_epsilon = 0.008    # 10^-4よりも方策に変化が少なくなったら学習終了

agent = Agent()
theta_0 = agent.theta_0
pi_0 = agent.softmax_convert_into_pi_from_theta(theta=agent.theta_0)


# 初期値で初期化
theta = theta_0
pi = pi_0


is_continue = True      # ループさせるフラグ
count = 1               # 学習回数（エピソード）カウント
while is_continue:
    s_a_history = agent.goal_maze(pi)

    # 更新
    new_theta = agent.update_theta(theta, pi, s_a_history)
    new_pi = agent.softmax_convert_into_pi_from_theta(new_theta)

    print(f"方策の変化：{np.sum(np.abs(new_pi - pi))}")
    print(f"迷路を解くのにかかったステップ数：{len(s_a_history) - 1}")

    if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
        is_continue = False
    else:
        theta = new_theta
        pi = new_pi
        count += 1

print(pi)

from maze_rl_gif import GIF
maze = MAZE()
state_history = [s_a[0] for s_a in s_a_history]
# print(state_history)
print(f"学習回数（エピソード）：{count}")
gif = GIF(maze.fig, maze.set_start(), state_history)

gif.create("maze_learning.gif")
