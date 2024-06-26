"""
迷路探索問題で強化学習を学ぶ
"""
from maze_rl_agent_random import Agent      # 作成したエージェントをモジュール化しているためインポート
from maze_rl_map import MAZE         # 作成した迷路をモジュール化しているためインポート
from matplotlib import animation as ani
from os.path import join, dirname, abspath

### 動いている様子を可視化 #############
class GIF():
    def __init__(self, fig, line, state_history) -> None:
        self.fig = fig
        self.state_history = state_history
        self.line = line

    def init_func(self):
        """背景画像の初期化"""
        line = self.line.set_data([], [])
        return (line,)

    def animate(self, i):
        """フレームごとの描画内容"""
        state = self.state_history[i]
        x = (state%3) + 0.5         # 状態のx座標は、3で割ったあまり + 0.5
        y = (2.5+3*2) - int(state/3)         # 状態のy座標は、2.5 - 3で割った商

        line = self.line.set_data(x, y)
        return (line,)

    def create(self, file_name="maze_random.gif"):
        anim = ani.FuncAnimation(self.fig,  self.animate, init_func=self.init_func, frames=len(self.state_history), interval=200, repeat=False)

        save_path = dirname(abspath(__file__))
        anim.save(f"{save_path}/{file_name}")

if __name__ == '__main__':
    # 迷路の作成
    maze = MAZE()
    line = maze.set_start()     # 動き回るエージェントの座標を変更できる変数を取得

    # エージェント
    agent = Agent()
    pi_0 = agent.simple_convert_into_pi_from_theta(theta=agent.theta_0)     # 初期の方策
    state_history = agent.goal_maze(pi_0)                                   # ゴールするまで1つの方策でランダム動き回る

    # 記録
    gif = GIF(maze.fig, line, state_history)
    gif.create(file_name="maze_random.gif")
