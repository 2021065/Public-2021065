"""
迷路探索問題で強化学習を学ぶ
"""
import matplotlib.pyplot as plt


class MAZE():
    def __init__(self) -> None:
        ### 迷路作成 ###############
        self.fig = plt.figure(figsize=(10, 10))    # 5x5のグリッド図を作成（1区画を1マスとする）
        self.ax = plt.gca()                      # get current axis 今はplt.subplot(111)と同じである。つまりは、左上のマスの操作ができる。エージェントの初期位置を描画するために用意

        # 赤い壁を描く（赤い壁は通ることができないという定義）：直線描画で表現
        '''
        plt.plot([1,1], [0,1], color='red', linewidth=2)        # plt.plot(x, y, color, linewidth)   xデータ（x座標）, yデータ（y座標）, 線色, 線幅
        plt.plot([1,2], [2,2], color='red', linewidth=2)
        plt.plot([1,1], [2,1], color='red', linewidth=2)
        plt.plot([2,3], [1,1], color='red', linewidth=2)
        '''

        plt.plot([1,1], [0,1], color='red', linewidth=2)        # plt.plot(x, y, color, linewidth)   xデータ（x座標）, yデータ（y座標）, 線色, 線幅
        plt.plot([1,2], [2,2], color='red', linewidth=2)
        plt.plot([1,1], [2,1], color='red', linewidth=2)
        plt.plot([2,3], [1,1], color='red', linewidth=2)

        plt.plot([0,1], [3,3], color='blue', linewidth=4)        # plt.plot(x, y, color, linewidth)   xデータ（x座標）, yデータ（y座標）, 線色, 線幅
        plt.plot([1,2], [3,3], color='blue', linewidth=4)
        plt.plot([2,3], [3,3], color='blue', linewidth=4)

        plt.plot([0,1], [6,6], color='blue', linewidth=4)        # plt.plot(x, y, color, linewidth)   xデータ（x座標）, yデータ（y座標）, 線色, 線幅
        plt.plot([1,2], [6,6], color='blue', linewidth=4)
        plt.plot([2,3], [6,6], color='blue', linewidth=4)

        # 状態を表す文字S0~S8を描く

        plt.text(x=0.5, y=2.5, s='u0', size=14, ha='center')
        plt.text(x=1.5, y=2.5, s='u1', size=14, ha='center')
        plt.text(x=2.5, y=2.5, s='u2', size=14, ha='center')
        plt.text(x=0.5, y=1.5, s='u3', size=14, ha='center')
        plt.text(x=1.5, y=1.5, s='u4', size=14, ha='center')
        plt.text(x=2.5, y=1.5, s='u5', size=14, ha='center')
        plt.text(x=0.5, y=0.5, s='u6', size=14, ha='center')
        plt.text(x=1.5, y=0.5, s='u7', size=14, ha='center')
        plt.text(x=2.5, y=0.5, s='u8', size=14, ha='center')

        plt.text(x=0.5, y=2.5+3, s='t0', size=14, ha='center')
        plt.text(x=1.5, y=2.5+3, s='t1', size=14, ha='center')
        plt.text(x=2.5, y=2.5+3, s='t2', size=14, ha='center')
        plt.text(x=0.5, y=1.5+3, s='t3', size=14, ha='center')
        plt.text(x=1.5, y=1.5+3, s='t4', size=14, ha='center')
        plt.text(x=2.5, y=1.5+3, s='t5', size=14, ha='center')
        plt.text(x=0.5, y=0.5+3, s='t6', size=14, ha='center')
        plt.text(x=1.5, y=0.5+3, s='t7', size=14, ha='center')
        plt.text(x=2.5, y=0.5+3, s='t8', size=14, ha='center')

        plt.text(x=0.5, y=2.5+3*2, s='s0', size=14, ha='center')
        plt.text(x=1.5, y=2.5+3*2, s='s1', size=14, ha='center')
        plt.text(x=2.5, y=2.5+3*2, s='s2', size=14, ha='center')
        plt.text(x=0.5, y=1.5+3*2, s='s3', size=14, ha='center')
        plt.text(x=1.5, y=1.5+3*2, s='s4', size=14, ha='center')
        plt.text(x=2.5, y=1.5+3*2, s='s5', size=14, ha='center')
        plt.text(x=0.5, y=0.5+3*2, s='s6', size=14, ha='center')
        plt.text(x=1.5, y=0.5+3*2, s='s7', size=14, ha='center')
        plt.text(x=2.5, y=0.5+3*2, s='s8', size=14, ha='center')

        plt.text(x=0.5, y=2.3+3*2, s='START', size=10, ha='center')
        plt.text(x=2.5, y=0.3, s='GOAL', size=10, ha='center')

        # 描画範囲の設定とメモリを消す設定
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 9)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    def set_start(self):
        # 現在地S0に緑丸を描画する
        line, = self.ax.plot([0.5], [2.5+3*2], marker='o', color='lightgreen', markersize=60)    # のちに更新するためにaxで戻り値としてlineを受け取っている。lineにアクセスして座標変更が可能（代入文）←コンマが必要
        return line                                                                                # 代入文：https://docs.python.org/ja/3/reference/simple_stmts.html#assignment-statements

    def show(self):
        plt.show()

if __name__ == '__main__':
    maze = MAZE()
    line = maze.set_start()
    maze.show()
