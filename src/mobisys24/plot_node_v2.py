#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Vector3Stamped
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# 受信機の位置データ
rx_locations = np.array([
    [2.0, 5],
    [2.2, 5],
    [2.4, 5],
    [2.6, 5],
    [2.8, 5],
    [3.0, 5]
])

# プロット用データ
x_data = []
y_data = []

plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 30

# プロットを更新する関数
def update_plot(frame):
    
    # plt.rcParams["figure.figsize"] = (14, 8)
    # plt.rcParams['figure.subplot.bottom'] = 0.2
    plt.cla()  # 現在のプロットをクリア
    plt.scatter(x_data, y_data, color='blue', s=90)  # 座標点をプロット
    plt.scatter(rx_locations[:, 0], rx_locations[:, 1], color='red', marker='v', s= 500)  # 受信機の位置をプロット
    plt.xlim(0, 5)  # X軸の範囲
    plt.ylim(0, 5)  # Y軸の範囲
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Real-time Plot of Positions')

# ROSのコールバック関数
def callback(msg):
    x_data.append(msg.vector.x)
    y_data.append(msg.vector.y)
    # データの上限を設ける場合はここで管理
    if len(x_data) > 3:
        x_data.pop(0)
        y_data.pop(0)

# ノードの初期化とサブスクライバーの設定
def listener():
    rospy.init_node('plotter_node', anonymous=True)
    rospy.Subscriber("result_data", Vector3Stamped, callback)
    plt.figure()
    ani = FuncAnimation(plt.gcf(), update_plot, interval=100)
    plt.show()

if __name__ == '__main__':
    listener()
