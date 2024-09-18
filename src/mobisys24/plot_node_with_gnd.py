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

# 真値（ground truth）の位置データ（動的に更新される）
gnd_truth = np.array([0.0, 0.0])  # 初期値として0を設定

# プロット用データ
x_data = []
y_data = []
error_data = []

plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 30

# プロットを更新する関数
def update_plot(frame):
    plt.clf()  # 全グラフをクリア
    plt.subplot(2, 1, 1)  # 2行1列の1番目
    plt.scatter(x_data, abs(y_data), color='blue', label='Estimated Positions')
    plt.scatter(rx_locations[:, 0], rx_locations[:, 1], color='red', marker='v', s=40, label='Receivers')
    plt.scatter(gnd_truth[0], gnd_truth[1], color='green', marker='x', s=100, label='Ground Truth')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Real-time Plot of Positions')
    plt.legend()

    plt.subplot(2, 1, 2)  # 2行1列の2番目
    plt.plot(error_data, color='red', label='Position Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title('Error Over Time')
    plt.legend()

# 推定位置のROSコールバック関数
def callback(msg):
    x_data.append(msg.vector.x)
    y_data.append(msg.vector.y)
    current_error = np.linalg.norm(np.array([msg.vector.x, msg.vector.y]) - gnd_truth)
    error_data.append(current_error)
    if len(x_data) > 50:  # データ上限を50に設定
        x_data.pop(0)
        y_data.pop(0)
        error_data.pop(0)

# 真値データのROSコールバック関数
def ground_truth_callback(msg):
    gnd_truth[0] = msg.vector.x
    gnd_truth[1] = msg.vector.y

# ノードの初期化とサブスクライバーの設定
def listener():
    rospy.init_node('plotter_node', anonymous=True)
    rospy.Subscriber("result_data", Vector3Stamped, callback)
    rospy.Subscriber("ground_truth_data", Vector3Stamped, ground_truth_callback)
    plt.figure(figsize=(12, 10))
    ani = FuncAnimation(plt.gcf(), update_plot, interval=100)
    plt.show()

if __name__ == '__main__':
    listener()
