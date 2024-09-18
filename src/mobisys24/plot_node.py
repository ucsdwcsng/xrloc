#!/usr/bin/env python
import rospy
import time
import matplotlib.pyplot as plt
from geometry_msgs.msg import Vector3Stamped

# TODO: Implement a class that plots the position and ground truth data

class PlottingPosition:
    def __init__(self):
        ## Subscribers
        self.sub_position = rospy.Subscriber("/position", Vector3Stamped, self.callbackPosition)
        self.sub_gnd_truth = rospy.Subscriber("/gnd_truth", Vector3Stamped, self.callbackGndTruth)
        ## Messages
        self.position = Vector3Stamped()
        self.gnd_truth = Vector3Stamped()
        ## Lists for x, y position and ground truth
        self.list_t = []
        self.list_x = []
        self.list_y = []
        self.list_gnd_x = []
        self.list_gnd_y = []
        ## Lines for x, y position and ground truth
        self.line_x = None
        self.line_y = None
        self.line_gnd_x = None
        self.line_gnd_y = None
        ## Time
        self.start_time = time.time()
        ## Parameters
        self.interval = 0.1
        self.pos_ylim = 10.0  # Assuming position range for visualization

        ## Initialization
        self.initializePlot()
        ## Main loop
        self.mainLoop()

    def callbackPosition(self, msg):
        self.position = msg

    def callbackGndTruth(self, msg):
        self.gnd_truth = msg

    def initializePlot(self):
        plt.ion()  # Interactive mode on
        plt.figure(figsize=(10, 6))
        
        # Plot for position and ground truth
        plt.subplot(1, 1, 1)
        plt.xlabel("time [s]")
        plt.ylabel("position [units]")
        plt.ylim(-self.pos_ylim, self.pos_ylim)
        plt.grid(True)
        self.line_x, = plt.plot(self.list_t, self.list_x, label='Estimated x')
        self.line_y, = plt.plot(self.list_t, self.list_y, label='Estimated y')
        self.line_gnd_x, = plt.plot(self.list_t, self.list_gnd_x, label='Ground Truth x', linestyle='--')
        self.line_gnd_y, = plt.plot(self.list_t, self.list_gnd_y, label='Ground Truth y', linestyle='--')
        plt.legend()

    def mainLoop(self):
        while not rospy.is_shutdown():
            self.updatePlot()
            self.drawPlot()

    def updatePlot(self):
        # Append time
        t = time.time() - self.start_time
        self.list_t.append(t)
        # Update Position data
        self.list_x.append(self.position.vector.x)
        self.list_y.append(self.position.vector.y)
        # Update Ground Truth data
        self.list_gnd_x.append(self.gnd_truth.vector.x)
        self.list_gnd_y.append(self.gnd_truth.vector.y)
        # Pop oldest data if necessary
        if len(self.list_t) > 100:
            self.list_t.pop(0)
            self.list_x.pop(0)
            self.list_y.pop(0)
            self.list_gnd_x.pop(0)
            self.list_gnd_y.pop(0)

        # Update plot for position and ground truth
        plt.subplot(1, 1, 1)
        self.line_x.set_xdata(self.list_t)
        self.line_x.set_ydata(self.list_x)
        self.line_y.set_xdata(self.list_t)
        self.line_y.set_ydata(self.list_y)
        self.line_gnd_x.set_xdata(self.list_t)
        self.line_gnd_x.set_ydata(self.list_gnd_x)
        self.line_gnd_y.set_xdata(self.list_t)
        self.line_gnd_y.set_ydata(self.list_gnd_y)
        plt.xlim(min(self.list_t), max(self.list_t))

    def drawPlot(self):
        plt.draw()
        plt.pause(self.interval)

def main():
    rospy.init_node('plotting_position', anonymous=True)
    plotting_position = PlottingPosition()
    rospy.spin()

if __name__ == '__main__':
    main()
