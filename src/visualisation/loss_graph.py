import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class loss_graph():
    def __init__(self, X=[], Y=[]) -> None:
        self.X: list = X
        self.Y: list = Y
        self.fig = plt.figure()
        self.sub = self.fig.add_subplot(1,1,1)

    def show(self):
        animation.FuncAnimation(self.fig, self.periodic_task, interval=10)
        plt.show()
    # to be called externally as a hook after every training epoch
    def addPoints(self, data :list[tuple]):
        for dp in data:
            self.X.append(dp[0])
            self.Y.append(dp[1])
    
    def periodic_task(self):
        self.sub.clear()
        self.sub.plot(self.X,self.Y)

#test
#loss_graph().show()