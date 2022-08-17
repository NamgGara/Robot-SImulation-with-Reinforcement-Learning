import matplotlib.pyplot as plt
import os
import seaborn as sns

plt.ion()
fig, ax= plt.subplots()

class process_plot:

    def __init__(self) -> None:
        self.value = [0]
        self.index = 0

    def __call__(self,value):
        self.value.append(value)
        self.index +=1

    def graph(self):

        ax.plot(range(self.index),self.value)
        # plt.draw()
        plt.savefig("test")
        # plt.pause(0.1)

graph = process_plot()

