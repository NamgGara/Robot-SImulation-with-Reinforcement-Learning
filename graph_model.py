import threading
import matplotlib.pyplot as plt
import os
import seaborn as sns

file_path = "performance"

if not os.path.exists(file_path):
    os.mkdir(file_path)

plt.ion()
fig, ax= plt.subplots()

def plot(ax,index,value):

    # plt.gca().cla() # optionally clear axes
    ax.scatter(index,value)
    plt.title(str(index))
    plt.draw()
    plt.pause(0.1)

