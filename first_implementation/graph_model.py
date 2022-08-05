from multiprocessing import Queue, Process
import multiprocessing
from queue import Queue
import matplotlib.pyplot as plt
import os
import seaborn as sns

file_path = "performance"

if not os.path.exists(file_path):
    os.mkdir(file_path)

plt.ion()
fig, ax= plt.subplots()

query = Queue()

# def plot(index,value,ax=ax):
#     # plt.gca().cla() # optionally clear axes
#     ax.scatter(index,value)
#     plt.title(str(index))
#     plt.draw()
#     plt.pause(0.1)

def get_values(index,value):
    query.put((index,value))

def multi_process_plot():
    index, value = query.get()
    ax.scatter(index,value)
    plt.title(str(index))
    plt.draw()
    plt.pause(0.1)

plot_thread = Process(target=multi_process_plot, daemon=True)
