# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:22:04 2019

@author: Allen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def TimePlot(time_data_modified, sampling_rate):
    plt.figure().set_size_inches(12, 8)
    time_series = pd.Series(time_data_modified, index = np.arange(0, len(time_data_modified)/sampling_rate, 1/sampling_rate))
    time_series = pd.to_numeric(time_series)
    plt.figure().set_size_inches(14, 10)
    time_series.plot()
    plt.title("Time domain",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Accelaration (g)")
    plt.xlabel('Time (s)')
    plt.show()
        
def DataThreshold(time_data, threshold):
    for i in range(len(time_data)):
        if float(time_data[i]) >= threshold:
            time_data[i] = 0
    return time_data

