# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:11:05 2019

@author: Allen
"""
import os
#import tensorflow as tf
import numpy as np
#import random
import csv
import math
#import matplotlib.pyplot as ma
#import keras
#from keras.utils import np_utils
#from keras.models import Sequential
#from keras.layers import Dense,Dropout,Activation,Flatten
#from keras.layers import Conv1D,MaxPooling1D,ZeroPadding1D,normalization
import pywt
import itertools
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox


def open_file():
    #window.withdraw()
    file_path = filedialog.askdirectory()
    file_path_text.set(''+file_path)
    list_file(file_path)
    #files = os.listdir(file_path)
    #for file in files:
    #    listbox.insert(tk.END, file)
    #print(file_path, type(file_path))
    
    
def list_file(path):
    listbox.delete(0,tk.END)
    list_files = os.listdir(path)
    for file in list_files:
        listbox.insert(tk.END, file)
    
def export_file():
    return 0

def WPT():
    path = file_path_text.get()
    if path == '':
        tk.messagebox.showerror(title = 'Oops', message = 'Please select a file path')
    elif output_label_str.get() == '':
        tk.messagebox.showerror(title='Oops', message = 'Please enter label number')
    elif output_decomposed_str.get() == '':
        tk.messagebox.showerror(title='Oops', message = 'Please enter decomposed level number')
    else:
        files = os.listdir(path)
        input_datas = []
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                tem_input_datas = list(csv.reader(open(path + "/" + file,'r')))
                del tem_input_datas[0]
                for i in range(len(tem_input_datas)):
                    tem_input_datas[i].pop(0)
                tem_input_datas = list(itertools.chain.from_iterable(tem_input_datas))
                input_datas.append(tem_input_datas)
                    #tk.messagebox.showerror(title = 'Oops', message = path)
                #wp = 0
        wp_node = []
############################
## Coefficient Initialize ##
        coef=[]
        reC = []
        reS = []
        decomposed_level = int(output_decomposed_str.get())
        for i in range(len(input_datas)):
            coef.append(np.zeros((pow(2,decomposed_level),1)))
    
            wpt = pywt.WaveletPacket(data = input_datas[i], maxlevel = decomposed_level, wavelet = 'db10', mode = 'symmetric')
            wp_node = [node.path for node in wpt.get_level(decomposed_level,'freq')]
            index = 0
            for i in range(len(wp_node)):
                reC.append(np.zeros((1,1)))
            new_wp = pywt.WaveletPacket(None,'db10')
            for i in wp_node:
                new_wp[i] = wpt[i]
                reC[index] = new_wp.reconstruct()
                index+=1
                new_wp = pywt.WaveletPacket(None, 'db10')
            reS.append(reC)
            reC=[]
                        #reC.append(np.zeros((len(wp_node), len(new_wp.reconstruct()))))
        temp_energy = []
        energy = []
        energy_ss = []
                #a=[]
                #for i in range(64):
                 #   a.append(np.zeros((1,1)))
                  #  a[i]=sum(map(lambda x:x*x, reS[7][i]))
                   # b = sum(a)

        for i in range(len(reS)):
            for j in range(len(wp_node)):
                temp_energy.append(np.zeros((1,1)))
                temp_energy[j] = sum(map(lambda x:x*x, reS[i][j]))
            energy.append(temp_energy)
            temp_energy = []
            energy_ss.append(np.zeros((1,1)))
        for i in range(len(energy)):
            energy_ss[i] = math.sqrt(sum(map(lambda x:x*x, energy[i])))
            energy[i] = [x / energy_ss[i] for x in energy[i]]
            
        label = int(output_label_str.get())
        for i in range(len(energy)):
            energy[i].append(label)
        with open('Energy_Spectrum_'+ output_label_str.get() + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(energy)):
                writer.writerow(energy[i])
        tk.messagebox.showinfo(title='WPT', message = 'Decomposition done !!')

tem_input_datas = []
input_datas = []
decomposed_level = 0

window = tk.Tk()
window.title('WPT App')
window.geometry('800x600')


menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff = 0)
menubar.add_cascade(label = 'File', menu = filemenu)
filemenu.add_command(label = 'Export', command = export_file)

title = tk.Label(window, text = 'Wavelet Packet Decomposition')
title.config(font = ('Times New Roman', 20))
title.pack()


folder_label = tk.Label(window, text = 'Folder Directory')
folder_label.config(font = ('Times New Roman', 12))
folder_label.pack()

tk.Button(window, text= 'Open', command = open_file).pack()

file_path_text = tk.StringVar(window)
tk.Label(window, textvariable = file_path_text).pack()

listbox = tk.Listbox(window, width = 80)
listbox.pack()
scroller = tk.Scrollbar(window)
#scroller.pack(side = tk.RIGHT, fill = tk.Y)
#scroller.config(command = listbox.yview)

output_frame = tk.Frame(window)
output_frame.pack(fill=tk.BOTH, pady=5)
output_label_label = tk.Label(window, text = 'Label').pack()
output_label_str=tk.StringVar()
output_label_entry = tk.Entry(window, textvariable=output_label_str).pack()
output_decomposed_label = tk.Label(window, text = 'Decomposition Level').pack()
output_decomposed_str = tk.StringVar()
output_decomposed_entry = tk.Entry(window, textvariable = output_decomposed_str).pack()

wpd_btn = tk.Button(window, text='Start', command = WPT, bg='red').pack()

window.config(menu = menubar)
window.mainloop()


"""
raw_data_direction = "C:/Users/allen/Desktop/Python Code/raw_data/20190927_台灣晶技_機台運轉行為/EXP001_研磨機_位置1/01_靜止"
tem_input_datas = []
input_datas = []
decomposed_level = 6
for file in os.listdir(raw_data_direction):
    tem_input_datas = list(csv.reader(open(raw_data_direction + "/" + file,'r')))
    del tem_input_datas[0]
    for i in range(len(tem_input_datas)):
        tem_input_datas[i].pop(0)
    tem_input_datas = list(itertools.chain.from_iterable(tem_input_datas))
    input_datas.append(tem_input_datas)

wp = 0
wp_node = []
############################
## Coefficient Initialize ##
coef=[]
reC = []
reS = []

for i in range(len(input_datas)):
    coef.append(np.zeros((pow(2,decomposed_level),1)))
    
    wpt = pywt.WaveletPacket(data = input_datas[i], maxlevel = decomposed_level, wavelet = 'db10', mode = 'symmetric')
    wp_node = [node.path for node in wpt.get_level(6,'freq')]
    index = 0
    for i in range(len(wp_node)):
        reC.append(np.zeros((1,1)))
    index = 0
    new_wp = pywt.WaveletPacket(None,'db10')
    for i in wp_node:
        new_wp[i] = wpt[i]
        reC[index] = new_wp.reconstruct()
        index+=1
        new_wp = pywt.WaveletPacket(None, 'db10')
    reS.append(reC)
    reC=[]
        #reC.append(np.zeros((len(wp_node), len(new_wp.reconstruct()))))
        
#print(reS[0][0])
    
        
        
        
#    for j in range(len(wp_node)):
#        coef[i][j] = wpt[wp_node[j]].data
    

###################
## Reconstuction ##
#for i in range(len(wp_node)):
#    coef[i]= wpt[wp_node[i]].data
#for i in range(len(wp_node)):
#    reC.append(np.zeros((1,1)))
#index = 0
#for i in wp_node:
#    new_wp[i] = wpt[i]
#    reC[index] = new_wp.reconstruct()
#    index+=1
#    new_wp = pywt.WaveletPacket(None, 'db10')

#####################
## Energy Spectrum ##
temp_energy = []
energy = []
energy_ss = []
a=[]
for i in range(64):
    a.append(np.zeros((1,1)))
    a[i]=sum(map(lambda x:x*x, reS[7][i]))
b = sum(a)

for i in range(len(reS)):
    for j in range(len(wp_node)):
        temp_energy.append(np.zeros((1,1)))
        temp_energy[j] = sum(map(lambda x:x*x, reS[i][j]))
    energy.append(temp_energy)
    temp_energy = []
    energy_ss.append(np.zeros((1,1)))
for i in range(len(energy)):
    energy_ss[i] = math.sqrt(sum(map(lambda x:x*x, energy[i])))
    energy[i] = [x / energy_ss[i] for x in energy[i]]

label = 1
for i in range(len(energy)):
    energy[i].append(label)




#A = new_wp.reconstruct(update = False)
#index=0
#for i in wp_node:
#    new_wp[i] = wpt[i]
#    reC[index] = new_wp.reconstruct(update = False)
#    index+=1
#reC = new_wp.reconstruct(update=False)

#arr, coeff_slices = pywt.coeffs_to_array((wpt[wp_node[0]].data).tolist())
#coeffs_from_arr = pywt.array_to_coeffs(arr,coeff_slices,output_format = 'wavedec')
#wpr = pywt.waverec(coeffs_from_arr, wavelet = 'db10')
"""