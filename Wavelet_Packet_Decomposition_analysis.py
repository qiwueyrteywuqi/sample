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
import matplotlib.pyplot as plt
import pandas as pd

def open_file():
    file_path = filedialog.askdirectory()
    if file_path:
        file_path_text.set(''+file_path)
        list_file(file_path)
    else:
        return None
    
    
def list_file(path):
    if path != '':
        listbox.delete(0,tk.END)
        list_files = os.listdir(path)
        for file in list_files:
            listbox.insert(tk.END, file)
    else:
        tk.messagebox.showerror(title = 'Oops', message = 'Please select a file path')

    
def export_file():
    return 0

def check_parameter(input_path):
    if input_path == '':
        tk.messagebox.showerror(title = 'Oops', message = 'Please select a file path')
    elif output_label_str.get() == '':
        tk.messagebox.showerror(title='Oops', message = 'Please enter label number')
        
    elif output_label_str.get():
        try:
            int(output_label_str.get())
        except ValueError:
            tk.messagebox.showerror(title='Oops', message = 'Please enter label number')
            return False
        else:
            if sampling_length_str.get() == '':
                tk.messagebox.showerror(title='Oops', message = 'Please enter sampling length')
                return False
            elif sampling_length_str.get():
                try:
                    int(sampling_length_str.get())
                except ValueError:
                    tk.messagebox.showerror(title='Oops', message = 'Please enter sampling length')
                    return False
                else:
                    if int(sampling_length_str.get()) < 2:
                        tk.messagebox.showerror(title='Oops', message = 'Please enter sampling length')
                        return False
                    elif output_decomposition_str.get() == '':
                        tk.messagebox.showerror(title='Oops', message = 'Please enter decomposition level number')
                    elif output_decomposition_str.get():
                        try:
                            int(output_decomposition_str.get())
                        except ValueError:
                            tk.messagebox.showerror(title='Oops', message = 'Please enter decomposition level number')
                            return False
                        else:
                            if int(output_decomposition_str.get()) < 0:
                                tk.messagebox.showerror(title='Oops', message = 'Please enter positive number')
                                return False
                            else:
                                return True
    else:
        return False
    
def enable_time_domain():    
    filemenu.entryconfig("Time domain plot", state = "normal")
    
def disable_time_domain():
    filemenu.entryconfig("Time domain plot", state = "disabled")

def time_domain_plot():
    try :
        
        label = output_label_str.get()
        plt.savefig("Time Domain" + label + ".jpg")
        plt.show(block = True)
        tk.messagebox.showinfo(title = "Time domain plot", message = "Image Save!!")
        disable_time_domain()
    except:
        tk.messagebox.showerror(title = "Oops", message = "Save failed")

def time_domain(data):
    time_series = pd.Series(data, index = np.arange(0, len(data)/10240, 1/10240))
    time_series = time_series.convert_objects(convert_numeric = True)
    plt.figure().set_size_inches(14, 10)
    time_series.plot()
    plt.title("Time domain",fontsize=18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Accelaration (g)")
    plt.xlabel('Time (s)')
    #plt.savefig("Time Domain.jpg")
    #plt.show()
    
    #print(time_series)
 
    
def data_sampling(length, data):
    section_data = []
    if length > (len(data) / 2):
        tk.messagebox.showerror(title= 'Oops', message = 'Your sampling length is out of range')
    else:
        if (len(data) / length) > int(len(data) / length):
            section_num = int(len(data) / length) + 1
        else:
            section_num = int(len(data) / length)
        for i in range(section_num):
            section_data.append( data[length*i:length*(i+1)] )
    
    return section_data
    
    

def WPT():
    path = file_path_text.get()
    if check_parameter(path):
        input_data = []
        sampling_length = int(sampling_length_str.get())
        files = os.listdir(path)
        
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                tem_input_data = list(csv.reader(open(path + "/" + file,'r')))
                del tem_input_data[0]
                for i in range(len(tem_input_data)):
                    tem_input_data[i].pop(0)
                tem_input_data = list(itertools.chain.from_iterable(tem_input_data))
                input_data.extend(tem_input_data)
                    #tk.messagebox.showerror(title = 'Oops', message = path)
                #wp = 0
        time_domain(input_data)
        
        sec_data = data_sampling(sampling_length, input_data)
############################
## Coefficient Initialize ##
        wp_node = []
        coef=[]
        reC = []
        reS = []
        decomposition_level = int(output_decomposition_str.get())
#####################################
## Decompostion and Reconstruction ##
        for i in range(len(sec_data)):
            coef.append(np.zeros((pow(2,decomposition_level),1)))

            wpt = pywt.WaveletPacket(data = sec_data[i], maxlevel = decomposition_level, wavelet = 'db10', mode = 'symmetric')
            wp_node = [node.path for node in wpt.get_level(decomposition_level,'freq')]
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
            
#################
## File output ##
        with open(output_DOE_str.get() + '_Energy_Spectrum_' + '0' + output_decomposition_str.get() + '_' + output_label_str.get() + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(energy)):
                writer.writerow(energy[i])
        tk.messagebox.showinfo(title='WPT', message = 'Decomposition done !!')
        enable_time_domain()
        

tem_input_data = []
input_data = []
decomposition_level = 0
image_plot = False

window = tk.Tk()
window.title('WPT App')
window.geometry('800x600')

menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff = 0)
menubar.add_cascade(label = 'File', menu = filemenu)
filemenu.add_command(label = 'Time domain plot', command = time_domain_plot)

input_frame = tk.Frame(window)
input_frame.pack()



title = tk.Label(input_frame, text = 'Wavelet Packet Decomposition')
title.config(font=('Lolita' , 20))
title.pack()


folder_label = tk.Label(input_frame, text = 'Folder Directory')
folder_label.config(font = ('Lolita', 12))
folder_label.pack()

open_dict = tk.Button(input_frame, text= 'Open', command = open_file)
open_dict.pack()

file_path_text = tk.StringVar(window)
tk.Label(input_frame, textvariable = file_path_text).pack()

listbox_frame = tk.Frame(window, bg = "bisque")
listbox_frame.pack()

listbox = tk.Listbox(listbox_frame, width = 80)
listbox.pack(side = tk.LEFT)
scroller = tk.Scrollbar(listbox_frame)
scroller.config(command = listbox.yview)
scroller.pack(side = tk.LEFT, fill = tk.Y)
listbox.config(yscrollcommand = scroller.set)


output_frame = tk.Frame(window, width=50, height=50)
output_frame.pack(pady = 10)

output_frame_left = tk.Frame(output_frame, width=50, height=50)
output_frame_left.pack(side = tk.LEFT, fill = tk.Y)

output_frame_right = tk.Frame(output_frame, width=50, height=50)
output_frame_right.pack(side = tk.RIGHT, fill = tk.Y, padx = 10)

#output_frame_bottom = tk.Frame(output_frame, width=50, height=50, background="#b15165")
#output_frame_bottom.pack(side = tk.BOTTOM, fill = tk.X)

output_DOE_label = tk.Label(output_frame_left, text = 'DOE')
output_DOE_label.config(font = ('Lolita', 12))
output_DOE_label.pack()
output_DOE_str=tk.StringVar()
output_DOE_entry = tk.Entry(output_frame_right, textvariable = output_DOE_str).pack(pady = 3)

output_label_label = tk.Label(output_frame_left, text = 'Label')
output_label_label.config(font = ('Lolita', 12))
output_label_label.pack()
output_label_str=tk.StringVar()
output_label_entry = tk.Entry(output_frame_right, textvariable=output_label_str).pack(pady = 3)


sampling_length_label = tk.Label(output_frame_left, text = 'Sampling Length')
sampling_length_label.config(font = ('Lolita', 12))
sampling_length_label.pack()
sampling_length_str = tk.StringVar()
sampling_length_entry = tk.Entry(output_frame_right, textvariable = sampling_length_str).pack(pady = 3.5)


output_decomposition_label = tk.Label(output_frame_left, text = 'Decomposition Level')
output_decomposition_label.config(font = ('Lolita', 12))
output_decomposition_label.pack()
output_decomposition_str = tk.StringVar()
output_decomposition_entry = tk.Entry(output_frame_right, textvariable = output_decomposition_str).pack(pady = 3.5)

output_button_frame = tk.Frame(window, width=50, height=50)
output_button_frame.pack()

wpd_btn = tk.Button(output_button_frame, text='Start', command = WPT, bg='red').pack(pady = 10)

window.config(menu = menubar)
disable_time_domain()



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