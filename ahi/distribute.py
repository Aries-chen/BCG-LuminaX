import numpy as np
import os
from scipy.signal import  lfilter, firwin
from scipy.signal import decimate
import pdb
import matplotlib.pyplot as plt



i = 0
fd_hb = 0.1
fu_hb = 25 
fd_snore = 30
fu_snore = 200 
fs = 500


down_sample_hb = 50
down_sample_snore = 200


parallel_root = '/mnt/wujidata/datasets/parallel_data'



def bandpass_filter(signal,fs,fd,fu):
    width = 1  # 过渡带宽
    numtaps = round_to_odd((6.6*0.5/width)*fs)  # 滤波器阶数
    
    # 带通滤波器的参数
    passband = [fd,fu]
    
    # 获取滤波器参数
    taps_bandpass = firwin(numtaps, passband, window='hamming', pass_zero='bandpass', scale=False, fs=fs)
    
    # 使用lfilter滤波
    filtered_signal = lfilter(taps_bandpass, [1], signal)
    # 补偿滤波器引起的时延
    delay = (numtaps - 1) // 2
    filtered_signal = np.roll(filtered_signal, -int(delay))
    return filtered_signal

def round_to_odd(f):
    return int(round(f)) // 2 * 2 + 1

# 预加重滤波
def pre_em(sig,coeff=0.97):
    return np.append(sig[0],sig[1:]-coeff*sig[:-1])

def de_overhigh(sig):
    mean = np.mean(np.abs(sig))
    sig[np.abs(sig)/mean > 1000] = 0

def bcg_filter(bcg_sig,bcg_fs,fd,fu):
    signal = bcg_sig.copy()
    de_overhigh(signal)
    signal = bandpass_filter(signal,bcg_fs,fd,fu)
    signal = pre_em(signal)
    return signal



def downsample(signal, original_sr, new_sr):

# 计算降采样因子
    downsample_factor = int(round(original_sr / new_sr))
# 进行降采样
    downsampled_signal = decimate(signal, downsample_factor)

    return downsampled_signal

def preprocess(signal, fd, fu, down_sample):
    
    bcg_fs = 500
    signal = bcg_filter(signal,bcg_fs,fd,fu)
    signal = downsample(signal,fs,down_sample)

    return signal






def filter_function(signal):
    fd_hb = 0.1
    fu_hb = 25 
    fd_snore = 30
    fu_snore = 200 
    fs = 500


    down_sample_hb = 50
    down_sample_snore = 200
    
    signal_hb    = preprocess(signal,fd_hb,fu_hb,down_sample_hb)
    signal_snore = preprocess(signal,fd_snore,fu_snore,down_sample_snore)

    return signal_hb, signal_snore

# 用于存储找到的.npz文件路径的列表
npz_files = []

# 遍历根目录下的所有文件夹
for folder_name in os.listdir(parallel_root):
    # 检查文件夹名称是否以数字结尾
    if folder_name[-1].isdigit():
        # 构建完整的文件夹路径
        folder_path = os.path.join(parallel_root, folder_name)
        # 检查这个路径是否确实是一个文件夹
        if os.path.isdir(folder_path):
            # 构建bcg.npz文件的完整路径a
            npz_file_path = os.path.join(folder_path, 'bcg_feature_raw.npz')
            # 检查文件是否存在
            if os.path.isfile(npz_file_path):
                # 如果文件存在，添加到列表中
                npz_files.append(npz_file_path)


for npz_path in npz_files:
    #pdb.set_trace()
    data = np.load(npz_path)
    # 假设npz文件中只有一个数组，我们将其命名为'data'
    signal_data = data['raw']  # 请根据实际情况调整键名
    
    # 应用带通滤波器
    signal_hb, signal_snore = filter_function(signal_data)
    # 构建输出文件的路径
    last_slash_index = npz_path.rfind('/')
    extracted_path = npz_path[:last_slash_index+1]
    output_file_path1 = os.path.join(extracted_path, 'bcg_heartbreath.npz')
    output_file_path2 = os.path.join(extracted_path, 'bcg_snore.npz')
    # 保存过滤后的数据到新的npz文件
    np.savez(output_file_path1, data=signal_hb, fs = down_sample_hb)
    np.savez(output_file_path2, data=signal_snore, fs = down_sample_snore)
    i+=1
    print(i)

    # 关闭npz文件
    data.close() 
