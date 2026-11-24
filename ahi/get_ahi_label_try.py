import torch
#from pytorch_lightning import Trainer
#from models.snore import YOHOEncoder, YOHOConfig
#from tasks.snore import SnoreTask
import pandas as pd
import sys
from wuji_dl.factory import create_instance
from omegaconf import DictConfig, OmegaConf
import typing as t
from scipy.io import wavfile
import librosa
import numpy as np
import glob
import os
from itertools import islice
import torch.nn as nn
import noisereduce as nr
import pdb

segment_length = 5*60
hop_length = 60
bcg_sr = 500
snore_sr = 250
heartbreath_sr = 50
nasal_sr = 20



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def merge_intervals(intervals , tolerance=2):  #这个超参数tolerance，是用来合并两个时间段的，如果两个时间段的间隔小于tolerance，那么就合并。这里的合并算法有待优化
    # Sort the intervals by start time
    #pdb.set_trace()
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current in intervals:
        previous = merged[-1]
        if current[0] <= previous[0] + previous[1] + tolerance:
            previous[1] = max(previous[0] + previous[1],current[0] + current[1]) - previous[0]
        else:
            merged.append(current)

    return merged

def _parse_hierarchy(path: str):
    config = t.cast(DictConfig, OmegaConf.load(path))
    bases = [_parse_hierarchy(base) for base in config.pop("_base", [])]
    return t.cast(DictConfig, OmegaConf.merge(*bases, config))


def _resolve_dict_config(path: str):
    config = OmegaConf.to_container(_parse_hierarchy(path), resolve=True)
    return t.cast(t.Dict[str, t.Any], config)

def set_config(config: str) -> None:
    config = _resolve_dict_config(config)
    return config

def time_to_seconds(time_str, sr = 8000):
    parts = list(map(float, time_str.split(':')))
    if len(parts) == 3:
        return (parts[0] * 60 * 60 + parts[1] * 60 + parts[2]) * sr
    else:
        return (parts[0] * 60 + parts[1]) * sr

def batch2label(batch_output, input, Start, ratio = 0.56):
    b = input.shape[0] # l = 8000*60
    l = segment_length
    batch_pre_seg = []
    batch_output = batch_output.detach().cpu().numpy()
    time_frames = np.linspace(0, l, batch_output.shape[1] + 1)
    for j in range(b):
        pre_seg = []
        for i in range(batch_output[j].shape[0]//5*2, batch_output[j].shape[0]//5*3):
            if sigmoid(batch_output[j,i]) > ratio:
                start_time = time_frames[i]
                stop_time = time_frames[i+1]
                pre_seg.append([start_time, stop_time, sigmoid(batch_output[j,i])]) 
        pre_seg = [[Start + j * hop_length + seg[0]/l * segment_length, seg[1] / l * segment_length - seg[0] / l * segment_length, seg[-1]]  for seg in pre_seg]
        batch_pre_seg += pre_seg
    return batch_pre_seg


def batch2label1(batch_output, input, Start, ratio = 0.56):
    b = input.shape[0] # l = 8000*60
    l = segment_length
    batch_pre_seg = []
    batch_output = batch_output[0].detach().cpu().numpy()
    #pdb.set_trace()
    time_frames = np.linspace(0, l, batch_output.shape[1] + 1)
    for j in range(b):
        pre_seg = []
        for i in range(batch_output[j].shape[0]//5*2, batch_output[j].shape[0]//5*3):
            if sigmoid(batch_output[j,i]) > ratio:
                start_time = time_frames[i]
                stop_time = time_frames[i+1]
                pre_seg.append([start_time, stop_time, sigmoid(batch_output[j,i])]) 
        pre_seg = [[Start + j * hop_length + seg[0]/l * segment_length, seg[1] / l * segment_length - seg[0] / l * segment_length, seg[-1]]  for seg in pre_seg]
        batch_pre_seg += pre_seg
    return batch_pre_seg

def remove_outer_model_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # 分割键名
        parts = k.split('.')
        # 如果第一个部分是'model'，则去掉它
        if parts[0] == 'model':
            new_key = '.'.join(parts[1:])
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def slice_number(num: int, slice_size: int) -> np.ndarray:
    # 计算可以整除slice_size的组数和剩余的部分
    full_slices = num // slice_size
    remainder = num % slice_size
    
    # 创建一个切片数组，除了最后一个切片是remainder，其他都是slice_size
    slices = [slice_size] * full_slices + ([remainder] if remainder > 0 else [])
    
    # 返回NumPy数组
    return np.array(slices)

def main(config_path, data_paths, checkpoint_path):
    # 加载配置
    
    #pdb.set_trace()
    config = set_config(config_path)
    # 初始化模型
    device = torch.device("cuda:2")
    #model = YOHOEncoder.build_model(YOHOConfig(**config['task']['model']),None).to(device)
    model = create_instance(nn.Module, config['task']['model']).to(device)
    # Load the checkpoint
    
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Remove "model." prefix from state_dict keys
    new_state_dict = remove_outer_model_prefix(ckpt['state_dict'])
    # Modify key names
    # Load the state dict into the model
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    off_set = 0
    
    with torch.no_grad():    
    
        for data_path in data_paths:
            df_labels = pd.DataFrame(columns=[ 'Type', 'Start', 'Duration'])
            data = np.load(data_path)
            #pdb.set_trace()
            heartbreath = data['heartbreath']
            snore = data['snore']
            duration = data['duration']

        
            #breath = data['breath']
            #body_movement = data['body_movement']
            #nasal_pressure = data['nasal_pressure']
            #nasal_thermo = data['nasal_thermo']

            #pdb.set_trace()
            # 提取音频信号和采样率
            #audio, sr = librosa.load(audio_path, sr=None)
            #audio = audio[int(segment_length*sr*off_set):]
            
            batch = {}
            #bcg_segs  = []
            heartbreath_segs = []
            snore_segs = []
            lengths = []
            #body_movement_segs = []
            #nasal_pressure_segs = []
            #nasal_thermo_segs = []
            num_segments = (len(heartbreath) - off_set*heartbreath_sr - heartbreath_sr * segment_length) // (heartbreath_sr * hop_length) + 1
            
            if num_segments == 0:
                print(f'{data_path} is too short or something')
                continue
            for i in range(num_segments):
                # Extract the bcg segment
                #bcg_seg = bcg_sig[i * bcg_sr * hop_length + off_set*bcg_sr: i * bcg_sr * hop_length + bcg_sr * segment_length + off_set*bcg_sr]
                #bcg_segs.append(bcg_seg)
                heartbreath_seg = heartbreath[i * heartbreath_sr * hop_length: i * heartbreath_sr * hop_length + heartbreath_sr * segment_length]
                heartbreath_segs.append(heartbreath_seg)
                snore_seg = snore[i * snore_sr * hop_length: i * snore_sr * hop_length + snore_sr * segment_length]
                snore_segs.append(snore_seg)
                length = heartbreath_seg.shape[0]//heartbreath_sr
                lengths.append(length)

                #body_movement_seg = body_movement[i * breath_sr * hop_length: i * breath_sr * hop_length + breath_sr * segment_length]
                #body_movement_segs.append(body_movement_seg)
                #nasal_pressure_seg = nasal_pressure[i * nasal_sr * hop_length: i * nasal_sr * hop_length + nasal_sr * segment_length]
                #nasal_pressure_segs.append(nasal_pressure_seg)
                #nasal_thermo_seg = nasal_thermo[i * nasal_sr * hop_length: i * nasal_sr * hop_length + nasal_sr * segment_length]
                #nasal_thermo_segs.append(nasal_thermo_seg)
            #inputs_bcg = torch.stack([torch.from_numpy(bcg.copy()).float() for bcg in bcg_segs])
            inputs_heartbreath = torch.stack([torch.from_numpy(heartbreath.copy()).float() for heartbreath in heartbreath_segs])
            inputs_snore = torch.stack([torch.from_numpy(snore.copy()).float() for snore in snore_segs])
            inputs_length= torch.tensor(lengths, dtype=torch.int64)
            #pdb.set_trace()
            #inputs_body_movement = torch.stack([torch.from_numpy(body_movement.copy()).float() for body_movement in body_movement_segs])
            #inputs_nasal_pressure = torch.stack([torch.from_numpy(nasal_pressure.copy()).float() for nasal_pressure in nasal_pressure_segs])
            #inputs_nasal_thermo = torch.stack([torch.from_numpy(nasal_thermo.copy()).float() for nasal_thermo in nasal_thermo_segs])
            bs = 256
            pre_seg = []
            for i in range(len(heartbreath_segs)//bs):
                #batch['source'] = inputs[i*bs: (i+1)*bs,:].to(device)
                #batch['bcg_raw'] = inputs_bcg[i*bs: (i+1)*bs,:].to(device)
                batch['length'] = inputs_length[i*bs: (i+1)*bs].to(device) 
                batch['snore'] = inputs_snore[i*bs: (i+1)*bs,:].to(device)               
                batch['heartbreath'] = inputs_heartbreath[i*bs: (i+1)*bs,:].to(device)
                #batch['body_movement'] = inputs_body_movement[i*bs: (i+1)*bs,:].to(device)
                '''
                batch['nasal_thermo'] = inputs_nasal_thermo[i*bs: (i+1)*bs,:].to(device)
                batch['nasal_pressure'] = inputs_nasal_pressure[i*bs: (i+1)*bs,:].to(device)
                '''
                
                batch_output = model(**batch)['ahi_event']
                #pdb.set_trace()
                batch_pre_seg = batch2label(batch_output, batch['heartbreath'], i*bs*hop_length+off_set)
                pre_seg += batch_pre_seg
            #pdb.set_trace()
            pre_seg = merge_intervals(pre_seg) if pre_seg else []
            df_temp = pd.DataFrame(pre_seg, columns=['Start', 'Duration', 'Confidence'])
            # Add the other columns
            df_temp['Type'] = 'AHI'
            # df_temp.pop('Confidence')
            # Move 'Name' column to the front
            Type = df_temp.pop('Type')
            df_temp.insert(0, 'Type', Type)
            df_labels = df_labels._append(df_temp, ignore_index = True)
            #去掉Duration小于10秒的
            df_labels = df_labels[df_labels['Duration'] > 10]
            # Save the labels to a CSV file
            #df_labels = post_labeling(df_labels, audio)
            csv_path = data_path.replace('.npz','_test_predicted_ahi.csv')
            df_labels.to_csv('./eval_signal_save/ahi_hb_sn/'+csv_path.split('/')[-1], index=False)


if __name__ == "__main__":
    #config_path = 'configs/bcg_ahi/exp_aux_feature.yaml'
    config_path = '/nvme3/wyc/ahi/configs/ahi/beta.yaml'
    # 获取所有YOHO模型的文件路径
    #yoho_models = glob.glob('log/snore/exp_yoho_audio-fit-*/checkpoints/*.ckpt')

    # 按照文件的修改时间排序，最新的文件在最后
    #yoho_models.sort(key=os.path.getmtime)

    # 获取最新的YOHO模型的文件路径
    #latest_yoho_model = 'log/bcg_ahi/exp_aux_feature-fit-42-1111-113202/checkpoints/epoch=13-step=1918.ckpt'#yoho_models[-1]
    latest_TUlite_model = '/nvme3/wyc/ahi/log/ah-events/beta-fit-42-1203-173957/checkpoints/epoch=21-step=5082.ckpt'#yoho_models[-1]
    checkpoint_path = latest_TUlite_model

    #id_roots = ['/local_data/wangwangxin/data_process/online/audio_1387009_2024-08-16 00_00_00_2024-08-16 12_00_00.wav']
    
    
    index_path = '/nvme3/wyc/ahi/index/ahi/bcg_ahi.csv'
    #index_path = 'index/ahi/aux_feature_filtered_index.csv'
    index_info = pd.read_csv(index_path)
    # 把split是test的数据的path取出变成一个list
    id_roots = index_info[index_info['split'] == 'val']['path'].tolist()
    #id_roots = index_info['path'].tolist()
    main(config_path, id_roots, checkpoint_path)
