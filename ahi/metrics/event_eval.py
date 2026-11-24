import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from wuji import sed_eval
from dcase_util.containers import MetaDataContainer
import pdb
import soundfile as sf

MAX_SNORE_TIME = 2.5
MIN_SNORE_TIME = 0.3
MIN_SNORE_INTERVAL = 0.3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cal_segment(mel_length, label, net_output,ratio = 0.5, anchor = 2):
    time_frames = np.linspace(0, mel_length, net_output.shape[0] + 1)
    gt_seg = []
    pre_seg = []
    for i in range(label.shape[0]):
        for j in range(anchor):
            if label[i, j*3] > 0:
                start_time = label[i, j+1] * (time_frames[i+1] - time_frames[i]) + time_frames[i]
                stop_time = label[i, j+2] * (time_frames[i+1] - time_frames[i]) + time_frames[i]
                gt_seg.append([start_time, stop_time])
            if sigmoid(net_output[i, j*3]) > ratio:
                start_time = net_output[i, j*3+1] * (time_frames[i+1] - time_frames[i]) + time_frames[i]
                stop_time = net_output[i, j*3+2] * (time_frames[i+1] - time_frames[i]) + time_frames[i]
                pre_seg.append([start_time, stop_time])
    return gt_seg, pre_seg

def merge_intervals(intervals , tolerance=2):  #这个超参数tolerance，是用来合并两个时间段的，如果两个时间段的间隔小于tolerance，那么就合并。这里的合并算法有待优化
    # Sort the intervals by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals:
        previous = merged[-1]
        if current[0] <= previous[1] + tolerance:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged



def cal_batch_segs(label, batch_idx, net_output, input_shape,threshold = 0.5):
    # b, mel, l = input_shape
    b, l = input_shape
    l = 300
    batch_gt = []
    batch_pred = []
    #pdb.set_trace()
    for i in range(b):
        #if label.detach().cpu().numpy()[:,:,0].max(axis = 1)[i] == 0:
        #    continue
        gt_seg, pre_seg = cal_segment(l, label[i].float().detach().cpu().numpy(), net_output['x'][0][i].float().detach().cpu().numpy(), ratio=threshold)
        gt_seg = merge_intervals(gt_seg) if gt_seg else []
        pre_seg = merge_intervals(pre_seg) if pre_seg else []
        #pre_seg = merge_intervals(pre_seg) if pre_seg and sigmoid(net_output['x'][1][i].detach().cpu().numpy()) > 0.5 else []
        gt_seg = MetaDataContainer([
            {'filename': str(batch_idx)+'-'+str(i)+'.wav', 'event_label': 'snore', 'onset': float(start), 'offset': float(end)}
            for start, end in gt_seg
        ])
        pre_seg = MetaDataContainer([
            {'filename': str(batch_idx)+'-'+str(i)+'.wav', 'event_label': 'snore', 'onset': float(start), 'offset': float(end)}
            for start, end in pre_seg
        ])
        batch_gt.append(gt_seg)
        batch_pred.append(pre_seg)
    if len(pre_seg) or len(gt_seg):
        pdb.set_trace()
    return batch_gt, batch_pred

output_path = '/nvme2/wangwangxin/datasets/raw_snore/bad_case/'

def cal_confusion_matrix(label, pred, input):
    cm = confusion_matrix(label, pred, labels=[0, 1])
    '''
    for i in range(len(label)):
        if label[i] == 0 and pred[i] == 1:
            sf.write(output_path+'false_positive_'+str(i)+'.wav', input[i].detach().cpu().numpy(), 8000)
        elif label[i] == 1 and pred[i] == 0:
            sf.write(output_path+'false_negative_'+str(i)+'.wav', input[i].detach().cpu().numpy(), 8000)
        elif label[i] == 1 and pred[i] == 1:
            sf.write(output_path+'true_positive_'+str(i)+'.wav', input[i].detach().cpu().numpy(), 8000)
        elif label[i] == 0 and pred[i] == 0:
            sf.write(output_path+'true_negative_'+str(i)+'.wav', input[i].detach().cpu().numpy(), 8000)
    pdb.set_trace()
    '''
    return cm

def cal_cls_confusion_matrix(label, pred, input):
    cm = confusion_matrix(label, np.round(sigmoid(pred)), labels=[0, 1])
    '''
    for i in range(len(label)):
        if label[i] == 0 and np.round(sigmoid(pred))[i] == 1:
            sf.write(output_path+'false_positive_'+str(i)+'.wav', input[i].detach().cpu().numpy(), 8000)
        elif label[i] == 1 and np.round(sigmoid(pred))[i] == 0:
            sf.write(output_path+'false_negative_'+str(i)+'.wav', input[i].detach().cpu().numpy(), 8000)
        elif label[i] == 1 and np.round(sigmoid(pred))[i] == 1:
            sf.write(output_path+'true_positive_'+str(i)+'.wav', input[i].detach().cpu().numpy(), 8000)
        elif label[i] == 0 and np.round(sigmoid(pred))[i] == 0:
            sf.write(output_path+'true_negative_'+str(i)+'.wav', input[i].detach().cpu().numpy(), 8000)
    pdb.set_trace()
    '''
    
    return cm
    

def save_test_result(path, gts, preds):
    save_path = path.replace('instance', 'test_results')
    np.savez(save_path, gts=gts, preds = preds)
            



def cal_batch_cls_segs(label,net_output, batch_idx, id):
    # b, mel, l = input_shape
    if len(net_output['x'][0].shape) == 3:
        b, l ,c = net_output['x'][0].shape
    else:
        b, l = net_output['x'][0].shape
        c = 2
    #seg_cls = net_output['seg_cls'].detach().cpu().numpy()
    #pdb.set_trace()
    batch_gt = []
    batch_pred = []
    for i in range(b):
        if c > 2:
            gt_seg = cal_cls_segment_multi(label[i].detach().cpu().numpy(),interval = 1, cls_num = c)
            pre_seg = cal_cls_segment_multi(np.argmax(net_output['x'][0][i].detach().cpu().numpy(),axis=1),interval = 1, cls_num = c)
            gt_seg_dict = {}
            pre_seg_dict = {}
            for j in range(1, c):
                gt_seg[j] = merge_intervals(gt_seg[j]) if gt_seg[j] else []
                pre_seg[j] = merge_intervals(pre_seg[j]) if pre_seg[j] else []
                gt_seg[j] = MetaDataContainer([
                    {'filename': id[i] +'.wav', 'event_label': f'{j}', 'onset': float(start), 'offset': float(end)}
                    for start, end in gt_seg[j]
                ])
                pre_seg[j] = MetaDataContainer([
                    {'filename': id[i]+'.wav', 'event_label': f'{j}', 'onset': float(start), 'offset': float(end)}
                    for start, end in pre_seg[j]
                ])
                gt_seg_dict[j] = gt_seg[j]
                pre_seg_dict[j] = pre_seg[j]
                
            batch_gt.append(gt_seg_dict)
            batch_pred.append(pre_seg_dict)
        else:
            gt_seg= cal_cls_segment(label[i].detach().cpu().numpy(),interval = 1)
            #pre_seg = cal_cls_segment(np.argmax(net_output['x'][0][i].detach().cpu().numpy(),axis=1))
            pre_seg = cal_cls_segment(np.where(net_output['x'][0][i].detach().cpu().numpy()>0 ,1, 0),interval = 1)
            gt_seg = merge_intervals(gt_seg) if gt_seg else []
            pre_seg = merge_intervals(pre_seg) if pre_seg else []
            # 去掉不符合长度的seg
            pre_seg = [seg for seg in pre_seg if seg[1] - seg[0] > 10]
            #pre_seg = merge_intervals(pre_seg) if (pre_seg and seg_cls[i,1] > seg_cls[i, 0])else []
            gt_seg = MetaDataContainer([
                {'filename': id[i] +'.wav', 'event_label': 'snore', 'onset': float(start), 'offset': float(end)}
                for start, end in gt_seg
            ])
            pre_seg = MetaDataContainer([
                {'filename': id[i]+'.wav', 'event_label': 'snore', 'onset': float(start), 'offset': float(end)}
                for start, end in pre_seg
            ])
            batch_gt.append(gt_seg)
            batch_pred.append(pre_seg)
        
    return batch_gt, batch_pred

def cal_cls_segment(cls_interval, interval = 1):
    # calculate the snore segment from the cls_interval ，interval = 0.01s
    snore_interval = []
    start = 0
    end = 0
    for i in range(len(cls_interval)):
        if cls_interval[i] == 1:
            if start == 0:
                start = i * interval
            end = i * interval
        else:
            if start != 0:
                snore_interval.append([start, end])
                start = 0
                end = 0
    return snore_interval

def cal_cls_segment_multi(cls_interval, interval = 1, cls_num = 3):
    # calculate the snore segment from the cls_interval ，interval = 0.01s
    result = {}
    start = 0
    end = 0
    for j in range(1, cls_num):
        result[j] = []
        for i in range(len(cls_interval)):
            if cls_interval[i] == j:
                if start == 0:
                    start = i * interval
                end = i * interval
            else:
                if start != 0:
                    result[j].append([start, end])
                    start = 0
                    end = 0
    return result


