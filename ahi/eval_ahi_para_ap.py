# compares the ground truth AHI index with the predicted AHI index
# step0 : read the gt_ahi and pred_ahi from the csv file, including the duration of the recording
# step1 : calculate the AHI index and event level precision recall for each subject, using intervals
# step2 : process interval index into [subject_name, gt_ahi, pred_ahi]
# step3 : calculate the correlation between the gt_ahi and pred_ahi
# input:
import pdb
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from wuji.Evaluator.NightlySleepMetrics import NightlySleepMetricsEvaluator

def match(interval1, inteval2, threshold=0.5):
    if interval1[0] > inteval2[1] or interval1[1] < inteval2[0]:
        return False
    else:
        overlap = min(interval1[1], inteval2[1]) - max(interval1[0], inteval2[0])
        intersect = max(interval1[1], inteval2[1]) - min(interval1[0], inteval2[0])
        if overlap / intersect > threshold:
            return True
        else:
            return False
    
def cal_metrics(gt_seg, pre_seg):
    TP = FP = FN = 0
    g = len(gt_seg)
    p = len(pre_seg)
    i = j = 0
    while i < g and j < p:
        if match(gt_seg[i], pre_seg[j]):
            TP += 1
            i += 1
            j += 1
        elif gt_seg[i][0] < pre_seg[j][0]:
            FN += 1
            i += 1
        else:
            FP += 1
            j += 1
    while i < g:
        FN += 1
        i += 1
    while j < p:
        FP += 1
        j += 1
    return TP, FP, FN

def merge_intervals_df(df, tolerance=3):
    if df.empty:
        return df
    df = df.sort_values(by='Start')
    merged = []
    current_start, current_end = df.iloc[0]['Start'], df.iloc[0]['Start'] + df.iloc[0]['Duration']
    for i in range(1, len(df)):
        start, end = df.iloc[i]['Start'], df.iloc[i]['Start'] + df.iloc[i]['Duration']
        if start <= current_end + tolerance:
            current_end = max(current_end, end)
        else:
            merged.append([current_start, current_end - current_start])
            current_start, current_end = start, end
    merged.append([current_start, current_end - current_start])
    return pd.DataFrame(merged, columns=['Start', 'Duration'])

def merge_intervals(intervals , tolerance=3):  #这个超参数tolerance，是用来合并两个时间段的，如果两个时间段的间隔小于tolerance，那么就合并。这里的合并算法有待优化
    # Sort the intervals by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals:
        previous = merged[-1]
        if current[0] <= previous[0] + previous[1] + tolerance:
            previous[1] = max(previous[0] + previous[1],current[0] + current[1]) - previous[0]
        else:
            merged.append(current)
    return merged

offset = 0
# step0
npz_root = '/nvme3/MOleni/preprocessed/bcg_heartbreath_and_snore/'
csv_root = './eval_signal_save/ahi_hb_sn'
parallel_root = '/mnt/wujidata/datasets/parallel_data'
testset_csv_list = glob.glob(os.path.join(csv_root, '*[0-9]_ahi_confidence_quality.csv')) # prediction interval
#pdb.set_trace()
AP_curve = []
# 从 0.1遍历到0.98
for threshold in np.arange(0.1, 0.99, 0.01):
    TP = FP = FN = 0
    for testset_csv_path in testset_csv_list:
        npz_path = testset_csv_path.replace('_ahi_confidence_quality.csv', '.npz')  # 为了得到 duration ,计算ahi的分母
        subject_id = os.path.basename(npz_path).split('.')[0] 
        gt_csv_path = os.path.join(parallel_root, subject_id, 'correct_AH_events.csv') # 得到gt_interval
        #gt_arousal_csv_path = os.path.join(parallel_root, subject_id, 'arousal_events.csv') # 得到gt_interval
        try:
            #去除开头和结尾的offset
            gt_interval = pd.read_csv(gt_csv_path)
            #gt_arousal = pd.read_csv(gt_arousal_csv_path)
            pred_interval = pd.read_csv(testset_csv_path)
            pred_interval = pred_interval[pred_interval['Confidence'] > threshold]
            #pred_interval = merge_intervals_df(pred_interval)
            
        except:
            print(f"Error: {subject_id}")
            continue
        gt_interval['interval'] = gt_interval.apply(lambda row: [row['Start'], row['Start'] + row['Duration']], axis=1)
        gt_seg = gt_interval['interval'].tolist()
        pred_interval['interval'] = pred_interval.apply(lambda row: [row['Start'], row['Duration']], axis=1)
        pred_seg = pred_interval['interval'].tolist() if len(pred_interval) > 0 else []
        pred_seg = merge_intervals(pred_seg) if len(pred_seg) > 0 else []
        pred_seg = [seg for seg in pred_seg if seg[1] > 10] # remove short intervals
        pred_seg = [[seg[0], seg[1] + seg[0]] for seg in pred_seg]
        _TP, _FP, _FN = cal_metrics(gt_seg, pred_seg)
        TP += _TP
        FP += _FP
        FN += _FN
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    AP_curve.append([precision, recall, threshold])
    print(f"threshold: {threshold}, precision: {precision}, recall: {recall}")

AP_curve = np.array(AP_curve)
plt.plot(AP_curve[:, 1], AP_curve[:, 0])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.savefig('precision_recall_curve.png')

# Calculate AP score, 11 points
AP = 0
for i in np.arange(0.1, 1.01, 0.01):
    recall = i
    precision = np.max(AP_curve[AP_curve[:, 1] >= recall, 0]) if np.sum(AP_curve[:, 1] >= recall) > 0 else 0
    AP += precision
AP /= 90
print(f"AP score: {AP}")  # AP score





