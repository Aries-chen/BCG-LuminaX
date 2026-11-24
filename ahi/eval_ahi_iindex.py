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
        return overlap / intersect > threshold

    
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

offset = 0
# step0
npz_root = '/nvme3/MOleni/preprocessed/bcg_heartbreath_and_snore'
csv_root = './eval_signal_save/ahi_hb_sn'
parallel_root = '/mnt/wujidata/datasets/parallel_data'
testset_csv_list = glob.glob(os.path.join(csv_root, '*[0-9]_test_predicted_ahi.csv')) # prediction interval
result_df = pd.DataFrame(columns=['edf_full', 'gt_anno_AHI', 'bcg_pred_AHI', 'bcg_precision', 'bcg_recall'])
#pdb.set_trace()
for testset_csv_path in testset_csv_list:
    
    npz_path = testset_csv_path.replace('_test_predicted_ahi.csv', '.npz')  # 为了得到 duration ,计算ahi的分母
    subject_id = os.path.basename(npz_path).split('.')[0] 
    npz_path = npz_root +'/'+ subject_id + '.npz'  # 为了得到 duration ,计算ahi的分母
    gt_csv_path = os.path.join(parallel_root, subject_id, 'standard_AH_events.csv') # 得到gt_interval
    
    
    
    # step1
    duration = np.load(npz_path)['duration']
    hours = duration / 3600 # 分母
    #gt_csv_path = testset_csv_path.replace('_predicted_ahi.csv', '_halfwindowoffset_predicted_ahi.csv') # 得到gt_interval
    # 计算ahi
    gt_interval = pd.read_csv(gt_csv_path)
    gt_ahi = len(gt_interval) / hours
    pred_interval = pd.read_csv(testset_csv_path)
    pred_ahi = len(pred_interval) / hours
    # 计算event level precision recall
    gt_interval['interval'] = gt_interval.apply(lambda row: [row['Start'], row['Start'] + row['Duration']], axis=1)
    gt_seg = gt_interval['interval'].tolist()
    pred_interval['interval'] = pred_interval.apply(lambda row: [row['Start'], row['Start'] + row['Duration']], axis=1)
    pred_seg = pred_interval['interval'].tolist()
    TP, FP, FN = cal_metrics(gt_seg, pred_seg)
    bcg_precision = TP / (TP + FP + 1e-6)
    bcg_recall = TP / (TP + FN + 1e-6)


    
    
    # step2
    result_df = result_df._append({'edf_full': subject_id, 'gt_anno_AHI': gt_ahi, 'bcg_pred_AHI': pred_ahi, 'bcg_precision': bcg_precision, 'bcg_recall': bcg_recall}, ignore_index=True)

# step3

num_result = result_df.shape[0]
print(f"Total valid result is {num_result}")
bcg_precision = 0
bcg_recall = 0
bcg_f1_score = 0
total_gt_anno_AHI = result_df['gt_anno_AHI'].sum()
total_pred_AHI = result_df['bcg_pred_AHI'].sum()
gt_anno_AHI_list = result_df['gt_anno_AHI'].tolist()
pred_AHI_list = result_df['bcg_pred_AHI'].tolist()
#pdb.set_trace()
MAE_AHI = sum([abs(gt_AHI - pred_AHI) for gt_AHI, pred_AHI in zip(gt_anno_AHI_list, pred_AHI_list)]) / len(gt_anno_AHI_list)
print(f'MAE_AHI is {MAE_AHI}')
for index, row in result_df.iterrows():
    weight_score_recall = row['gt_anno_AHI'] / total_gt_anno_AHI
    weight_score_precision = row['bcg_pred_AHI'] / total_pred_AHI
    
    bcg_precision += row['bcg_precision'] * weight_score_precision
    bcg_recall += row['bcg_recall'] * weight_score_recall
    bcg_f1_score = bcg_precision * bcg_recall * 2 / (bcg_precision + bcg_recall)
    

print('bcg_precision:', bcg_precision)
print('bcg_recall:', bcg_recall)
print('bcg_f1_score:', bcg_f1_score)

predicted_ahi = []
actual_ahi = []
valid_count = 0
for index, row in  result_df.iterrows():
    if row['bcg_recall'] != 0 and row['bcg_precision'] != 0:
        predicted_ahi.append(row['bcg_pred_AHI'])
        actual_ahi.append(row['gt_anno_AHI'])
        valid_count = valid_count + 1
    else:
        # print('Except User', row['edf_full'])
        predicted_ahi.append(row['bcg_pred_AHI'])
        actual_ahi.append(row['gt_anno_AHI'])

evaluator = NightlySleepMetricsEvaluator(actual_ahi, predicted_ahi)
evaluator.bland_altman_plot()
print('Pearson:', evaluator.pearsonr())
plt.scatter(actual_ahi, predicted_ahi)
plt.xlabel('Ground Truth AHI')
plt.ylabel('Predicted AHI')
plt.title('Pearson Correlation Coefficient: ' + str(evaluator.pearsonr()))
# plot x=y line， max value is max of x and y
x = max(max(actual_ahi), max(predicted_ahi))
plt.plot([0, x], [0, x], color='red')
plt.savefig('ahi.png')

thresholds = [5, 15, 30]
for threshold in thresholds:
    screening_metrics = evaluator.screen_OSA(threshold)
    print(f"Threshold: {threshold}")
    print(screening_metrics)
# print(f'Pass Count is {pass_count}')



    