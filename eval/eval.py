import sys, time, os
import numpy as np
import pandas as pd
import csv


def get_results(pred, r1, r2):
    pred_b = []
    gt_b = []
    pred_w = []
    gt_w = []


    for i in range(len(race)):
        if race[i] == r1:
            pred_b.append(pred[i])
            gt_b.append(gt_loan[i])
        elif race[i] == r2:
            pred_w.append(pred[i])
            gt_w.append(gt_loan[i])
        else: continue

    n_b = len(gt_b)
    tp_b = 1
    fp_b = 1
    tn_b = 1
    fn_b = 1
    for i in range(n_b):
        if gt_b[i] == 1 and pred_b[i] == 1:
            tp_b+=1
        elif gt_b[i] == 0 and pred_b[i] == 1:
            fp_b+=1
        elif gt_b[i] == 0 and pred_b[i] == 0:
            tn_b+=1
        else: fn_b +=1

    n_w = len(gt_w)
    tp_w = 1
    fp_w = 1
    tn_w = 1
    fn_w = 1
    for i in range(n_w):
        if gt_w[i] == 1 and pred_w[i] == 1:
            tp_w+=1
        elif gt_w[i] == 0 and pred_w[i] == 1:
            fp_w+=1
        elif gt_w[i] == 0 and pred_w[i] == 0:
            tn_w +=1
        else: fn_w +=1

    p_true_b = (tp_b + fp_b)/float(n_b)
    p_true_w = (tp_w + fp_w)/float(n_w)

    parity_gap = abs(p_true_b - p_true_w)

    p_corr_den_b = tn_b / float(tn_b + fp_b)
    p_corr_den_w = tn_w / float(tn_w + fp_w)
    equality_gap_denied = abs(p_corr_den_b - p_corr_den_w)


    p_corr_appr_b = tp_b / float(tp_b + fn_b)
    p_corr_appr_w = tp_w / float(tp_w + fn_w)
    equality_gap_appr = abs(p_corr_appr_b - p_corr_appr_w)
    
    return [parity_gap, equality_gap_appr, equality_gap_denied]


test_data = pd.read_csv('/Users/kashishgarg/Downloads/test_clean.csv')

race = np.asarray(test_data.iloc[:, -4])
gt_loan = np.asarray(test_data.iloc[:, -1])

pred_rf = np.load('model/y_predict_rfclf.npy')
pred_lg = np.load('model/y_predict_lg.npy')
pred_gb = np.load('model/y_predict_gbclf.npy')
pred_kn = np.load('model/y_predict_knclf.npy')

American_Indian= 0
Asian = 1
Black = 2
Native_Hawaiian = 3
White = 4


for i in range(5):
    for j in range(i+1, 5):
        print("RF: (%d, %d) => %f"%(i, j, get_results(pred_rf, i, j)[0]))
        # print("GB: (%d, %d) => %f"%(i, j, get_results(pred_gb, i, j)[0]))
        # print("KN: (%d, %d) => %f"%(i, j, get_results(pred_kn, i, j)[0]))
        print("LR: (%d, %d) => %f"%(i, j, get_results(pred_lg, i, j)[0]))
        print("\n")
