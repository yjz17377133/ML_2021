import numpy as np

def calculate_accuracy(gt_result, pt_result, classes=None, info=''):
    # compute the label-based accuracy
    if gt_result.shape != pt_result.shape:
        print('Shape beteen groundtruth and predicted results are different')
    # compute the label-based accuracy
    label_acc = np.mean((gt_result == pt_result))
    return label_acc