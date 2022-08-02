
from scipy.optimize import linear_sum_assignment
import sys
import os
import numpy as np 
# sys.path.append(['/hsb/optimal_surgery/pose_estimation/ours/tools/'])
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from post_processing import Post_Processing

class instrument_pose_metric():
    def __init__(self, configs):
        self.threshold = configs['metric']['threshold'] 

        self.joint_num = configs['dataset']['num_parts'] 

    def forward(self, preds, targets):
        # preds = preds.cpu().numpy()
        # targets = targets.cpu().numpy()
        false_pos = np.zeros((5, ), dtype=np.float32)
        false_neg = np.zeros((5, ), dtype=np.float32)
        true_pos = np.zeros((5, ), dtype=np.float32)
        rmse = np.zeros_like(true_pos)
        mae = np.zeros_like(rmse)
        counter = np.zeros_like(rmse)

        precision = lambda fp, tp: tp / (tp + fp)
        recall = lambda fn, tp: tp / (tp + fn)

        # array order : batch * [[left, right, head, shaft, end], [left, right, head, shaft, end], [left, right, head, shaft, end], ...]
        # array order : batch *[[left1, left2, left3, ...], [right1, right2, right3, ...], ...]
 
        for batch_idx in range(len(preds)):
            pred = preds[batch_idx]
            target = targets[batch_idx]
            for k in range(self.joint_num):
      
                cost_matrix = np.zeros((len(target[k]), len(pred[k])), dtype=np.float32)

   
                for real_i, e_true in enumerate(target[k]):
                    for pred_j, e_pred in enumerate(pred[k]):  # (preds[k]):
               
                        if len(e_pred) == 0:
                            cost_matrix[real_i, pred_j] = 1e9
                        else:
                            # print(e_pred, e_true)
                            cost_matrix[real_i, pred_j] = ((e_pred[0] - e_true[0])) ** 2 + ((e_pred[1] - e_true[1])) ** 2
          
            
                row_idx, col_idx = linear_sum_assignment(cost_matrix)

                if len(pred) < len(target):
                    false_neg[k] += abs(len(pred) - len(target))
   
                for r, c in zip(row_idx, col_idx):
                    # check true positive and additional false positive
                    mae[k] += np.sqrt(cost_matrix[r, c])
                    counter[k] += 1

                    if np.sqrt(cost_matrix[r, c]) < self.threshold:
                        rmse[k] += cost_matrix[r, c]
                        true_pos[k] += 1
                        
                    else:
                        false_pos[k] += 1
                        
        f1 = lambda p, r: (2 * p * r) / (p + r)
        p = precision(false_pos, true_pos)
        r = recall(false_neg, true_pos)

        metric = dict()
        metric = {
            "TP/FN/FP": [true_pos.tolist(), false_neg.tolist(), false_pos.tolist()],
            'RMSE': np.sqrt(rmse / true_pos),
            "Precision": p,
            "Recall": r,
            "F1": f1(p, r),
            "MEA": mae / counter
        }
        # print(true_pos, false_neg, false_pos)
        # print("RMSE", np.sqrt(rmse / true_pos))
        # print("Precision", p)
        # print("Recall", r)
        # print("F1", f1(p, r))
        # print("MEA", mae / counter)
        # print("\n")
        return metric

if __name__ == "__main__":
    gt_path = '/raid/datasets/public/EndoVisPose/annotation/training_labels_postprocessing_v2/train1/img_000175_raw_train1.npy' #train4/img_000290_raw_train4.npy' img_000175_raw_train1
    gt = np.load(gt_path)
    p = Post_Processing(num_parts=5, num_connections=4)

    heatmap = np.random.randn(576,720,9)

    gt = p.run(gt)

    

    pred_path = '/raid/datasets/public/EndoVisPose/annotation/training_labels_postprocessing_v2/train4/img_000290_raw_train4.npy' #train4/img_000290_raw_train4.npy' img_000175_raw_train1
    pred = np.load(pred_path)
    p = Post_Processing(num_parts=5, num_connections=4)



    pred = [[i.tolist()] for i in np.squeeze(p.run(pred))]
    print(gt)
    print(pred)
    
    
    
    configs = dict()
    configs['metric'] = dict()
    configs['metric']['threshold'] = 250
    configs['dataset'] = dict()
    configs['dataset']['num_parts'] = 5

    metric = instrument_pose_metric(configs)
    print(metric.forward([pred], [gt]))