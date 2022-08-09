from utils.line_integral import *
from utils.nms import nms
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np



class Post_Processing():
    def __init__(self, num_parts=5, num_connections=4):
        self.num_parts = num_parts
        self.num_connections = num_connections
        # LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint, EndPoint (0,1,2,3,4)
        # joint1, joint2, joint1-joint2 pair id
        self.joint_pair_list = [(0, 2, 5), (1, 2, 6), (2, 3, 7), (3, 4, 8)]
 
    def run(self, bacth_iamges , window=100):
        result_batches = []
        for heatmap in bacth_iamges:
            heatmap = np.transpose(heatmap, (1,2,0)) # CHW - > HWC
            candidates = self.bipartite_graph_matching(heatmap, window)
            parsed = self.parsing(candidates)

            self.parse_failed = False
        
            # if len(parsed) == 2:
            #     inst1, inst2 = parsed
            #     final_prediction = list(zip(inst1, inst2))
            # elif len(parsed) == 1:
            #     final_prediction = list(zip((*parsed)))#parsed
            # else:
            #     final_prediction = [[], [], [], [], []] 
            #     self.parse_failed = True
            #     for i, pair in enumerate(candidates):
            #         # print(pair[0][0])
            #         if len(pair) >= 2:
            #             final_prediction[i] = [pair[0][0], pair[0][1]]
            #         elif len(pair) == 1:
            #             final_prediction[i] = [pair[0][0]]
            #         else:
            #             final_prediction[i] = []
            # inst = len(parsed)
            result_batches.append(list(zip(*parsed)))
            # result_batches.append(final_prediction)
        return result_batches

    def pred_init(self, heatmap, window):
        _, heatmap[:, :, :self.num_parts] = nms(heatmap[:, :, :self.num_parts], self.num_parts, window)
        loc_pred = [[] for i in range(self.num_parts)]  
        candidates_num = 5
        for i in range(self.num_parts):
            image = heatmap[:, :, i].copy()
            for j in range(candidates_num):
                _, max_val , _, max_loc = cv2.minMaxLoc(image)
                if max_loc[0] == max_loc[1] == 0:
                    break
                loc_pred[i].append(max_loc)
                y, x = max_loc

                image[x-5:x+5, y-5:y+5] = 0.
   
        return  loc_pred

    

    def bipartite_graph_matching(self, heatmap, window):
        
        loc_pred = self.pred_init(heatmap, window)

        candidates = [[] for i in range(self.num_connections)]  
        

        for idx, tmp in enumerate(self.joint_pair_list):

            joint_idx1, joint_idx2, connection_idx = tmp[0], tmp[1], tmp[2]

            matching_scores = np.zeros((len(loc_pred[joint_idx1]), len(loc_pred[joint_idx2])), dtype=np.float32)

            for idx1, pt1 in enumerate(loc_pred[joint_idx1]):
                for idx2, pt2 in enumerate(loc_pred[joint_idx2]):
                    # 해당 라인의 weight들을 더함
                    matching_scores[idx1, idx2] = compute_integral(pt1, pt2, heatmap[:, :, connection_idx])
                    # print("left2head", matching_scores)

 
            row_idx, col_idx = linear_sum_assignment(-matching_scores) # minimum weight matching in bipartite graphs.

            for r, c in zip(row_idx, col_idx):
                candidates[idx].append((loc_pred[joint_idx1][r], loc_pred[joint_idx2][c]))
  
        return candidates

    def parsing(self, candidates):
        # HeadPoint, ShaftPoint, EndPoint Connecting
        parsed = []
        for pairs in candidates[-1]:
            shaft, end = pairs
            for next_pairs in candidates[-2]:
                head, shaft_next = next_pairs
                if shaft[0] == shaft_next[0] and shaft[1] == shaft_next[1]: 
                    parsed.append([head, shaft, end])
      
        # LeftClasperPoint, RightClasperPoint, ShaftPoint Connecting
        for i, partial_pose in enumerate(parsed):
            head, _, _ = partial_pose
            for next_pairs in candidates[-3]:
                right, head_next = next_pairs
                if head[0] == head_next[0] and head[1] == head_next[1]:
           
                    parsed[i].insert(0, right)

            for next_pairs in candidates[-4]:
                left, head_next = next_pairs
                if head[0] == head_next[0] and head[1] == head_next[1]:
                    parsed[i].insert(0, left)
  
        # joint 가 missing 된 부분 ()으로 채우기 
        for i, pose in enumerate(parsed):
            if len(pose) < self.num_parts:
                for _ in range(self.num_parts - len(pose)):
                    parsed[i].insert(0, ())

        return parsed

        

if __name__ == "__main__":
    gt_path = '/raid/datasets/public/EndoVisPose/annotation/regression_train_labels_raduis20/train1/img_000565_raw_train1.npy' #train4/img_000290_raw_train4.npy' img_000175_raw_train1
    gt = np.load(gt_path)
    gt = np.transpose(gt, (2,0,1))
    p = Post_Processing(num_parts=5, num_connections=4)

    heatmap = np.random.randn(1, 9, 576, 720)

    # heatmap[0] = gt
    final_prediction = p.run(heatmap)

    print(final_prediction)

    tool_list = ['LeftClasperPoint', 'RightClasperPoint', 'HeadPoint', 'ShaftPoint', 'EndPoint']
    for name, point in zip(tool_list, final_prediction):
        print(name)
        print('tool1')
        print('x : ', point[0][0])
        print('y : ', point[0][1])

        print('tool2')
        print('x : ', point[1][0])
        print('y : ', point[1][1])
        print("=====================")