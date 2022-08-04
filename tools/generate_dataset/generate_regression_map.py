import sys; sys.path.append("../")
import numpy as np
import os
from glob import glob
import argparse
import json
from utils.annotation_parser import parse_annotation
from utils.heatmap_ import eval_gaussian, eval_line

import natsort
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", default='/raid/datasets/public/EndoVisPose', help="Directory that contains the labels", type=str, required=True)
    parser.add_argument("--save_dir", default='/SSL-2D-Pose/save', help="Directory to save the heatmaps", required=True)
    parser.add_argument("--dataset", type=str, default="ENDOVIS")
    parser.add_argument("--post_processing_labels", type=int, default=0,
                        help="Generate labels for post-processing module")
    args = parser.parse_args()

    if args.dataset == "RMIT":
        RMIT(args)
    elif args.dataset == "ENDOVIS":
        ENDOVIS(args)


def ENDOVIS(args):
    endovis_error = []

    
    def anno_sanity_check(tool1, tool2):
        return tool1[-1] == tool2[-1]

    def get_line(pt1, pt2):
        # (x, y)
        m = float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0]) # 두 pt 간의 기울기
        b = pt2[1] - m * pt2[0] # y 절편
        return m, b

    num_classes = 5
    num_connections = 4

    read_folder_name = ("train_labels", "test_labels")
    save_folder_name = ("training_labels_postprocessing_v2", "test_labels_postprocessing_v2")
    num_instruments = {}

    for i in range(2):
        # ## -------- 1. test only -------- ##
        # if i == 0:
        #     continue


        label_dir = os.path.join(args.label_dir, read_folder_name[i])
        json_list = glob(os.path.join(label_dir, "*.json"))

        json_list = natsort.natsorted(json_list)

        img_list = glob(os.path.join('/raid/datasets/public/EndoVisPose/extract/labelled_train/*', '*.jpg'))
        img_list = natsort.natsorted(img_list)

        mapper = {"LeftClasperPoint": 0,
                  "RightClasperPoint": 1,
                  "ShaftPoint": 3,
                  "EndPoint": 4,
                  "HeadPoint": 2}

        for json_ in json_list:
            with open(json_, "r") as j:
                dict_list = json.loads(j.read())
                print(json_)
                seq_id = json_.split("/")[-1]
                seq_id = seq_id.split("_")[0]
                
            print('\nseq_id : ', seq_id)

            # ## -------- 2. test5 only -------- ##
            # if seq_id != 'test5':
            #     continue

            for element in dict_list:
                save_flag = True
                print(element['filename'])
                ## -------- 3. specific video only -------- ##
                # if element['filename'] != '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Testing/Dataset5/Raw/img_0825.png':
                # if element['filename'] == '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Training/Training/Dataset1/Raw/img_000130_raw.png':
                #     print('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')

                # if element['filename'] == '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Training/Training/Dataset1/Raw/img_000180_raw.png':
                #     print('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
                    
                if element["annotations"] == []:
                    continue

                if not args.post_processing_labels:
                    try:
                        element["annotations"] = sorted(element['annotations'], key=lambda d: d['id'])
                    except:
                        print('ERROR: annotation key missing')
                        print(element['annotations'])

                    heatmap = np.zeros((576, 720, num_classes+num_connections), dtype=np.float32)
                    counter = 0
                    part_coord = {"LeftClasperPoint": [],
                                    "RightClasperPoint": [],
                                    "ShaftPoint": [],
                                    "EndPoint": [],
                                    "HeadPoint": []}

                    for e in element["annotations"]:
                        if e["class"] in mapper.keys():
                            idx = mapper[e["class"]]
                            part_coord[e["class"]].append((e["x"], e["y"], e["id"]))
                            heatmap[:, :, idx] += eval_gaussian([e["y"], e["x"]], h=576, w=720)
                            counter += 1

                    fname = element["filename"].split("/")[-1]
                    fname = fname.split(".")[0] + "_" + seq_id + ".npy"
                    num_instruments[fname] = counter // num_classes
                    # print(fname)
                    # return

                    print(part_coord)

                    # determine the connections
                    for idx in range(counter // num_classes + 1):
                        try:
                            # left2head
                            if anno_sanity_check(part_coord["LeftClasperPoint"][idx], part_coord["HeadPoint"][idx]) != True:
                                endovis_error.append([seq_id, element['filename']])
                                save_flag = False
                                print('endovis error : {}'.format(endovis_error))
                                break
                            
                            m, b = get_line(part_coord["LeftClasperPoint"][idx], part_coord["HeadPoint"][idx])
                            heatmap[:, :, 5] += eval_line(part_coord["LeftClasperPoint"][idx], part_coord["HeadPoint"][idx],
                                                    m, b)
                        except:
                            pass

                        try: 
                            # rigth2head
                            if anno_sanity_check(part_coord["RightClasperPoint"][idx], part_coord["HeadPoint"][idx]) != True:
                                endovis_error.append([seq_id, element['filename']])
                                save_flag = False
                                print('endovis error : {}'.format(endovis_error))
                                break

                            m, b = get_line(part_coord["RightClasperPoint"][idx], part_coord["HeadPoint"][idx])
                            heatmap[:, :, 6] += eval_line(part_coord["RightClasperPoint"][idx], part_coord["HeadPoint"][idx],
                                                            m, b)
                        except:
                            pass
                        
                        try:
                            # head2shaft
                            if anno_sanity_check(part_coord["HeadPoint"][idx], part_coord["ShaftPoint"][idx]) != True:
                                endovis_error.append([seq_id, element['filename']])
                                save_flag = False
                                print('endovis error : {}'.format(endovis_error))
                                break

                            m, b = get_line(part_coord["HeadPoint"][idx], part_coord["ShaftPoint"][idx])
                            heatmap[:, :, 7] += eval_line(part_coord["HeadPoint"][idx], part_coord["ShaftPoint"][idx], m, b)
                        except:
                            pass
                        
                        try:
                            # shaft2end
                            if anno_sanity_check(part_coord["ShaftPoint"][idx], part_coord["EndPoint"][idx]) != True:
                                endovis_error.append([seq_id, element['filename']])
                                save_flag = False
                                print('endovis error : {}'.format(endovis_error))
                                break

                            m, b = get_line(part_coord["ShaftPoint"][idx], part_coord["EndPoint"][idx])
                            heatmap[:, :, 8] += eval_line(part_coord["ShaftPoint"][idx], part_coord["EndPoint"][idx], m, b) 
                        except:
                            pass

                    
                    '''
                    ## -------- Visual ---------- ##
                    from visualization import np_visual
                    # img_path = os.path.join('/raid/datasets/public/EndoVisPose/extract/labelled_train/Dataset1', element['filename'].split('/')[-1].split('.')[0]+'_train1.jpg')
                    img_path = '/raid/datasets/public/EndoVisPose/extract/labelled_test/Dataset5/img_0825_test5.jpg'
                    save_dir = '/SSL-2D-Pose/utils/save5/{}'.format(img_path.split('_')[-2])

                    save_f = 'LeftClasperPoint-' + img_path.split('/')[-1] 
                    target = heatmap[:, :, 0]
                    np_visual(target, img_path, save_dir, save_f)

                    save_f = 'RightClasperPoint-' + img_path.split('/')[-1] 
                    target = heatmap[:, :, 1]
                    np_visual(target, img_path, save_dir, save_f)

                    save_f = 'HeadPoint-' + img_path.split('/')[-1] 
                    target = heatmap[:, :, 2]
                    np_visual(target, img_path, save_dir, save_f)

                    save_f = 'ShaftPoint-' + img_path.split('/')[-1] 
                    target = heatmap[:, :, 3]
                    np_visual(target, img_path, save_dir, save_f)

                    save_f = 'EndPoint-' + img_path.split('/')[-1] 
                    target = heatmap[:, :, 4]
                    np_visual(target, img_path, save_dir, save_f)

                    
                    save_f = 'left2head-' + img_path.split('/')[-1] 
                    target = heatmap[:, :, 5]
                    np_visual(target, img_path, save_dir, save_f)

                    save_f = 'rigth2head-' + img_path.split('/')[-1] 
                    target = heatmap[:, :, 6]
                    np_visual(target, img_path, save_dir, save_f)

                    save_f = 'head2shaft-' + img_path.split('/')[-1] 
                    target = heatmap[:, :, 7]
                    np_visual(target, img_path, save_dir, save_f)

                    save_f = 'shaft2end-' + img_path.split('/')[-1] 
                    target = heatmap[:, :, 8]
                    np_visual(target, img_path, save_dir, save_f)
                    '''                    

                else:
                    heatmap = np.zeros((576, 720, 10), dtype=np.float32)

                    mapper = {"tool1": {"LeftClasperPoint": 0,
                                        "RightClasperPoint": 1,
                                        "ShaftPoint": 3,
                                        "EndPoint": 4,
                                        "HeadPoint": 2
                                        },
                                "tool2": {"LeftClasperPoint": 5,
                                        "RightClasperPoint": 6,
                                        "ShaftPoint": 8,
                                        "EndPoint": 9,
                                        "HeadPoint": 7
                                        },
                                "tool4": {"LeftClasperPoint": 5,
                                        "RightClasperPoint": 6,
                                        "ShaftPoint": 8,
                                        "EndPoint": 9,
                                        "HeadPoint": 7
                                        }
                                }

                    for e in element["annotations"]:
                        try:
                            if e["class"] in mapper[e["id"]].keys():
                                idx = mapper[e["id"]][e["class"]]
                                heatmap[:, :, idx] += eval_gaussian([e["y"], e["x"]], h=576, w=720)
                                # import cv2
                                # cv2.imshow("heatmap", (heatmap[:, :, idx] * 255).astype("uint8"))
                                # cv2.waitKey(0)
                        except KeyError as err:
                            print(err)
                            print(e)
                            if e["x"] >= 720//2:
                                tool_id = "tool1"
                                idx = mapper[tool_id][e["class"]]
                                heatmap[:, :, idx] += eval_gaussian([e["y"], e["x"]], h=576, w=720)
                            else:
                                raise ValueError
                    fname = element["filename"].split("/")[-1]
                    fname = fname.split(".")[0] + "_" + seq_id + ".npy"

                if save_flag:
                    os.makedirs(os.path.join(args.save_dir, save_folder_name[i], seq_id), exist_ok=True)
                    np.save(os.path.join(args.save_dir, save_folder_name[i], seq_id, fname), heatmap)


    with open(os.path.join(args.save_dir, "instrument_count.json"), "w") as d:
        json.dump(num_instruments, d)

    with open(os.path.join(args.save_dir, "endovis_error.csv"),'w') as file :
        write = csv.writer(file)
        write.writerow(['seq_id', 'filename'])
        write.writerows(endovis_error)



def RMIT(args):
    save_dir = args.save_dir + "/heatmaps"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    label_list = sorted(glob(args.label_dir + "/*txt"), key=lambda x: x[0] < x[1])
    parsed = parse_annotation(label_list)

    for i in range(3):
        data = parsed["seq" + str(i + 1)]
        size = len(data["fname"])

        for j in range(size):
            heatmap = np.zeros((480, 640, 4), dtype=np.float32)
            heatmap[:, :, 0] = eval_gaussian(data["p1"][j])
            heatmap[:, :, 1] = eval_gaussian(data["p2"][j])
            heatmap[:, :, 2] = eval_gaussian(data["p3"][j])
            heatmap[:, :, 3] = eval_gaussian(data["p4"][j])
            np.save(save_dir + "/" + data["fname"][j] + ".npy", heatmap)


if __name__ == "__main__":
    main()
    