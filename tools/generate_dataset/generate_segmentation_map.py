import sys; sys.path.append("../")
import numpy as np
import os
from glob import glob
import argparse
import json
import csv
import cv2
import natsort

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
    save_folder_name = ("segmentation_training_labels", "segmentation_test_labels")
    num_instruments = {}

    for i in range(2):
        # ## -------- 1. test only -------- ##
        # if i == 0:
        #     continue


        label_dir = os.path.join(args.label_dir, read_folder_name[i])
        json_list = glob(os.path.join(label_dir, "*.json"))

        json_list = natsort.natsorted(json_list)

        # img_list = glob(os.path.join('/raid/datasets/public/EndoVisPose/extract/labelled_train/*', '*.jpg'))
        # img_list = natsort.natsorted(img_list)

        mapper = {"LeftClasperPoint": 0,
                  "RightClasperPoint": 1,
                  "ShaftPoint": 3,
                  "EndPoint": 4,
                  "HeadPoint": 2}

        for json_ in json_list:
            with open(json_, "r") as j:
                dict_list = json.loads(j.read())
                # print(json_)
                seq_id = json_.split("/")[-1]
                seq_id = seq_id.split("_")[0]
                
            print('\nseq_id : ', seq_id)

            # ## -------- 2. test5 only -------- ##
            # if seq_id != 'test5':
            #     continue

            for element in dict_list:
                save_flag = True
                # print(element['filename'])
                ## -------- 3. specific video only -------- ##
                # if element['filename'] != '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Testing/Dataset5/Raw/img_0825.png':
                # if element['filename'] == '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Training/Training/Dataset1/Raw/img_000130_raw.png':
                #     print('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')

                # if element['filename'] == '/Users/xiaofeidu/mData/MICCAI_tool/Tracking_Robotic_Training/Training/Dataset1/Raw/img_000180_raw.png':
                #     print('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
                    
                if element["annotations"] == []:
                    continue

                try:
                    element["annotations"] = sorted(element['annotations'], key=lambda d: d['id'])
                except:
                    print('ERROR: annotation key missing')
                    print(element['annotations'])

                seg_map = np.zeros((576, 720, num_classes+num_connections), dtype=np.float32)
            
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

                        seg_map[:, :, idx] += cv2.circle(np.array(seg_map[:,:, idx]), (int(e['x']), int(e['y'])), 20, (255, 0, 0), -1)
                        
                        counter += 1

                        # img_path = '/raid/datasets/public/EndoVisPose/extract/labelled_train/Dataset1/img_000060_raw_train1.jpg'
                        # coloredImg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                        # cv2.imwrite('filename2.jpeg', seg_map[:, :, idx]+coloredImg)
            
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
                        
                        pt1, pt2, _ = part_coord["LeftClasperPoint"][idx]
                        pt3, pt4, _ = part_coord["HeadPoint"][idx]

                        seg_map[:, :, 5] += cv2.line(np.array(seg_map[:,:, 5]), (int(pt1), int(pt2)), (int(pt3), int(pt4)), (255, 0, 0), 20)
                                         

                    except:
                        pass

                    try: 
                        # rigth2head
                        if anno_sanity_check(part_coord["RightClasperPoint"][idx], part_coord["HeadPoint"][idx]) != True:
                            endovis_error.append([seq_id, element['filename']])
                            save_flag = False
                            print('endovis error : {}'.format(endovis_error))
                            break

                        pt1, pt2, _ = part_coord["RightClasperPoint"][idx]
                        pt3, pt4, _ = part_coord["HeadPoint"][idx]

                        seg_map[:, :, 6] += cv2.line(np.array(seg_map[:,:, 6]), (int(pt1), int(pt2)), (int(pt3), int(pt4)), (255, 0, 0), 20)
                    except:
                        pass
                    
                    try:
                        # head2shaft
                        if anno_sanity_check(part_coord["HeadPoint"][idx], part_coord["ShaftPoint"][idx]) != True:
                            endovis_error.append([seq_id, element['filename']])
                            save_flag = False
                            print('endovis error : {}'.format(endovis_error))
                            break

                        pt1, pt2, _ = part_coord["HeadPoint"][idx]
                        pt3, pt4, _ = part_coord["ShaftPoint"][idx]

                        seg_map[:, :, 7] += cv2.line(np.array(seg_map[:,:, 7]), (int(pt1), int(pt2)), (int(pt3), int(pt4)), (255, 0, 0), 20)
                    except:
                        pass
                    
                    try:
                        # shaft2end
                        if anno_sanity_check(part_coord["ShaftPoint"][idx], part_coord["EndPoint"][idx]) != True:
                            endovis_error.append([seq_id, element['filename']])
                            save_flag = False
                            print('endovis error : {}'.format(endovis_error))
                            break

                        pt1, pt2, _ = part_coord["ShaftPoint"][idx]
                        pt3, pt4, _ = part_coord["EndPoint"][idx]

                        seg_map[:, :, 8] += cv2.line(np.array(seg_map[:,:, 8]), (int(pt1), int(pt2)), (int(pt3), int(pt4)), (255, 0, 0), 20)
                    except:
                        pass                 

                if save_flag:
                    os.makedirs(os.path.join(args.save_dir, save_folder_name[i], seq_id), exist_ok=True)
                    np.save(os.path.join(args.save_dir, save_folder_name[i], seq_id, fname), seg_map)


    with open(os.path.join(args.save_dir, "segmentation_instrument_count.json"), "w") as d:
        json.dump(num_instruments, d)

    with open(os.path.join(args.save_dir, "segmentation_endovis_error.csv"),'w') as file :
        write = csv.writer(file)
        write.writerow(['seq_id', 'filename'])
        write.writerows(endovis_error)


def visaul(set_no):
    img_path = '/raid/datasets/public/EndoVisPose/extract/labelled_train/Dataset{}/'.format(set_no)
    img_list = glob(os.path.join(img_path, '*.jpg'))
    img_list = natsort.natsorted(img_list)

    segmentation_path = '/raid/datasets/public/EndoVisPose/annotation/segmentation_training_labels/train{}/'.format(set_no)
    segmentation_map = glob(os.path.join(segmentation_path, '*.npy'))
    segmentation_map = natsort.natsorted(segmentation_map)

    for i, s in zip(img_list, segmentation_map):
        
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        np_ = np.load(s)

        os.makedirs('./seg_map/train{}/{}'.format(set_no, i.split('/')[-1].split('_')[1]), exist_ok=True)
        cv2.imwrite('./seg_map/train{}/{}/LeftClasperPoint-{}'.format(set_no, i.split('/')[-1].split('_')[1] ,i.split('/')[-1]), np_[:,:,0]+img)
        cv2.imwrite('./seg_map/train{}/{}/RightClasperPoint-{}'.format(set_no, i.split('/')[-1].split('_')[1] ,i.split('/')[-1]), np_[:,:,1]+img)
        cv2.imwrite('./seg_map/train{}/{}/HeadPoint-{}'.format(set_no, i.split('/')[-1].split('_')[1] ,i.split('/')[-1]), np_[:,:,2]+img)
        cv2.imwrite('./seg_map/train{}/{}/ShaftPoint-{}'.format(set_no, i.split('/')[-1].split('_')[1] ,i.split('/')[-1]), np_[:,:,3]+img)
        cv2.imwrite('./seg_map/train{}/{}/EndPoint-{}'.format(set_no, i.split('/')[-1].split('_')[1] ,i.split('/')[-1]), np_[:,:,4]+img)
        cv2.imwrite('./seg_map/train{}/{}/left2head-{}'.format(set_no, i.split('/')[-1].split('_')[1] ,i.split('/')[-1]), np_[:,:,5]+img)
        cv2.imwrite('./seg_map/train{}/{}/rigth2head-{}'.format(set_no, i.split('/')[-1].split('_')[1] ,i.split('/')[-1]), np_[:,:,6]+img)
        cv2.imwrite('./seg_map/train{}/{}/head2shaft-{}'.format(set_no, i.split('/')[-1].split('_')[1] ,i.split('/')[-1]), np_[:,:,7]+img)
        cv2.imwrite('./seg_map/train{}/{}/shaft2end-{}'.format(set_no, i.split('/')[-1].split('_')[1] ,i.split('/')[-1]), np_[:,:,8]+img)


if __name__ == "__main__":
    # main()
    visaul(4)
