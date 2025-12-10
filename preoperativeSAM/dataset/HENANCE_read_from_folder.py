"""
Read HENANCE dataset fro original folder
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
import argparse
import tqdm
import seaborn as sns
from PIL import Image
import math
import cv2

def process_dataset_info(dataset_info, args):
    """
    Process dataset info from HENANCE dataset anch check the quality of json
    """
    print("Processing dataset info...")

    print('Dataset info')
    for key in dataset_info.keys():
        print(f" - {key}: number of subjects = {len(dataset_info[key])}")
        pre_tot = 0
        intra_tot = 0
        for sub_dict in dataset_info[key]:
            sub_idx = list(sub_dict.keys())[0]

            sub_pre = sub_dict[sub_idx]['pre']
            sub_intra = sub_dict[sub_idx]['intra']

            ## chech if all item in sub_pre are present as files in folder
            total_img = sub_pre + sub_intra
            lesions = []
            for img_name in total_img:
                img_path = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.jpg')
                ann_path_1 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.json')
                ann_path_2 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.xml')
                if not os.path.exists(img_path):
                    print(f"   - WARNING: image {img_name} not found for subject {sub_idx}")
                if not os.path.exists(ann_path_1) and not os.path.exists(ann_path_2):
                    print(f"   - WARNING: annotation for image {img_name} not found for subject {sub_idx}")

                ## read the json and check the category of lesions
                
                if os.path.exists(ann_path_1):
                    with open(ann_path_1) as f:
                        ann_data = json.load(f)
                        annotations = ann_data.get('annotations', [])
                        for annotation in annotations:
                            lesions.append(annotation['category_id'])
                else: 
                    lesions = []
                # unique lesions
                lesions = list(set(lesions))
            pre_tot += len(sub_pre)
            intra_tot += len(sub_intra)
            print(f"   - {sub_idx}: pre={len(sub_pre)} intra={len(sub_intra)} - lesions_category={lesions}")

        print(f"   Total pre: {pre_tot}, Total intra: {intra_tot}")
        print()
    print()

def read_json_annotation(ann_path):
    """
    Read json annotation file and return the annotations
    """
    with open(ann_path) as f:
        ann_data = json.load(f)
        annotations = ann_data.get('annotations', [])
    
    points_ann = []
    for ann in annotations:
        category_id = ann['category_id']
        three_class_id = ann['three_class_id']
        segmentation = ann['segmentation']

        # get segmentation:
        if len(segmentation) > 0:
            # print(f"   - category_id: {category_id}, three_class_id: {three_class_id}, number of segmentation points: {len(segmentation)}")
            coords = segmentation[0]
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        else:
            points = []
        points_ann.append(points)
    return points_ann

def read_xml_annotation(ann_path):
    """
    Read xml annotation file and return the annotations
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(ann_path)
    root = tree.getroot()
    annotations = []
    
    points_ann = []
    for obj in root.findall('object'):
        category_id = obj.find(".//classification_id").text
        three_class_id = obj.find(".//three_class_classification").text
        # print(f"   - category_id: {category_id}, three_class_id: {three_class_id}")
        pts = obj.findall(".//pt")
        if pts:
            polygon = [(int(pt.find("x").text), int(pt.find("y").text)) for pt in pts]
            points_ann.extend(polygon)

    if len(points_ann) == 0:
        return []

    # Convert to numpy
    pts = np.array(points_ann, dtype=np.float32)

    # 1. Remove duplicate points
    pts = np.unique(pts, axis=0)

    # 2. Sort clockwise around center of mass
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)
    pts_sorted = pts[np.argsort(angles)]

    return [pts_sorted.astype(int).tolist()]

def processing_dataset(dataset_info, args):
    """
    Process HENANCE dataset and create Dataset_HENANCE comaptible with PreSAM
    """
    print("Processing HENANCE dataset...")
     ## pre and intra dataset
    
    save_dir = args.save_dir

    # create 'pre' and 'intra' folders
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pre_dir = os.path.join(save_dir, 'pre')
    intra_dir = os.path.join(save_dir, 'intra')
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    if not os.path.exists(intra_dir):
        os.makedirs(intra_dir)
    
    # process 'pre_and_intra' dataset
    dataset_pre_and_intra = dataset_info['pre_and_intra']
    for sub_dict in dataset_pre_and_intra:
        sub_idx = list(sub_dict.keys())[0]
        print(f"Processing subject: {sub_idx}")

        sub_pre = sub_dict[sub_idx]['pre']
        sub_intra = sub_dict[sub_idx]['intra']

        # create sub folder in pre and intra
        sub_pre_dir = os.path.join(pre_dir, str(sub_idx))
        sub_intra_dir = os.path.join(intra_dir, str(sub_idx))
        os.makedirs(sub_pre_dir, exist_ok=True)
        os.makedirs(sub_intra_dir, exist_ok=True)
            
        img_pre_dir = os.path.join(sub_pre_dir, 'img')
        label_pre_dir = os.path.join(sub_pre_dir, 'label')
        os.makedirs(img_pre_dir, exist_ok=True)
        os.makedirs(label_pre_dir, exist_ok=True)
        
        img_intra_dir = os.path.join(sub_intra_dir, 'img')
        label_intra_dir = os.path.join(sub_intra_dir, 'label')
        os.makedirs(img_intra_dir, exist_ok=True)
        os.makedirs(label_intra_dir, exist_ok=True)

        ## process pre imags
        for img_name in sub_pre:
            img_path = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.jpg')
            ann_path_1 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.json')
            ann_path_2 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.xml')
            if not os.path.exists(img_path):
                print(f"   - WARNING: image {img_name} not found for subject {sub_idx}")
            if not os.path.exists(ann_path_1):
                ann_path = ann_path_2
            else:
                ann_path = ann_path_1

            print(os.path.basename(img_path), os.path.basename(ann_path))
            img = Image.open(img_path).convert("RGB")
            points = read_json_annotation(ann_path) if ann_path.endswith('.json') else read_xml_annotation(ann_path)
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
            for m in points:
                if len(m) > 0:
                    cv2.fillPoly(mask, [np.array(m, dtype=np.int32)], 1)

            ## save img and mask
            img.save(os.path.join(img_pre_dir, f'{img_name}.jpg'))
            Image.fromarray(mask, mode="L").save(os.path.join(label_pre_dir, f'{img_name}.png'))

            # plt.figure(figsize=(14,6), num=f'{os.path.basename(img_path)} - Preoperative', tight_layout=True)
            # plt.subplot(1,3,1)
            # plt.imshow(img)
            # plt.subplot(1,3,2)
            # plt.imshow(mask, cmap='gray')
            # plt.subplot(1,3,3)
            # plt.imshow(img)
            # plt.imshow(mask, cmap='jet', alpha=0.4)
            # plt.show()
        print()

        ## process intra imags
        for img_name in sub_intra:
            img_path = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.jpg')
            ann_path_1 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.json')
            ann_path_2 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.xml')
            if not os.path.exists(img_path):
                print(f"   - WARNING: image {img_name} not found for subject {sub_idx}")
            if not os.path.exists(ann_path_1):
                ann_path = ann_path_2
            else:
                ann_path = ann_path_1

            print(os.path.basename(img_path), os.path.basename(ann_path))
            img = Image.open(img_path).convert("RGB")
            points = read_json_annotation(ann_path) if ann_path.endswith('.json') else read_xml_annotation(ann_path)
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
            for m in points:
                if len(m) > 0:
                    cv2.fillPoly(mask, [np.array(m, dtype=np.int32)], 1)

            ## save img and mask
            img.save(os.path.join(img_intra_dir, f'{img_name}.jpg'))
            Image.fromarray(mask, mode="L").save(os.path.join(label_intra_dir, f'{img_name}.png'))

            # plt.figure(figsize=(14,6), num=f'{os.path.basename(img_path)} - Intraoperative', tight_layout=True)
            # plt.subplot(1,3,1)
            # plt.imshow(img)
            # plt.subplot(1,3,2)
            # plt.imshow(mask, cmap='gray')
            # plt.subplot(1,3,3)
            # plt.imshow(img)
            # plt.imshow(mask, cmap='jet', alpha=0.4)
            # plt.show()
        

    ## process only intra dataset
    dataset_only_intra = dataset_info['only_intra']
    for sub_dict in dataset_only_intra:
        sub_idx = list(sub_dict.keys())[0]
        print(f"Processing subject: {sub_idx}")

        sub_pre = sub_dict[sub_idx]['pre']
        sub_intra = sub_dict[sub_idx]['intra']

        # create sub folder in pre and intra
        sub_pre_dir = os.path.join(pre_dir, str(sub_idx))
        sub_intra_dir = os.path.join(intra_dir, str(sub_idx))
        os.makedirs(sub_pre_dir, exist_ok=True)
        os.makedirs(sub_intra_dir, exist_ok=True)
            
        img_pre_dir = os.path.join(sub_pre_dir, 'img')
        label_pre_dir = os.path.join(sub_pre_dir, 'label')
        os.makedirs(img_pre_dir, exist_ok=True)
        os.makedirs(label_pre_dir, exist_ok=True)
        
        img_intra_dir = os.path.join(sub_intra_dir, 'img')
        label_intra_dir = os.path.join(sub_intra_dir, 'label')
        os.makedirs(img_intra_dir, exist_ok=True)
        os.makedirs(label_intra_dir, exist_ok=True)

        ## process pre imags
        for img_name in sub_pre:
            img_path = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.jpg')
            ann_path_1 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.json')
            ann_path_2 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.xml')
            if not os.path.exists(img_path):
                print(f"   - WARNING: image {img_name} not found for subject {sub_idx}")
            if not os.path.exists(ann_path_1):
                ann_path = ann_path_2
            else:
                ann_path = ann_path_1

            print(os.path.basename(img_path), os.path.basename(ann_path))
            img = Image.open(img_path).convert("RGB")
            points = read_json_annotation(ann_path) if ann_path.endswith('.json') else read_xml_annotation(ann_path)
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
            for m in points:
                if len(m) > 0:
                    cv2.fillPoly(mask, [np.array(m, dtype=np.int32)], 1)

            ## save img and mask
            img.save(os.path.join(img_pre_dir, f'{img_name}.jpg'))
            Image.fromarray(mask, mode="L").save(os.path.join(label_pre_dir, f'{img_name}.png'))

        print()

        ## process intra imags
        for img_name in sub_intra:
            img_path = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.jpg')
            ann_path_1 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.json')
            ann_path_2 = os.path.join(args.dataset, str(sub_idx), sub_dict[sub_idx]['folder'], f'{img_name}.xml')
            if not os.path.exists(img_path):
                print(f"   - WARNING: image {img_name} not found for subject {sub_idx}")
            if not os.path.exists(ann_path_1):
                ann_path = ann_path_2
            else:
                ann_path = ann_path_1

            print(os.path.basename(img_path), os.path.basename(ann_path))
            img = Image.open(img_path).convert("RGB")
            points = read_json_annotation(ann_path) if ann_path.endswith('.json') else read_xml_annotation(ann_path)
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
            for m in points:
                if len(m) > 0:
                    cv2.fillPoly(mask, [np.array(m, dtype=np.int32)], 1)

            ## save img and mask
            img.save(os.path.join(img_intra_dir, f'{img_name}.jpg'))
            Image.fromarray(mask, mode="L").save(os.path.join(label_intra_dir, f'{img_name}.png'))

def get_data_splittings(dataset_info, args, splitting_seed=42):
    """
    Given the path of the dataset return the list of patient for train, valindation and test
    """
    print("Creating data splittings...")
    np.random.seed(splitting_seed)

    per_and_intra_subject_split = [13, 3, 3]
    only_intra_subject_split = [4, 3, 4]

    pre_and_intra_dicts = dataset_info['pre_and_intra']
    only_intra_dicts = dataset_info['only_intra']
    pre_and_intra_subjects = [list(sub_dict.keys())[0] for sub_dict in pre_and_intra_dicts]
    only_intra_subjects = [list(sub_dict.keys())[0] for sub_dict in only_intra_dicts]

    np.random.shuffle(pre_and_intra_subjects)
    np.random.shuffle(only_intra_subjects)

    n_train_pre_and_intra = pre_and_intra_subjects[:per_and_intra_subject_split[0]]
    n_val_pre_and_intra = pre_and_intra_subjects[per_and_intra_subject_split[0]:per_and_intra_subject_split[0]+per_and_intra_subject_split[1]]
    n_test_pre_and_intra = pre_and_intra_subjects[per_and_intra_subject_split[0]+per_and_intra_subject_split[1]:]

    n_train_only_intra = only_intra_subjects[:only_intra_subject_split[0]]
    n_val_only_intra = only_intra_subjects[only_intra_subject_split[0]:only_intra_subject_split[0]+only_intra_subject_split[1]]
    n_test_only_intra = only_intra_subjects[only_intra_subject_split[0]+only_intra_subject_split[1]:]
    
    train = n_train_pre_and_intra + n_train_only_intra
    val = n_val_pre_and_intra + n_val_only_intra
    test = n_test_pre_and_intra + n_test_only_intra
    splitting_dict = {'train': train, 'val': val, 'test': test}

    ## save splitting dict
    save_dir = args.save_dir
    splitting_path = os.path.join(save_dir, 'splitting.json')
    with open(splitting_path, 'w') as f:
        json.dump(splitting_dict, f, indent=4)
    print(f"Done!")


    return splitting_dict

def main(args):
    """
    Process HENANCE dataset and create Dataset_HENANCE comaptible with PreSAM
    """
    ## read dataset info
    dataset_info_path = os.path.join(args.dataset, 'dataset_info.json')
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)
    
    ## process dataset info
    process_dataset_info(dataset_info, args)

    ## process dataset
    processing_dataset(dataset_info, args)

    ## create data splittings
    get_data_splittings(dataset_info, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read HENANCE dataset')
    parser.add_argument('--dataset', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE/HENANCE", help='Path to the dataset pre')
    parser.add_argument('--save_dir', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE/Dataset_HENANCE", help='Path to save the sliced dataset')
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    main(args)
    