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
    
    exit()





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

    ##


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read HENANCE dataset')
    parser.add_argument('--dataset', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE/HENANCE", help='Path to the dataset pre')
    parser.add_argument('--save_dir', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE/Dataset_HENANCE", help='Path to save the sliced dataset')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    main(args)
    