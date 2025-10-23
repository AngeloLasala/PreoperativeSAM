
"""
Read images form slicer
Note: this script is specific for the RESECT iUS , use ius environment !!!
"""
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
import argparse
import tqdm
import seaborn as sns
from PIL import Image

def check_subject_volume(dataset_path):
    """
    Given the path of the RESECT dataset iUS, return the list of subject with volume
    """
    subjects_with_volume = []
    for item in os.listdir(dataset_path):
        # chek if the sting item finished with .nii
        if item.endswith(".nii.gz"):
            subjects_with_volume.append(item.split('.')[0])
    return subjects_with_volume 

def subject_label_dict(datasets, save_dict=False):
    """
    Create the dictionary with the subject and the label

    keys: subject (Case1-US-)
    values: dict with keys: pre and post, values: path of the volume and path to the label
    """

    # check if pre and post have same subjects 
    patients_list = {'pre':None, 'post':None}
    
    ## check if pre and post have same subjects
    for key, value in datasets.items():
        subjects_with_volume = check_subject_volume(datasets[key])
        patients_list[key] = [o.split('-')[0]+'-US-' for o in subjects_with_volume]
    common_patients = list(set(patients_list['pre']).intersection(set(patients_list['post'])))
    if len(common_patients) == len(patients_list['pre']) == len(patients_list['post']):
        logging.info('Pre and post datasets have the same subjects')
    else:
        raise ValueError('Pre and post datasets have different subjects')

    dataset_dict = {'pre':{}, 'post':{}}

    for subject in common_patients:
        dataset_dict['pre'][subject] = {'volume':f'{subject}before.nii', 'tumor':None}
        dataset_dict['post'][subject] = {'volume':f'{subject}during.nii', 'tumor':None}
        
        ## Pre dataset
        if os.path.exists(os.path.join(datasets['pre'], f'{subject}before-tumor.nii.gz.seg.nrrd')):
            dataset_dict['pre'][subject]['tumor'] = f'{subject}before-tumor.nii.gz.seg.nrrd'
        else:
            # chek if exitst the item that start with f'{item}-tumor.nii.sef.nrrd'
            if os.path.exists(os.path.join(datasets['pre'], f'{subject}before-tumor.nii.seg.nrrd')):
                dataset_dict['pre'][subject]['tumor'] = f'{subject}before-tumor.nii.seg.nrrd'
            else:
                # print(f'No tumor for subject {subject}')
                pass
        
        ## Post dataset
        if os.path.exists(os.path.join(datasets['post'], f'{subject}during-resection.nii.gz.seg.nrrd')):
            dataset_dict['post'][subject]['tumor'] = f'{subject}during-resection.nii.gz.seg.nrrd'
        else:
            # chek if exitst the item that start with f'{item}-tumor.nii.sef.nrrd'
            if os.path.exists(os.path.join(datasets['post'], f'{subject}during-resection.nii.seg.nrrd')):
                dataset_dict['post'][subject]['tumor'] = f'{subject}during-resection.nii.seg.nrrd'
            else:
                # print(f'No tumor for subject {subject}')
                pass

    if save_dict:
        # save dict ad json
        with open(os.path.join(dataset_path, 'dataset_dict.json'), 'w') as json_file:
            json.dump(dataset_dict, json_file)
    return dataset_dict, common_patients
       

def compute_volume_mm3(segmentation_map):
    """
    Compute the volume in mm^3 of the segmentation maps

    Parameters
    ----------
    volume_array : 
        Segmentation map, SimpleITK image

    Returns
    -------
    volume_mm3 : float
        Volume in mm^3
    """
    spacing = segmentation_map.GetSpacing()
    volume_mm3 = np.sum(sitk.GetArrayFromImage(segmentation_map)) * np.prod(spacing)
    return volume_mm3, spacing

def read_volume_label_subject(datasets, dataset_dict, subject, show_plot=True):
    """
    Read the volume and the label of a SINGLE subject

    Parameters
    ----------
    dataset_path : str
        Path of the dataset
    dataset_dict : dict
        Dictionary with the subject and the label, from the function subject_label_dict
    subject : str
        Subject to read, 'Case1-US-before'
    """

    value_pre = dataset_dict['pre'][subject]
    value_post = dataset_dict['post'][subject]
    logging.info(f'subject: {subject}')

    ## PRE DATASET
    if value_pre['volume'] is not None : 
        volume_pre = os.path.join(datasets['pre'], value_pre['volume'])
        volume_pre = sitk.ReadImage(volume_pre)
        volume_array_pre = sitk.GetArrayFromImage(volume_pre)  # Shape: (Z, Y, X)
        logging.info(f'PRE volume shape: {volume_array_pre.shape}')
    else:
        logging.info('No PRE volume for this subject')

    if value_pre['tumor'] is not None :
        tumor_pre = os.path.join(datasets['pre'], value_pre['tumor'])
        tumor_pre = sitk.ReadImage(tumor_pre)
        tumor_size_pre, spacing_pre = compute_volume_mm3(tumor_pre)
        # covert the order of the spacinf in 2, 1, 0
        # spacing = np.array(spacing)[::-1]
        logging.info(f'PRE tumor size: {tumor_size_pre/1000.0:.4f} [cm^3]')
        tumor_array_pre = sitk.GetArrayFromImage(tumor_pre)
    else:
        tumor_array_pre = None
        logging.info('No PRE tumor for this subject')

    ## POST DATASET
    if value_post['volume'] is not None :
        volume_post = os.path.join(datasets['post'], value_post['volume'])
        volume_post = sitk.ReadImage(volume_post)
        volume_array_post = sitk.GetArrayFromImage(volume_post)
        logging.info(f'POST volume shape: {volume_array_post.shape}')
    else:
        logging.info('No POST volume for this subject')
    if value_post['tumor'] is not None :
        tumor_post = os.path.join(datasets['post'], value_post['tumor'])
        tumor_post = sitk.ReadImage(tumor_post)
        tumor_size_post, spacing_post = compute_volume_mm3(tumor_post)
        # covert the order of the spacinf in 2, 1, 0
        # spacing = np.array(spacing)[::-1]
        logging.info(f'POST tumor size: {tumor_size_post/1000.0:.4f} [cm^3]')
        tumor_array_post = sitk.GetArrayFromImage(tumor_post)
    else:
        ## create a empty array with the same shape of the volume
        print('sto qui')
        tumor_array_post = np.zeros_like(volume_array_post)
        tumor_size_post = 0.0
        spacing_post = 0.0

        
    # in tumor, find the slice with the maximun number of 1 pixel
    def find_max_tumor_slice(tumor_array):
        max_tumor = 0
        slice_tumor = 0
        for i in range(tumor_array.shape[0]):
            if np.sum(tumor_array[i, :, :]) > max_tumor:
                max_tumor = np.sum(tumor_array[i, :, :])
                slice_tumor = i
        return slice_tumor

    slice_tumor_pre = find_max_tumor_slice(tumor_array_pre)
    slice_tumor_post = find_max_tumor_slice(tumor_array_post)

    if show_plot:
        ## plot the volume with the tumor in the max_tumor slice 
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), num=f'{subject}', tight_layout=True) 
        ax[0].imshow(volume_array_pre[slice_tumor_pre, :, :], cmap="gray")
        ax[0].imshow(tumor_array_pre[slice_tumor_pre, :, :], alpha=0.2, cmap="jet")  # Overlay
        ax[0].axis('off')
        ax[0].set_title(f'PRE - max tumor slice {slice_tumor_pre}')

        ax[1].imshow(volume_array_post[slice_tumor_post, :, :], cmap="gray")
        ax[1].imshow(tumor_array_post[slice_tumor_post, :, :], alpha=0.2, cmap="jet")  # Overlay
        ax[1].axis('off')
        ax[1].set_title(f'POST - max tumor slice {slice_tumor_post}')
        
    return volume_array_pre, tumor_array_pre, tumor_size_pre, spacing_pre, slice_tumor_pre, volume_array_post, tumor_array_post, tumor_size_post, spacing_post, slice_tumor_post

def slicing_volume(volume_array, tumor_array, spacing, slice_tumor, subject, slice_spacing=1,
                   save_dir=None, dataset_type=None, save_dataset=False):
    """
    Slice the volume each 5 mm upward and backward from
    the slice of tumor that contains the maximum number of 1 pixel, 'bigger surface'

    Parameters
    ----------
    volume_array : 
        Volume array, numpy or None
    tumor_array :
        Tumor array, numpy or None
    spacing :
        Spacing of the volume in mm, 3d array
    slice_tumor :
        Slice of the tumor that contains the maximum number of 1 pixel, bigger surface
    slice_spacing :
        Spacing between the slices in mm, default 1 mm
    dataset_path :
        Path of the dataset, default None
    save_dataset :
        Save the dataset, default False
    """

    ## create folders structure for the dataset to develop DL model
    if save_dataset:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, dataset_type), exist_ok=True)

        subject_path = os.path.join(save_dir, dataset_type, subject)
        os.makedirs(os.path.join(subject_path, 'img'), exist_ok=True)
        os.makedirs(os.path.join(subject_path, 'label'), exist_ok=True)
        os.makedirs(os.path.join(subject_path, 'bbox'), exist_ok=True)
    
    ## surface of the max tumor slice
    if tumor_array is not None and np.sum(tumor_array) > 0.0:
        print("tumor_array type:", type(tumor_array))
        print("slice_tumor:", slice_tumor, "type:", type(slice_tumor))
        print("spacing:", spacing, "type:", type(spacing))
        max_tumor_surface = np.sum(tumor_array[slice_tumor, :, :]) * spacing[1] * spacing[2]
        slicing_window = tumor_array.shape[0] * spacing[0] // 4

        int_slicing = int(slice_spacing/spacing[0]) #int of slice
        int_window = int(slicing_window/spacing[0]) #int of window)
        tumor_surface_list = []
        for slice_i in np.arange(slice_tumor - (int_window//2), slice_tumor + (int_window//2) + 1 ,int_slicing):
            tumor_surface_list.append(np.sum(tumor_array[slice_i, :, :]) * spacing[1] * spacing[2])


            ## save the dataset
            if save_dataset:
                vol = volume_array[slice_i, :, :]
                tum = tumor_array[slice_i, :, :]
            
                # convert un PIL image
                vol = Image.fromarray(vol).convert('L')
                tum = Image.fromarray(tum).convert('L')
        
                # save the image, only image with tumor
                if np.sum(tum) > 0.0: 
                    vol.save(os.path.join(subject_path, 'img', f'{subject}_{slice_i}.png'))
                    tum.save(os.path.join(subject_path, 'label', f'{subject}_{slice_i}.png'))

        return max_tumor_surface, tumor_surface_list
    
    else:
        logging.info('No tumor array provided for slicing')
        print("tumor_array type:", type(tumor_array))
        print("slice_tumor:", slice_tumor, "type:", type(slice_tumor))
        print("spacing:", spacing, "type:", type(spacing))
        return None, None

def main(args):
    """
    Create the folder of pre and post image and label

    Note: this is  for iUS dataset, look at the specific format required for SAM
    """
    datasets = {'pre':args.dataset_pre, 'post':args.dataset_post}

    # create the dict for per and post
    dataset_dict, subject_list = subject_label_dict(datasets, save_dict=False)

    # Read the volume and the label of a SINGLE subject
    subject_random = subject_list[np.random.randint(0, len(subject_list))]
    read_volume_label_subject(datasets, dataset_dict, 'Case27-US-', show_plot=True)
    
    ## read entire dataset
    for subject in tqdm.tqdm(subject_list):
        print(subject)
        data_subject = read_volume_label_subject(datasets, dataset_dict, subject, show_plot=False)
        volume_pre, tumor_pre, tumor_size_pre, spacing_pre, slice_tumor_pre, volume_post, tumor_post, tumor_size_post, spacing_post, slice_tumor_post = data_subject

        # slicing pre
        slicing_volume(volume_pre, tumor_pre, spacing_pre, slice_tumor_pre, subject,
                      slice_spacing=1.0, save_dir=args.save_dir, dataset_type='pre', save_dataset=True)

        # slicing post
        slicing_volume(volume_post, tumor_post, spacing_post, slice_tumor_post, subject,
                        slice_spacing=1.0, save_dir=args.save_dir, dataset_type='post', save_dataset=True)  

    plt.show()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read the dataset of the RESECT iUS dataset')
    parser.add_argument('--dataset_pre', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_dataset", help='Path to the dataset pre')
    parser.add_argument('--dataset_post', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Visiting_Imperial/RESECT_iUS_during", help='Path to the dataset post')
    parser.add_argument('--save_dir', type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE/Dataset_iUS", help='Path to save the sliced dataset')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()

    ## set the logger
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    main(args)
    
