"""
Main test script for PreoperativeSAM.

Adapted from Samus: https://github.com/xianlin7/SAMUS
"""
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import random
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
from thop import profile

from preoperativeSAM.cfg import get_config
from preoperativeSAM.utils.evaluation import get_eval
from preoperativeSAM.models.model_dict import get_model
from preoperativeSAM.utils.visualization import get_model_parameters
from preoperativeSAM.dataset.dataset import JointTransform2D, IntroperativeiUS, PrePostiUS
from preoperativeSAM.utils.loss_functions.sam_loss import get_criterion

def main(args):
    """
    Test the PreoperativeSAM model based on the provided argument
    """
    ## set logging level  ###########################################################
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])
    
    ## set opt and device and logging folder ##################################################################
    opt = get_config(args.task)
    opt.mode = "val"
    opt.visual = True
    opt.modelname = args.modelname
    device = torch.device(opt.device)
    logging.info(' Options information')
    logging.info(f' task - {args.task}')
    logging.info(f' model - {opt.modelname}')
    logging.info(f' checkpoint - {opt.load_path}')
    logging.info(f' device - {device}\n')

    ## set random seed for reproducibility  #######################################
    seed_value = opt.seed           
    np.random.seed(seed_value)                        # set random seed for numpy
    random.seed(seed_value)                           # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)    # avoid hash random
    torch.manual_seed(seed_value)                     # set random seed for CPU
    torch.cuda.manual_seed(seed_value)                # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)            # set random seed for all GPU
    torch.backends.cudnn.deterministic = True         # set random seed for convolution

    ## load the dataset  ##########################################################
    opt.batch_size = args.batch_size * args.n_gpu

    logging.info(' Creating val or test dataloader...')
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, 
                                p_flip=0, color_jitter_params=None, long_mask=True)
    
    ## based on a args i want to select the datatset loader
    dataset_cls = {
        'pre_and_post': IntroperativeiUS,
        'post': PrePostiUS,
    }
    dataset_class = dataset_cls[args.dataset_loader]
    val_dataset = dataset_class(main_path = opt.main_path, 
                                    dataset_name = opt.dataset_name, 
                                    split = opt.test_split, 
                                    joint_transform = tf_val, 
                                    img_size = args.encoder_input_size,
                                    degree_prompt = 1)  # return image, mask, and filename
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    logging.info(f'  - loader: {args.dataset_loader} - {dataset_class}')
    logging.info(f'  - test dataset: {len(val_dataset)}\n')


    ## get the model  ##############################################################
    logging.info(f" Creating model: {args.modelname} ...")
    model = get_model(modelname=args.modelname, args=args, opt=opt)
    
    
    ## load trained checkpoint
    load_path = os.path.join(opt.main_path, opt.result_folder, opt.dataset_name, opt.modelname, 
                            opt.save_folder, args.checkpoint, f'{opt.modelname}_best.pth')
    logging.info(f' load checkpoint: {load_path}')
    checkpoint = torch.load(load_path)

    #------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    get_model_parameters(model)
    # TO DO: add the code for estimate Gflops with profile 
    logging.info(' Done!\n')
    

#  == begin to evaluate the model ==================================================================
    model.eval()
    criterion = get_criterion(modelname=args.modelname, opt=opt)
    if opt.mode == "train":
        dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        print("mean dice:", mean_dice)
    else:
        mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        print("dataset:" + args.task + " -----------model name: "+ args.modelname)
        metrics = {"Dice": (mean_dice, std_dice),
            "Hausdorff": (mean_hdis, std_hdis),
            "IoU": (mean_iou, std_iou),
            "Accuracy": (mean_acc, std_acc),
            "Sensitivity": (mean_se, std_se),
            "Specificity": (mean_sp, std_sp)}

        # Stampa a video
        for name, (mean_vals, std_vals) in metrics.items():
            msg = f"{name}: "
            for m, s in zip(mean_vals[1:], std_vals[1:]):
                msg += f"{m:.4f} ± {s:.4f}  "
            print(msg)

        # Salvataggio su file
        text_results = os.path.join(opt.main_path, opt.result_folder, opt.dataset_name, opt.modelname, 
                            opt.save_folder, args.checkpoint, "test_result.txt")
        with open(text_results, "w") as file:
            file.write(f"{args.task} {args.modelname} \n")
            for name, (mean_vals, std_vals) in metrics.items():
                line = f"{name}: "
                for m, s in zip(mean_vals[1:], std_vals[1:]):
                    line += f"{m:.4f}±{s:.4f}  "
                file.write(line + "\n")  
            file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PreoperativeSAM model')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('--task', default='BUSI', help='task or dataset name')
    parser.add_argument('--dataset_loader', default='post', help='dataset - set dataset loader - pre_and_post = all imgs as as dataset; post = only intraopreative img as dataset')
    parser.add_argument('--checkpoint', default='post_data', help='checkpoint - set the checkpoint to load , an example is pre_and_post_25-11-2025_09_15')
    parser.add_argument('--encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('--low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
   
    args = parser.parse_args()
    main(args)
