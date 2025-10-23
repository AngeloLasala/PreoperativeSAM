"""
Main train script for PreoperativeSAM.
"""
import argparse
import os
import torch
import numpy as np
import random
import logging

from preoperativeSAM.cfg import get_config
from preoperativeSAM.models.model_dict import get_model
from preoperativeSAM.utils.visualization import get_model_parameters


def main(args):
    """
    Train the PreoperativeSAM model based on the provided arguments.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing model and task specifications.
        - modelname: Type of model to use (e.g., SAM).
        - task: Task or dataset name.
    """

    ## set logging level  ###########################################################
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])
    

    ## set opt and device  ##################################################################
    opt = get_config(args.task)
    device = torch.device(opt.device)

    ## set random seed for reproducibility  #######################################
    seed_value = opt.seed           
    np.random.seed(seed_value)                        # set random seed for numpy
    random.seed(seed_value)                           # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)    # avoid hash random
    torch.manual_seed(seed_value)                     # set random seed for CPU
    torch.cuda.manual_seed(seed_value)                # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)            # set random seed for all GPU
    torch.backends.cudnn.deterministic = True         # set random seed for convolution

    ## get the model  ##############################################################
    logging.info(f" Creating model: {args.modelname}")
    model = get_model(modelname=args.modelname, args=args, opt=opt)
    model.to(device)
    get_model_parameters(model)

    ## load the dataset  ##########################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PreoperativeSAM model')
    parser.add_argument('--modelname', default='SAM', type=str, help='type of model, e.g., SAM, ...')
    parser.add_argument('--task', default='PreDura', help='task or dataset name')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    args = parser.parse_args()    
    
    main(args)
