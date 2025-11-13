"""
Main train script for PreoperativeSAM.

Adapted from Samus: https://github.com/xianlin7/SAMUS
"""
import argparse
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
from tqdm import tqdm
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from preoperativeSAM.cfg import get_config
from preoperativeSAM.models.model_dict import get_model
from preoperativeSAM.utils.visualization import get_model_parameters
from preoperativeSAM.utils.loss_functions.sam_loss import get_criterion
from preoperativeSAM.utils.generate_prompts import get_click_prompt
from preoperativeSAM.utils.evaluation import get_eval
from preoperativeSAM.dataset.dataset import JointTransform2D, IntroperativeiUS, PrePostiUS


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


    ## set opt and device and logging folder ##################################################################
    opt = get_config(args.task)
    device = torch.device(opt.device)

    if args.keep_log:
        logtimestr = time.strftime('%d-%m-%Y_%H-%M')  # initialize the tensorboard for record the training process
        boardpath = os.path.join(opt.main_path, opt.result_folder, args.modelname, opt.tensorboard_folder, opt.dataset_name, logtimestr)
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

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
    logging.info(f" Creating model: {args.modelname} ...")
    model = get_model(modelname=args.modelname, args=args, opt=opt)
    get_model_parameters(model)
    logging.info(' Done!\n')

    ## load the dataset  ##########################################################
    opt.batch_size = args.batch_size * args.n_gpu

    logging.info(' Creating train and val dataloader...')
    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, 
                                p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop,
                             p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = PrePostiUS(main_path = opt.main_path, 
                                    dataset_name = opt.dataset_name, 
                                    split = opt.train_split, 
                                    joint_transform = tf_train, 
                                    img_size = args.encoder_input_size)
    val_dataset = PrePostiUS(main_path = opt.main_path, 
                                    dataset_name = opt.dataset_name, 
                                    split = opt.val_split, 
                                    joint_transform = tf_val, 
                                    img_size = args.encoder_input_size)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    ## Train initialization ########################################################################
    logging.info(' Train initialization...')
    logging.info(f'  - device: {device}')
    logging.info(f'  - load pre-trained model: {opt.pre_trained}')
    logging.info(f'  - base lr: {args.base_lr}')
    logging.info(f'  - warmup: {args.warmup}')

    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.warmup:
        logging.info(f'  - warmup perido: {args.warmup_period}')
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
   
    criterion = get_criterion(modelname=args.modelname, opt=opt)
    logging.info(' Done!\n')

    ## Model training ################################################################################################
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)

    logging.info('Start training ...')        
    for epoch in range(opt.epochs):
        progress_bar = tqdm(total=len(trainloader), disable=False)
        progress_bar.set_description(f"Epoch {epoch + 1}/{opt.epochs}")

        model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)

            ## forward
            pred = model(imgs, pt, bbox)
            train_loss = criterion(pred, masks)
            
            ## backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            

            ## adjust learning rate if need
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

            progress_bar.update(1)
            logs = {"loss": train_loss.detach().item()}
            progress_bar.set_postfix(**logs)

        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        ## evaluation
        if epoch % opt.eval_freq == 0:
            model.eval()
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            logging.info(f' epoch [{epoch+1}/{opt.epochs}], val loss:{val_losses:.4f}')
            logging.info(f' epoch [{epoch+1}/{opt.epochs}], val dice:{mean_dice:.4f}')
            if args.keep_log:
                ## logger scalar
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice

                #logger images
                idxv = np.random.randint(0, len(val_dataset))
                datapack = val_dataset[idxv]
                imgs = datapack['image'].unsqueeze(0).to(dtype=torch.float32, device=opt.device)
                label = datapack['label'].unsqueeze(0).to(dtype=torch.float32, device=opt.device)

                pt = get_click_prompt(datapack, opt)

                with torch.no_grad():
                    pred = model(imgs, pt)

                predict = torch.sigmoid(pred['masks'])
                pred = predict.detach().cpu().numpy()[0, 0, :, :]  # (b, c, h, w)

                fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                axes[0].imshow(datapack['image'][0], cmap='gray')
                axes[0].set_title("Imm")

                axes[1].imshow(datapack['label'][0], cmap='gray')
                x, y = datapack["pt"][0]
                axes[1].scatter(x, y, c='red', s=80, marker='x', label='Click')
                axes[1].set_title("Label + pt")

                axes[2].imshow(pred, cmap='gray')
                axes[2].scatter(x, y, c='red', s=80, marker='x', label='Click')
                axes[2].set_title("Pred + pt")

                TensorWriter.add_figure('Image', fig, epoch)

            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                save_path = os.path.join(opt.main_path, opt.result_folder, args.modelname, opt.save_folder, opt.dataset_name, logtimestr)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(save_path, f'{args.modelname}_best')
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch == (opt.epochs-1):
            ## save last model
            save_path = os.path.join(opt.main_path, opt.result_folder, args.modelname, opt.save_folder, opt.dataset_name, logtimestr)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, f'{args.modelname}_last')
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

            ## save args
            args_path = os.path.join(save_path + "_args.txt")
            with open(args_path, "w") as f:
                f.write("#### Training Arguments ####\n")
                for k, v in vars(args).items():
                    f.write(f"{k}: {v}\n")

            ## save opt
            opt_path = os.path.join(save_path + "_opt.txt")
            with open(opt_path, "w") as f:
                f.write("#### Configuration Options ####\n")
                # se opt è un oggetto tipo Namespace o una classe con attributi
                if hasattr(opt, '__dict__'):
                    for k, v in vars(opt).items():
                        f.write(f"{k}: {v}\n")
                # se opt è un dizionario
                elif isinstance(opt, dict):
                    for k, v in opt.items():
                        f.write(f"{k}: {v}\n")

            logging.info(f"Saved last model and configuration to {save_path}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PreoperativeSAM model')
    parser.add_argument('--modelname', default='SAM', type=str, help='type of model, e.g., SAM, ...')
    parser.add_argument('--encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('--low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='PreDura', help='task or dataset name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr, default=False') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid when warmup is activated')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    parser.add_argument('--keep_log', action='store_true', help='keep the loss&lr&dice during training or not, default=False')

    args = parser.parse_args()
    
    main(args)
