"""
Configuration class

local main_path:    "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE"
cluster main_path:  "/leonardo_work/IscrC_AIM-ORAL/Angelo/AIRCARE"
"""
class PreDura:
    ## Paths    ##########################################################################
    main_path = "/leonardo_work/IscrC_AIM-ORAL/Angelo/AIRCARE"
    dataset_name = "Dataset_iUS"       # note here i have two folder, pre and post
    save_folder = "checkpoints"
    result_folder = "results"
    tensorboard_folder = "tensorboard"
    sam_ckpt = "pretreined_SAM/sam_vit_b_01ec64.pth"

    ## Training parameters    ##########################################################
    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                      # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # then umber of classes (background + foreground)
    img_size = 256                      # the input size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluate the model, slice level or patient level
    seed = 42                           # random seed
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class PrePost:
    ## Paths    ##########################################################################
    main_path = "/leonardo_work/IscrC_AIM-ORAL/Angelo/AIRCARE"
    dataset_name = "Dataset_iUS"       # note here i have two folder, pre and post
    save_folder = "checkpoints"
    result_folder = "results"
    tensorboard_folder = "tensorboard"
    sam_ckpt = "pretreined_SAM/sam_vit_b_01ec64.pth"
    load_path = "results/Dataset_iUS/SAMUS/checkpoints/post_15-11-2025_11-56/SAMUS_best.pth"

    ## Training parameters    ##########################################################
    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                      # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # then umber of classes (background + foreground)
    img_size = 256                      # the input size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluate the model, slice level or patient level
    seed = 42                           # random seed
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"


def get_config(task):
    """
    Get the configuration class based on the task name.
    """
    if task == "PreDura":
        return PreDura()
    
    elif task == "PrePostiUS":
        return PrePost()
        
    else:
        raise RuntimeError("Could not find the task:", task)