"""
Ausiliar file to load different models
"""
from preoperativeSAM.models.segment_anything.build_sam import sam_model_registry
from preoperativeSAM.models.segment_anything_samus.build_sam_us import samus_model_registry
from preoperativeSAM.utils.visualization import get_model_parameters
import os

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint = os.path.join(opt.main_path, opt.sam_ckpt))

    elif modelname == "SAMUS":
        model = samus_model_registry['vit_b'](args=args, checkpoint = os.path.join(opt.main_path, opt.sam_ckpt))
    
    ## here you can add more models if needed
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model

if __name__ == "__main__":
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)

    print('SAM')
    class args_sam:
        main_path = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE"
        dataset_name = "Dataset_iUS"       # note here i have two folder, pre and post
        save_folder = "checkpoints"
        result_folder = "results"
        tensorboard_folder = "tensorboard"
        sam_ckpt = "pretreined_SAM/sam_vit_b_01ec64.pth"

    model = get_model("SAM", args=None, opt=args_sam)
    get_model_parameters(model)
    print()

    dummy_input = [
    {
        "image": torch.randn(1, 256, 256),  # 3xHxW immagine random
        "original_size": (256, 256),
        "point_coords": torch.tensor([[[100, 150], [400, 500]]], dtype=torch.float32),  # 1 batch, 2 punti
        "point_labels": torch.tensor([[1, 0]], dtype=torch.int64),  # 1 batch, 2 label
        "boxes": torch.tensor([[50, 60, 300, 400]], dtype=torch.float32),  # 1 box
        "mask_inputs": torch.randn(1, 1, 256, 256),  # maschera dummy
    }
    ]

    # Esegui il forward pass
    outputs = model(dummy_input, multimask_output=True)
    print("Output keys:", outputs[0].keys())

    print('SAMUS')
    class args_samus:
        encoder_input_size = 256

    model = get_model("SAMUS", args=args_samus, opt=None)
    get_model_parameters(model)
    image_encoder = model.image_encoder
    get_model_parameters(image_encoder)
    print()