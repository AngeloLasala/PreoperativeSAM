"""
Ausiliar file to load different models
"""
from preoperativeSAM.models.segment_anything.build_sam import sam_model_registry
from preoperativeSAM.models.segment_anything_samus.build_sam_us import samus_model_registry
from preoperativeSAM.models.preoperative_samus.build_presamus import presamus_model_registry
from preoperativeSAM.models.preoperative_sam.build_presam import presam_model_registry
from preoperativeSAM.models.segment_anything_samus_autoprompt.build_samus import autosamus_model_registry
from preoperativeSAM.utils.visualization import get_model_parameters
import os

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint = os.path.join(opt.main_path, opt.sam_ckpt))

    elif modelname == "SAMUS":
        model = samus_model_registry['vit_b'](args=args, checkpoint = os.path.join(opt.main_path, opt.sam_ckpt))
    
    elif modelname == "PRESAMUS":
        model = presamus_model_registry['vit_b'](args=args, checkpoint = os.path.join(opt.main_path, opt.sam_ckpt))
    
    elif modelname == "PRESAM":
        model = presam_model_registry['vit_b'](args=args, checkpoint = os.path.join(opt.main_path, opt.sam_ckpt))
    
    elif modelname == "AutoSAMUS":
        model = autosamus_model_registry['vit_b'](args=args, checkpoint = os.path.join(opt.main_path, opt.load_path)) # checkpoint=opt.load_path)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model

if __name__ == "__main__":
    import torch
    import logging
    logging.basicConfig(level=logging.INFO)

    print('======== SAM ========')
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

    # dummy_input = [
    # {
    #     "image": torch.randn(1, 256, 256),  # 3xHxW immagine random
    #     "original_size": (256, 256),
    #     "point_coords": torch.tensor([[[100, 150], [400, 500]]], dtype=torch.float32),  # 1 batch, 2 punti
    #     "point_labels": torch.tensor([[1, 0]], dtype=torch.int64),  # 1 batch, 2 label
    #     "boxes": torch.tensor([[50, 60, 300, 400]], dtype=torch.float32),  # 1 box
    #     "mask_inputs": torch.randn(1, 1, 256, 256),  # maschera dummy
    # }
    # ]

    dummy_input = torch.randn(1, 1, 256, 256)
    t1 = torch.randn(1, 1, 2)
    t2 = torch.randint(0, 2, (1, 1))
    dummy_pt = (t1,t2)

    # Esegui il forward pass
    outputs = model(dummy_input, pt=dummy_pt)
    print("Output masks:", outputs['low_res_logits'].shape)
    print(outputs.keys())
    print()

    print('======== PRESAM ========')
    class args_sam:
        main_path = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE"
        dataset_name = "Dataset_iUS"       # note here i have two folder, pre and post
        save_folder = "checkpoints"
        result_folder = "results"
        tensorboard_folder = "tensorboard"
        sam_ckpt = "pretreined_SAM/sam_vit_b_01ec64.pth"
        encoder_input_size = 256

    model = get_model("PRESAM", args=args_sam, opt=args_sam)
    print('Full model')
    get_model_parameters(model)
    print()
    print('image encoder')
    image_encoder = model.image_encoder
    get_model_parameters(image_encoder)
    print()
    print('prompt encoder')
    prompt_encoder = model.prompt_encoder
    get_model_parameters(prompt_encoder)

     # Esegui il forward pass
    outputs = model(dummy_input, pt=dummy_pt, pre_imgs=dummy_input, intra_imgs=dummy_input)
    print(outputs.keys())
    print("Output masks:", outputs['low_res_logits'].shape)
    print()
    exit()



    print('======== SAMUS ========')
    class args_samus:
        main_path = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE"
        dataset_name = "Dataset_iUS"       # note here i have two folder, pre and post
        save_folder = "checkpoints"
        result_folder = "results"
        tensorboard_folder = "tensorboard"
        sam_ckpt = "pretreined_SAM/sam_vit_b_01ec64.pth"
        encoder_input_size = 256

    model = get_model("SAMUS", args=args_samus, opt=args_samus)
    print('Full model')
    get_model_parameters(model)
    print()
    print('image encoder')
    image_encoder = model.image_encoder
    get_model_parameters(image_encoder)
    print()
    print('prompt encoder')
    prompt_encoder = model.prompt_encoder
    get_model_parameters(prompt_encoder)
     # Esegui il forward pass
    outputs = model(dummy_input, pt=dummy_pt)
    print(outputs.keys())
    print("Output masks:", outputs['low_res_logits'].shape)
    print()

    print('======== PRESAMUS ========')
    class args_samus:
        main_path = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE"
        dataset_name = "Dataset_iUS"       # note here i have two folder, pre and post
        save_folder = "checkpoints"
        result_folder = "results"
        tensorboard_folder = "tensorboard"
        sam_ckpt = "pretreined_SAM/sam_vit_b_01ec64.pth"
        encoder_input_size = 256

    model = get_model("PRESAMUS", args=args_samus, opt=args_samus)
    print('Full model')
    get_model_parameters(model)
    print()
    print('image encoder')
    image_encoder = model.image_encoder
    get_model_parameters(image_encoder)
    print()
    prompt_encoder = model.prompt_encoder
    get_model_parameters(prompt_encoder)

     # Esegui il forward pass
    outputs = model(dummy_input, pt=dummy_pt, pre_imgs=dummy_input, intra_imgs=dummy_input)
    print(outputs.keys())
    print("Output masks:", outputs['low_res_logits'].shape)
    print()
    

    print('======== AutoSAMUS ========')
    class args_autosamus:
        main_path = "/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/Assistant_Researcher/AIRCARE"
        dataset_name = "Dataset_iUS"       # note here i have two folder, pre and post
        save_folder = "checkpoints"
        result_folder = "results"
        tensorboard_folder = "tensorboard"
        sam_ckpt = "pretreined_SAM/sam_vit_b_01ec64.pth"
        encoder_input_size = 256

    model = get_model("AutoSAMUS", args=args_autosamus, opt=args_autosamus)
    get_model_parameters(model)
    print('image encoder')
    image_encoder = model.image_encoder
    get_model_parameters(image_encoder)
    print()
    print('prompt generator')
    prompt_encoder = model.prompt_generator
    get_model_parameters(prompt_encoder)


    outputs = model(dummy_input, pt=dummy_pt)
    print(outputs.keys())
    print("Output masks:", outputs['masks'].shape)
    