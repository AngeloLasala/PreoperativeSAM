"""
Ausiliar file to load different models
"""
from preoperativeSAM.models.segment_anything.build_sam import sam_model_registry
from preoperativeSAM.utils.visualization import get_model_parameters

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=None)
    ## here you can add more models if needed
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model

if __name__ == "__main__":
    import torch

    model = get_model("SAM", args=None, opt=None)
    get_model_parameters(model)

    dummy_input = [
    {
        "image": torch.randn(3, 256, 256),  # 3xHxW immagine random
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