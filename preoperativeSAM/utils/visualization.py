"""
Ausiliar funxtion to visualize img and output and the info about the model
"""
import torch
import logging

def get_model_parameters(model):
    """
    Get the number of total, trainable and frozen parameters of a model.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to analyze.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params
    logging.info(f" Number of params: {total_params/1e6:.2f}M")
    logging.info(f"   - trainable: {trainable_params/1e6:.2f}M")
    logging.info(f"   - untrainable: {frozen_params/1e6:.2f}M")