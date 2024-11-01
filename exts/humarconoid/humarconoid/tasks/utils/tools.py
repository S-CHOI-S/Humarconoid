"""*******************************************************************************
* HumARConoid
*
* Advanced Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
*******************************************************************************"""

"* Authors: Sol Choi (Jennifer) *"

import torch

def backlash(input_tensor, threshold):
    """
    Simulates backlash by ignoring changes smaller than a specified threshold.
    
    Parameters:
        input_tensor (torch.Tensor): The input tensor with position or velocity data.
        threshold (float): The backlash threshold. Changes below this value will be ignored.
        
    Returns:
        torch.Tensor: The output tensor after applying backlash.
    """
    # Define the output tensor with the same shape as the input
    output_tensor = torch.zeros_like(input_tensor)
    
    # Apply backlash: changes smaller than the threshold are set to zero
    mask = torch.abs(input_tensor) > threshold
    output_tensor[mask] = input_tensor[mask]
    
    return output_tensor
