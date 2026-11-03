import torch

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    seq_len, d = x.shape
     
    i = torch.arange(0, d//2, dtype=torch.float32, device=x.device)
    theta = 1.0 / (base**(2.0 * i/d))

    angles = torch.outer(positions.float(), theta)

    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    x_rot_even = x_even * cos_angles - x_odd * sin_angles
    x_rot_odd = x_even * sin_angles + x_odd * cos_angles

    result = torch.zeros_like(x)
    result[:, 0::2] = x_rot_even
    result[:, 1::2] = x_rot_odd
    return result