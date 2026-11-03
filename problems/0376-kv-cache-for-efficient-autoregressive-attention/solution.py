import torch

def kv_cache_attention_step(x_new: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step
    
    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """
    q = torch.matmul(x_new, W_Q)
    k = torch.matmul(x_new, W_K)
    v = torch.matmul(x_new, W_V)

    if cache is None:
        K_cache = k.unsqueeze(0)
        V_cache = v.unsqueeze(0)
    else:
        K_cache = torch.cat([cache[0], k.unsqueeze(0)], dim=0)
        V_cache = torch.cat([cache[1], v.unsqueeze(0)], dim=0)
    
    d_k = W_Q.shape[1]
    scores = torch.matmul(q, K_cache.T) / torch.sqrt(torch.tensor(d_k, dtype=x_new.dtype))

    weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(weights, V_cache)
    return output, (K_cache, V_cache)