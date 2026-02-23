
import numpy as np


def softmax(x):

    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
   
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K and V must be 2D matrices (n, d).")

    n_queries, d_k = Q.shape
    n_keys_k, d_k_k = K.shape
    n_keys_v, d_v = V.shape

    if d_k != d_k_k:
        raise ValueError(
            f"The d_k dimension of Q ({d_k}) and K ({d_k_k}) must be equal."
        )
    if n_keys_k != n_keys_v:
        raise ValueError(
            f"The number of rows in K ({n_keys_k}) and V ({n_keys_v}) must be equal."
        )

    scaling_factor = np.sqrt(d_k)
    scores = (Q @ K.T) / scaling_factor 

    attention_weights = softmax(scores) 

    output = attention_weights @ V  

    return output, attention_weights
