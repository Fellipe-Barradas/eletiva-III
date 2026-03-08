# Scaled Dot-Product Attention

Implementation of the Scaled Dot-Product Attention mechanism as described
in the paper "Attention Is All You Need" (Vaswani et al., 2017).

---

## Repository Structure

    .
    attention.py        -> Main implementation
    test_attention.py   -> Unit tests
    Makefile            -> Build and test automation
    README.md           -> This documentation

---

## How to Run

### Using Makefile (Recommended)

The project includes a Makefile to facilitate common tasks:

    # View all available commands
    make help

    # Install dependencies
    make install

    # Run tests
    make test

    # Run usage example
    make run

    # Clean temporary files
    make clean

### Manually

#### Prerequisites

- Python 3.10+
- NumPy

    pip install numpy

#### Run tests

    python test_attention.py

Expected output:

    =======================================================
      Tests: Scaled Dot-Product Attention
    =======================================================
    ...
      Result: 12/12 tests passed.
      All tests PASSED successfully!
    =======================================================

#### Use the function directly

    import numpy as np
    from attention import scaled_dot_product_attention

    Q = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    V = np.array([[1.0, 0.0],
                  [0.0, 1.0]])

    output, weights = scaled_dot_product_attention(Q, K, V)
    print("Output:\n", output)
    print("Attention Weights:\n", weights)

---

## Reference Equation

    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V

---

## How sqrt(d_k) Normalization Was Applied

The dot product QK^T grows in magnitude as the dimension d_k increases.
This pushes Softmax into regions of very small gradients (saturated gradients).

To compensate, each element of the scores matrix is divided by sqrt(d_k)
before applying Softmax:

    scaling_factor = np.sqrt(d_k)       # e.g.: d_k=64 -> scaling=8.0
    scores = (Q @ K.T) / scaling_factor # shape: (n_queries, n_keys)

This division keeps the variance of the scores approximately constant (~1),
independent of d_k, ensuring more stable training.

Softmax is then applied row by row (each query normalizes its attention
distribution over all keys independently):

    attention_weights = softmax(scores)  # softmax along axis=-1

For numerical stability, Softmax subtracts the maximum value from each row
before exponentiation (standard technique, without changing the mathematical result).

---

## Example Input and Expected Output

    Input:

    Q = [[1.0, 0.0],    K = [[1.0, 0.0],    V = [[1.0, 0.0],
         [0.0, 1.0]]         [0.0, 1.0]]         [0.0, 1.0]]

    Attention Weights (output):

    [[0.6742, 0.3258],
     [0.3258, 0.6742]]

    The first query pays more attention to the first key (0.67 vs 0.33),
    and the second query does the opposite -- expected behavior since
    Q and K are identity matrices.

    Output:

    [[0.6742, 0.0000],
     [0.0000, 0.6742]]

---

## Reference

Vaswani, A. et al. Attention Is All You Need. NeurIPS 2017.
https://arxiv.org/abs/1706.03762
