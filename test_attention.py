import numpy as np
from attention import scaled_dot_product_attention, softmax

_passed = 0
_failed = 0


def assert_allclose(actual, expected, atol=1e-6, label=""):
    global _passed, _failed
    if np.allclose(actual, expected, atol=atol):
        print(f"  [PASSED] {label}")
        _passed += 1
    else:
        print(f"  [FAILED] {label}")
        print(f"           Expected:\n{expected}")
        print(f"           Got:\n{actual}")
        _failed += 1


def assert_true(condition, label=""):
    global _passed, _failed
    if condition:
        print(f"  [PASSED] {label}")
        _passed += 1
    else:
        print(f"  [FAILED] {label}")
        _failed += 1

def test_softmax_rows():
    print("\n=== Test 1: Softmax row by row ===")

    x = np.array([[1.0, 2.0, 3.0],
                   [1.0, 1.0, 1.0]])

    result = softmax(x)

    
    row_sums = result.sum(axis=1)
    assert_allclose(row_sums, np.ones(2), label="Rows sum to 1")

    
    assert_true((result >= 0).all() and (result <= 1).all(),
                label="Values between 0 and 1")

    
    expected_uniform = np.array([1/3, 1/3, 1/3])
    assert_allclose(result[1], expected_uniform, label="Uniform input -> equal probabilities")


def test_attention_simple():
   
    print("\n=== Test 2: Simple numerical example ===")

    Q = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    V = np.array([[1.0, 0.0],
                  [0.0, 1.0]])

    output, weights = scaled_dot_product_attention(Q, K, V)

    d_k = 2
    scaling = np.sqrt(d_k)
    scores = (Q @ K.T) / scaling
    expected_weights = softmax(scores)
    expected_output = expected_weights @ V

    assert_allclose(weights, expected_weights, label="Correct attention weights")
    assert_allclose(output, expected_output,   label="Correct attention output")

    
    assert_allclose(weights.sum(axis=1), np.ones(Q.shape[0]),
                    label="Weight rows sum to 1")


def test_scaling_factor():
   
    print("\n=== Test 3: Scaling factor sqrt(d_k) ===")

    np.random.seed(42)
    Q = np.random.randn(3, 4)
    K = np.random.randn(3, 4)
    V = np.random.randn(3, 4)

    output_scaled, weights_scaled = scaled_dot_product_attention(Q, K, V)

   
    scores_unscaled = Q @ K.T
    weights_unscaled = softmax(scores_unscaled)
    output_unscaled = weights_unscaled @ V

    
    different = not np.allclose(weights_scaled, weights_unscaled)
    assert_true(different, label="Scaled output differs from unscaled")

   
    scores_expected = (Q @ K.T) / np.sqrt(4)
    expected_weights = softmax(scores_expected)
    assert_allclose(weights_scaled, expected_weights,
                    label="Scaling sqrt(d_k=4) = 2.0 applied correctly")

def test_output_shapes():
    print("\n=== Test 4: Input and output shapes ===")

    
    Q = np.random.randn(3, 8)
    K = np.random.randn(5, 8)
    V = np.random.randn(5, 6)

    output, weights = scaled_dot_product_attention(Q, K, V)

    assert_true(output.shape == (3, 6),
                label=f"Output shape (3, 6) -- obtido: {output.shape}")
    assert_true(weights.shape == (3, 5),
                label=f"Weights shape (3, 5) -- obtido: {weights.shape}")

def test_invalid_inputs():
    print("\n=== Test 5: Invalid inputs ===")

    
    try:
        scaled_dot_product_attention(
            np.ones((2, 3)), np.ones((2, 4)), np.ones((2, 4))
        )
        assert_true(False, label="Should raise ValueError (incompatible d_k)")
    except ValueError:
        assert_true(True, label="ValueError raised for incompatible d_k")


    try:
        scaled_dot_product_attention(
            np.ones((2, 4)), np.ones((3, 4)), np.ones((5, 4))
        )
        assert_true(False, label="Should raise ValueError (incompatible n_keys)")
    except ValueError:
        assert_true(True, label="ValueError raised for incompatible n_keys")



if __name__ == "__main__":
    print("=" * 55)
    print("  Tests: Scaled Dot-Product Attention")
    print("=" * 55)

    test_softmax_rows()
    test_attention_simple()
    test_scaling_factor()
    test_output_shapes()
    test_invalid_inputs()

    print("\n" + "=" * 55)
    total = _passed + _failed
    print(f"  Result: {_passed}/{total} tests passed.")
    if _failed == 0:
        print("  All tests PASSED successfully!")
    else:
        print(f"  {_failed} test(s) FAILED.")
    print("=" * 55)
