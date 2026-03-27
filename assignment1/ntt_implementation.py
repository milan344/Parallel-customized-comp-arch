"""
Negacyclic Number Theoretic Transform (NTT) implementation with tests.
"""
import jax.numpy as jnp
import numpy as np

# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------
def mod_add(a, b, q):
    """Return (a + b) mod q, elementwise."""
    return (a + b) % q

def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise."""
    return (a - b) % q

def mod_mul(a, b, q):
    """Return (a * b) mod q, elementwise."""
    return (a * b) % q

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def bit_reverse(n, num_bits):
    """Reverse the bits of n using num_bits bits."""
    result = 0
    for i in range(num_bits):
        if n & (1 << i):
            result |= 1 << (num_bits - 1 - i)
    return result

def compute_twiddles_negacyclic(N, psi, q):
    """
    Compute twiddle factors for negacyclic NTT.
    Twiddles are powers of ψ^2 = ω in bit-reversed order.
    """
    num_bits = int(np.log2(N))
    twiddles = np.zeros(N, dtype=np.int64)
    
    omega = pow(psi, 2, q)  # ω = ψ^2
    
    for i in range(N):
        # Bit-reverse to get proper ordering for CT butterfly
        rev_i = bit_reverse(i, num_bits)
        twiddles[i] = pow(omega, rev_i, q)
    
    return jnp.array(twiddles)

# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------
def ntt(x, *, q, psi_powers, twiddles):
    """
    Compute the forward negacyclic NTT using Cooley-Tukey algorithm.
    
    The negacyclic NTT first applies a "twist" (multiply by ψ^n),
    then performs a standard cyclic NTT.
    
    Args:
        x: Input coefficients, shape (batch, N), values in [0, q)
        q: Prime modulus satisfying (q - 1) % 2N == 0
        psi_powers: Precomputed ψ^n table, shape (N,)
        twiddles: Precomputed twiddle factors (bit-reversed ω powers), shape (N,)
    
    Returns:
        jnp.ndarray: NTT output, same shape as input
    """
    batch_size, N = x.shape
    num_stages = int(np.log2(N))
    
    # Step 1: Apply negacyclic twist - multiply by ψ^i for position i
    # This converts negacyclic convolution to cyclic convolution
    y = mod_mul(x, psi_powers[jnp.newaxis, :], q)
    
    # Step 2: Cooley-Tukey butterfly network
    # We perform log2(N) stages of butterflies
    for stage in range(num_stages):
        # At each stage, butterfly pairs are separated by 'distance'
        distance = 1 << stage  # 2^stage
        
        # Each block processes 2*distance elements
        block_size = distance << 1  # 2^(stage+1)
        
        # Number of blocks at this stage
        num_blocks = N // block_size
        
        # Process each block
        for block in range(num_blocks):
            block_start = block * block_size
            
            # Process butterflies within the block
            for j in range(distance):
                # Indices for the butterfly pair
                idx_top = block_start + j
                idx_bottom = idx_top + distance
                
                # Twiddle factor index
                # At stage s, we use every 2^(log2(N) - s - 1)-th twiddle
                twiddle_idx = j << (num_stages - stage - 1)
                w = twiddles[twiddle_idx]
                
                # Cooley-Tukey butterfly:
                # top_new = top + bottom * w
                # bottom_new = top - bottom * w
                top = y[:, idx_top]
                bottom = y[:, idx_bottom]
                
                temp = mod_mul(bottom, w, q)
                
                y = y.at[:, idx_top].set(mod_add(top, temp, q))
                y = y.at[:, idx_bottom].set(mod_sub(top, temp, q))
    
    # Step 3: Bit-reverse permutation to get normal order output
    num_bits = int(np.log2(N))
    reversed_indices = jnp.array([bit_reverse(i, num_bits) for i in range(N)])
    y = y[:, reversed_indices]
    
    return y

def intt(y, *, q, psi_powers_inv, twiddles_inv, n_inv):
    """
    Compute the inverse negacyclic NTT using Gentleman-Sande algorithm.
    
    Args:
        y: NTT coefficients, shape (batch, N)
        q: Prime modulus
        psi_powers_inv: Precomputed ψ^(-n) table
        twiddles_inv: Inverse twiddle factors
        n_inv: Modular inverse of N mod q
    
    Returns:
        jnp.ndarray: Original coefficients
    """
    batch_size, N = y.shape
    num_stages = int(np.log2(N))
    
    # Step 1: Bit-reverse the input
    num_bits = int(np.log2(N))
    reversed_indices = jnp.array([bit_reverse(i, num_bits) for i in range(N)])
    x = y[:, reversed_indices]
    
    # Step 2: Gentleman-Sande butterfly stages
    for stage in range(num_stages):
        # Distance between butterfly pairs
        distance = N >> (stage + 1)
        
        # Block size
        block_size = distance << 1
        
        # Number of blocks
        num_blocks = N // block_size
        
        # Process each block
        for block in range(num_blocks):
            block_start = block * block_size
            
            # Process butterflies within block
            for j in range(distance):
                idx_top = block_start + j
                idx_bottom = idx_top + distance
                
                # Twiddle factor index
                twiddle_idx = j << stage
                w_inv = twiddles_inv[twiddle_idx]
                
                # Gentleman-Sande butterfly:
                # top_new = top + bottom
                # bottom_new = (top - bottom) * w_inv
                top = x[:, idx_top]
                bottom = x[:, idx_bottom]
                
                x = x.at[:, idx_top].set(mod_add(top, bottom, q))
                x = x.at[:, idx_bottom].set(mod_mul(mod_sub(top, bottom, q), w_inv, q))
    
    # Step 3: Remove negacyclic twist (multiply by ψ^(-i))
    x = mod_mul(x, psi_powers_inv[jnp.newaxis, :], q)
    
    # Step 4: Scale by N^(-1)
    x = mod_mul(x, n_inv, q)
    
    return x

def prepare_tables(*, q, psi_powers, twiddles):
    """
    Optional one-time table preparation.
    """
    return psi_powers, twiddles

# -----------------------------------------------------------------------------
# Setup Functions
# -----------------------------------------------------------------------------
def find_primitive_root(q):
    """Find a primitive root (generator) of Z_q."""
    known_roots = {
        7681: 17,
        12289: 11,
        3329: 17,
        8380417: 3
    }
    
    if q in known_roots:
        return known_roots[q]
    
    raise ValueError(f"Primitive root not known for q={q}")

def find_psi(q, n):
    """
    Find ψ, the primitive 2n-th root of unity.
    
    For prime q where (q-1) is divisible by 2n:
    ψ = g^((q-1)/(2n)) where g is a primitive root of q
    """
    if (q - 1) % (2 * n) != 0:
        raise ValueError(f"2n={2*n} does not divide q-1={q-1}")
    
    g = find_primitive_root(q)
    exponent = (q - 1) // (2 * n)
    psi = pow(g, exponent, q)
    
    # Verify
    assert pow(psi, 2*n, q) == 1, f"ψ^(2n) should equal 1"
    assert pow(psi, n, q) == q - 1, f"ψ^n should equal -1"
    
    return psi

def setup_ntt_params(N, q):
    """
    Set up all parameters needed for NTT.
    
    Args:
        N: Polynomial degree (must be power of 2)
        q: Prime modulus where (q-1) % (2*N) == 0
    
    Returns:
        dict with psi, psi_powers, and twiddles
    """
    if N & (N - 1) != 0:
        raise ValueError(f"N={N} must be a power of 2")
    
    # Find ψ
    psi = find_psi(q, N)
    
    # Compute ψ^i for i = 0, 1, ..., N-1
    psi_powers = jnp.array([pow(psi, i, q) for i in range(N)])
    
    # Compute twiddle factors
    twiddles = compute_twiddles_negacyclic(N, psi, q)
    
    return {
        'psi': psi,
        'psi_powers': psi_powers,
        'twiddles': twiddles
    }

def setup_intt_params(N, q, psi):
    """Setup parameters for inverse NTT."""
    # Compute ψ^(-1)
    psi_inv = pow(psi, q-2, q)
    
    # Compute ψ^(-i) for i = 0, 1, ..., N-1
    psi_powers_inv = jnp.array([pow(psi_inv, i, q) for i in range(N)])
    
    # Compute ω^(-1) where ω = ψ^2
    omega_inv = pow(pow(psi, 2, q), q-2, q)
    
    # Compute inverse twiddle factors
    num_bits = int(np.log2(N))
    twiddles_inv = np.zeros(N, dtype=np.int64)
    
    for i in range(N):
        rev_i = bit_reverse(i, num_bits)
        twiddles_inv[i] = pow(omega_inv, rev_i, q)
    
    # Compute N^(-1) mod q
    n_inv = pow(N, q-2, q)
    
    return {
        'psi_powers_inv': psi_powers_inv,
        'twiddles_inv': jnp.array(twiddles_inv),
        'n_inv': n_inv
    }

# -----------------------------------------------------------------------------
# Testing Functions
# -----------------------------------------------------------------------------
def test_ntt_small():
    """Test with n=4, q=7681 (from the paper examples)."""
    print("=" * 70)
    print("TEST 1: Small example (n=4, q=7681)")
    print("=" * 70)
    
    N = 4
    q = 7681
    
    params = setup_ntt_params(N, q)
    psi = params['psi']
    psi_powers = params['psi_powers']
    twiddles = params['twiddles']
    
    print(f"N = {N}")
    print(f"q = {q}")
    print(f"ψ = {psi}")
    print(f"ψ powers: {psi_powers}")
    print(f"Twiddles: {twiddles}")
    print()
    
    print("Verifying ψ properties:")
    print(f"  ψ^2 = {pow(psi, 2, q)} (ω, the 4th root of unity)")
    print(f"  ψ^4 = {pow(psi, 4, q)} (should be {q-1} = -1 mod q)")
    print(f"  ψ^8 = {pow(psi, 8, q)} (should be 1)")
    print()
    
    x = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    print(f"Input: {x[0]}")
    
    expected = jnp.array([1467, 2807, 3471, 7621])
    
    result = ntt(x, q=q, psi_powers=psi_powers, twiddles=twiddles)
    print(f"Output: {result[0]}")
    print(f"Expected: {expected}")
    
    if jnp.allclose(result[0], expected):
        print("✓ TEST PASSED!")
    else:
        print("✗ TEST FAILED!")
        print(f"Difference: {result[0] - expected}")
    print()
    return jnp.allclose(result[0], expected)

def test_ntt_second_input():
    """Test with second input from paper: [5, 6, 7, 8]."""
    print("=" * 70)
    print("TEST 2: Second example (n=4, q=7681)")
    print("=" * 70)
    
    N = 4
    q = 7681
    
    params = setup_ntt_params(N, q)
    psi_powers = params['psi_powers']
    twiddles = params['twiddles']
    
    x = jnp.array([[5, 6, 7, 8]], dtype=jnp.int32)
    print(f"Input: {x[0]}")
    
    expected = jnp.array([2489, 7489, 6478, 6607])
    
    result = ntt(x, q=q, psi_powers=psi_powers, twiddles=twiddles)
    print(f"Output: {result[0]}")
    print(f"Expected: {expected}")
    
    if jnp.allclose(result[0], expected):
        print("✓ TEST PASSED!")
    else:
        print("✗ TEST FAILED!")
        print(f"Difference: {result[0] - expected}")
    print()
    return jnp.allclose(result[0], expected)

def test_ntt_multiplication():
    """Test polynomial multiplication using NTT."""
    print("=" * 70)
    print("TEST 3: Polynomial multiplication via NTT")
    print("=" * 70)
    
    N = 4
    q = 7681
    
    params = setup_ntt_params(N, q)
    psi_powers = params['psi_powers']
    twiddles = params['twiddles']
    
    g = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    h = jnp.array([[5, 6, 7, 8]], dtype=jnp.int32)
    
    print(f"g = {g[0]}")
    print(f"h = {h[0]}")
    print()
    
    g_ntt = ntt(g, q=q, psi_powers=psi_powers, twiddles=twiddles)
    h_ntt = ntt(h, q=q, psi_powers=psi_powers, twiddles=twiddles)
    
    print(f"NTT(g) = {g_ntt[0]}")
    print(f"NTT(h) = {h_ntt[0]}")
    print()
    
    result_ntt = mod_mul(g_ntt, h_ntt, q)
    print(f"NTT(g) ⊙ NTT(h) = {result_ntt[0]}")
    
    expected_ntt = jnp.array([2888, 6407, 2851, 2992])
    print(f"Expected (in NTT domain): {expected_ntt}")
    print()
    
    if jnp.allclose(result_ntt[0], expected_ntt):
        print("✓ Element-wise multiplication in NTT domain correct!")
    else:
        print("✗ Element-wise multiplication failed!")
        print(f"Difference: {result_ntt[0] - expected_ntt}")
    
    print()
    return jnp.allclose(result_ntt[0], expected_ntt)

def test_inverse_ntt():
    """Test inverse NTT to recover original polynomial."""
    print("=" * 70)
    print("TEST 4: Inverse NTT")
    print("=" * 70)
    
    N = 4
    q = 7681
    
    params = setup_ntt_params(N, q)
    psi = params['psi']
    psi_powers = params['psi_powers']
    twiddles = params['twiddles']
    
    intt_params = setup_intt_params(N, q, psi)
    
    x_orig = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    print(f"Original input: {x_orig[0]}")
    
    x_ntt = ntt(x_orig, q=q, psi_powers=psi_powers, twiddles=twiddles)
    print(f"After NTT: {x_ntt[0]}")
    
    x_recovered = intt(x_ntt, q=q, psi_powers_inv=intt_params['psi_powers_inv'],
                       twiddles_inv=intt_params['twiddles_inv'], 
                       n_inv=intt_params['n_inv'])
    print(f"After INTT: {x_recovered[0]}")
    
    if jnp.allclose(x_recovered[0], x_orig[0]):
        print("✓ TEST PASSED! INTT correctly recovers original input.")
    else:
        print("✗ TEST FAILED!")
        print(f"Difference: {x_recovered[0] - x_orig[0]}")
    
    print()
    return jnp.allclose(x_recovered[0], x_orig[0])

def test_full_multiplication_pipeline():
    """Test complete polynomial multiplication: NTT -> multiply -> INTT."""
    print("=" * 70)
    print("TEST 5: Full multiplication pipeline (NTT -> multiply -> INTT)")
    print("=" * 70)
    
    N = 4
    q = 7681
    
    params = setup_ntt_params(N, q)
    psi = params['psi']
    psi_powers = params['psi_powers']
    twiddles = params['twiddles']
    
    intt_params = setup_intt_params(N, q, psi)
    
    g = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    h = jnp.array([[5, 6, 7, 8]], dtype=jnp.int32)
    
    print(f"g(x) = 1 + 2x + 3x² + 4x³")
    print(f"h(x) = 5 + 6x + 7x² + 8x³")
    print(f"g = {g[0]}")
    print(f"h = {h[0]}")
    print()
    
    g_ntt = ntt(g, q=q, psi_powers=psi_powers, twiddles=twiddles)
    h_ntt = ntt(h, q=q, psi_powers=psi_powers, twiddles=twiddles)
    print(f"NTT(g) = {g_ntt[0]}")
    print(f"NTT(h) = {h_ntt[0]}")
    
    result_ntt = mod_mul(g_ntt, h_ntt, q)
    print(f"NTT(g) ⊙ NTT(h) = {result_ntt[0]}")
    
    result = intt(result_ntt, q=q, psi_powers_inv=intt_params['psi_powers_inv'],
                  twiddles_inv=intt_params['twiddles_inv'], n_inv=intt_params['n_inv'])
    print(f"INTT(result) = {result[0]}")
    
    expected = jnp.array([7625, 7645, 2, 60])
    print(f"Expected: {expected}")
    
    result_signed = np.array(result[0], dtype=np.int32)
    expected_signed = np.array(expected, dtype=np.int32)
    for i in range(len(result_signed)):
        if result_signed[i] > q // 2:
            result_signed[i] -= q
        if expected_signed[i] > q // 2:
            expected_signed[i] -= q
    
    print(f"Result (signed): {result_signed}")
    print(f"Expected (signed): {expected_signed}")
    print()
    
    if jnp.allclose(result[0], expected):
        print("✓ TEST PASSED! Complete multiplication pipeline works!")
        print(f"  Negacyclic product: {result_signed}")
    else:
        print("✗ TEST FAILED!")
        print(f"Difference: {result[0] - expected}")
    
    print()
    return jnp.allclose(result[0], expected)

def test_larger_size():
    """Test with larger polynomial size (n=8)."""
    print("=" * 70)
    print("TEST 6: Larger size (n=8, q=7681)")
    print("=" * 70)
    
    N = 8
    q = 7681
    
    params = setup_ntt_params(N, q)
    psi = params['psi']
    psi_powers = params['psi_powers']
    twiddles = params['twiddles']
    
    print(f"N = {N}")
    print(f"q = {q}")
    print(f"ψ = {psi}")
    print()
    
    x = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    print(f"Input: {x[0]}")
    
    result = ntt(x, q=q, psi_powers=psi_powers, twiddles=twiddles)
    print(f"NTT output: {result[0]}")
    
    intt_params = setup_intt_params(N, q, psi)
    x_recovered = intt(result, q=q, psi_powers_inv=intt_params['psi_powers_inv'],
                       twiddles_inv=intt_params['twiddles_inv'],
                       n_inv=intt_params['n_inv'])
    
    print(f"Recovered: {x_recovered[0]}")
    
    if jnp.allclose(x_recovered[0], x[0]):
        print("✓ TEST PASSED! Round-trip works for n=8.")
    else:
        print("✗ TEST FAILED!")
        print(f"Difference: {x_recovered[0] - x[0]}")
    
    print()
    return jnp.allclose(x_recovered[0], x[0])

def test_batch_processing():
    """Test batch processing of multiple polynomials."""
    print("=" * 70)
    print("TEST 7: Batch processing")
    print("=" * 70)
    
    N = 4
    q = 7681
    
    params = setup_ntt_params(N, q)
    psi_powers = params['psi_powers']
    twiddles = params['twiddles']
    
    x = jnp.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [2, 4, 6, 8]
    ], dtype=jnp.int32)
    
    print(f"Input batch (3 polynomials):")
    for i in range(3):
        print(f"  [{i}]: {x[i]}")
    print()
    
    result = ntt(x, q=q, psi_powers=psi_powers, twiddles=twiddles)
    
    print(f"NTT outputs:")
    for i in range(3):
        print(f"  [{i}]: {result[i]}")
    
    expected = jnp.array([
        [1467, 2807, 3471, 7621],
        [2489, 7489, 6478, 6607],
        [0, 0, 0, 0]  # We'll compute this
    ])
    
    if jnp.allclose(result[0], expected[0]) and jnp.allclose(result[1], expected[1]):
        print("\n✓ TEST PASSED! Batch processing works correctly.")
    else:
        print("\n✗ TEST FAILED!")
    
    print()
    return jnp.allclose(result[0], expected[0]) and jnp.allclose(result[1], expected[1])

# -----------------------------------------------------------------------------
# Main Test Runner
# -----------------------------------------------------------------------------
def run_all_tests():
    """Run all tests and report results."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 15 + "NTT IMPLEMENTATION TESTS" + " " * 29 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    tests = [
        ("Small Example (n=4)", test_ntt_small),
        ("Second Input (n=4)", test_ntt_second_input),
        ("NTT Multiplication", test_ntt_multiplication),
        ("Inverse NTT", test_inverse_ntt),
        ("Full Pipeline", test_full_multiplication_pipeline),
        ("Larger Size (n=8)", test_larger_size),
        ("Batch Processing", test_batch_processing),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n")
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
    
    print("=" * 70)
    print()

if __name__ == "__main__":
    run_all_tests()