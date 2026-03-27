"""
Negacyclic Number Theoretic Transform (NTT) implementation.

Optimizations:
  1. Montgomery multiplication — replaces slow % (division) with
     3 fast multiplies + shifts. All tables pre-converted in prepare_tables.
  2. Reshape-based butterflies — instead of gather/scatter with index arrays,
     reshape to (B, num_blocks, block_size) and slice. Contiguous memory = fast.
  3. Python for-loop over stages — unrolled by XLA at JIT time, each stage
     has a clean reshape with no index indirection.
"""

import math
import jax
import jax.numpy as jnp


# =============================================================================
# Montgomery arithmetic (R = 2^32)
# =============================================================================

def _mont_reduce(T, q, neg_q_inv):
    """
    Montgomery reduction:  T  →  T · R⁻¹ mod q

    T is uint64.  All intermediate math stays in uint64.
    Result is uint32 in [0, q).
    """
    T_low = T & jnp.uint64(0xFFFFFFFF)
    m = (T_low * jnp.uint64(neg_q_inv)) & jnp.uint64(0xFFFFFFFF)
    t = (T + m * jnp.uint64(q)) >> jnp.uint64(32)
    t = t.astype(jnp.uint32)
    return jnp.where(t >= jnp.uint32(q), t - jnp.uint32(q), t)


def _mont_mul(a, b, q, neg_q_inv):
    """
    Montgomery multiply:  (ā, b̄)  →  ā·b̄·R⁻¹ mod q

    If ā = aR mod q and b̄ = bR mod q, then the result is (ab)R mod q
    — i.e. the Montgomery form of a·b.
    """
    T = a.astype(jnp.uint64) * b.astype(jnp.uint64)
    return _mont_reduce(T, q, neg_q_inv)


# =============================================================================
# Public modular helpers (kept for the API contract)
# =============================================================================

def mod_add(a, b, q):
    """Return (a + b) mod q, elementwise."""
    s = a + b
    return jnp.where(s >= q, s - q, s)


def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise."""
    return jnp.where(a >= b, a - b, a + (jnp.uint32(q) - b))


def mod_mul(a, b, q):
    """Return (a * b) mod q, elementwise."""
    return (
        (a.astype(jnp.uint64) * b.astype(jnp.uint64)) % jnp.uint64(q)
    ).astype(jnp.uint32)


# =============================================================================
# Core NTT
# =============================================================================

def ntt(x, *, q, psi_powers, twiddles):
    """
    Forward negacyclic NTT using Montgomery arithmetic and reshape butterflies.

    After prepare_tables:
      psi_powers  — ψ^n table in Montgomery form  (uint32, shape (N,))
      twiddles    — (stage_twiddles, rev, neg_q_inv, R2_mod_q)
        stage_twiddles : list of per-stage twiddle arrays in Montgomery form
        rev            : bit-reversal permutation (int32, shape (N,))
        neg_q_inv      : −q⁻¹ mod 2³² (uint32 scalar)
        R2_mod_q       : R² mod q     (uint32 scalar)
    """
    B, N = x.shape
    stage_twiddles, rev, neg_q_inv, R2_mod_q = twiddles
    log_n = len(stage_twiddles)

    x = x.astype(jnp.uint32)

    # 1. Convert input to Montgomery form:  x̄ = x·R mod q
    #    Done via mont_mul(x, R²_mod_q):  x · R² · R⁻¹ = x·R  ✓
    x = _mont_mul(x, R2_mod_q, q, neg_q_inv)

    # 2. Twist:  x̄[n] *= ψ̄[n]   (negacyclic → cyclic)
    x = _mont_mul(x, psi_powers[jnp.newaxis, :], q, neg_q_inv)

    # 3. Bit-reversal permutation
    x = x[:, rev]

    # 4. Cooley-Tukey DIT butterfly stages (reshape, no index arrays)
    for s in range(log_n):
        half = 1 << s
        block_size = half << 1
        num_blocks = N // block_size

        x = x.reshape(B, num_blocks, block_size)

        even = x[:, :, :half]          # first half of each block
        odd  = x[:, :, half:]          # second half of each block

        # twiddle multiply  (tw is shape (half,), broadcasts over B & blocks)
        tw = stage_twiddles[s]
        t = _mont_mul(odd, tw[jnp.newaxis, jnp.newaxis, :], q, neg_q_inv)

        x = jnp.concatenate([mod_add(even, t, q),
                             mod_sub(even, t, q)], axis=-1)

    x = x.reshape(B, N)

    # 5. Convert back from Montgomery form:  x · R⁻¹ mod q
    x = _mont_reduce(x.astype(jnp.uint64), q, neg_q_inv)

    return x


# =============================================================================
# One-time table preparation (not timed)
# =============================================================================

def prepare_tables(*, q, psi_powers, twiddles):
    """
    Precompute Montgomery constants, convert all tables to Montgomery form,
    build bit-reversal permutation and per-stage twiddle arrays.
    """
    N = psi_powers.shape[0]
    log_n = int(round(math.log2(N)))
    assert 1 << log_n == N, "N must be a power of two"

    psi_powers = psi_powers.astype(jnp.uint32)

    # ---- Montgomery constants ------------------------------------------------
    R = 1 << 32
    q_int = int(q)
    q_inv_R = pow(q_int, -1, R)                    # q⁻¹ mod R
    neg_q_inv = jnp.uint32((R - q_inv_R) % R)      # −q⁻¹ mod R
    R2_mod_q  = jnp.uint32((R * R) % q_int)        # R² mod q

    R_mod_q = jnp.uint64(R % q_int)   # R mod q  (for converting to Montgomery form)

    # helper: convert normal → Montgomery form:  ā = a·R mod q
    def to_mont(a):
        return ((a.astype(jnp.uint64) * R_mod_q)
                % jnp.uint64(q)).astype(jnp.uint32)

    # ---- bit-reversal permutation --------------------------------------------
    idx = jnp.arange(N, dtype=jnp.int32)
    rev = jnp.zeros(N, dtype=jnp.int32)
    for i in range(log_n):
        rev = rev | (((idx >> i) & 1) << (log_n - 1 - i))

    # ---- omega table: ω = ψ², primitive N-th root of unity -------------------
    k = jnp.arange(N, dtype=jnp.int32)
    idx_2k = 2 * k
    omega = jnp.where(
        idx_2k < N,
        psi_powers[idx_2k % N],
        jnp.uint32(q) - psi_powers[(idx_2k - N) % N],
    )

    # convert to Montgomery form
    psi_mont   = to_mont(psi_powers)
    omega_mont = to_mont(omega)

    # ---- per-stage twiddles in Montgomery form --------------------------------
    stage_twiddles = []
    for s in range(log_n):
        half   = 1 << s
        stride = N >> (s + 1)
        j  = jnp.arange(half, dtype=jnp.int32)
        tw = omega_mont[(j * stride) % N]
        stage_twiddles.append(tw)

    return psi_mont, (stage_twiddles, rev, neg_q_inv, R2_mod_q)
