"""
Negacyclic Number Theoretic Transform (NTT) implementation.

Strategy:
  1. Twist inputs by psi_powers to convert negacyclic → cyclic NTT
  2. Bit-reverse the twisted input
  3. Iterative Cooley-Tukey DIT butterflies using precomputed per-stage
     twiddle factors and index arrays (all via jax.lax.fori_loop)

All heavy precomputation (omega table, per-stage twiddles, bit-reversal
permutation, butterfly indices) happens once in prepare_tables and is
excluded from timing.
"""

import math
import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------

def mod_add(a, b, q):
    """Return (a + b) mod q, elementwise.  Assumes a, b in [0, q), q < 2^31."""
    s = a + b
    return jnp.where(s >= q, s - q, s)


def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise.  Assumes a, b in [0, q), q < 2^31."""
    return jnp.where(a >= b, a - b, a + (jnp.uint32(q) - b))


def mod_mul(a, b, q):
    """Return (a * b) mod q, elementwise.  Promotes to uint64 to avoid overflow."""
    return (
        (a.astype(jnp.uint64) * b.astype(jnp.uint64)) % jnp.uint64(q)
    ).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------

def ntt(x, *, q, psi_powers, twiddles):
    """
    Compute the forward negacyclic NTT.

    After prepare_tables, *twiddles* is a tuple:
        (stage_tw, even_idx, odd_idx, rev)
    and *psi_powers* is the uint32 ψ^n table used for the twist step.
    """
    B, N = x.shape
    stage_tw, even_idx, odd_idx, rev = twiddles
    log_n = stage_tw.shape[0]

    x = x.astype(jnp.uint32)

    # --- Step 1: twist  x[n] *= ψ^n  (converts negacyclic → cyclic) ----------
    x = mod_mul(x, psi_powers[jnp.newaxis, :], q)

    # --- Step 2: bit-reversal permutation -------------------------------------
    x = x[:, rev]

    # --- Step 3: Cooley-Tukey DIT butterfly stages ----------------------------
    def _stage(s, x):
        tw = stage_tw[s]          # (N//2,)
        ei = even_idx[s]          # (N//2,)
        oi = odd_idx[s]           # (N//2,)

        a = x[:, ei]              # (B, N//2)
        b = x[:, oi]              # (B, N//2)

        t = mod_mul(b, tw[jnp.newaxis, :], q)

        x = x.at[:, ei].set(mod_add(a, t, q))
        x = x.at[:, oi].set(mod_sub(a, t, q))
        return x

    x = jax.lax.fori_loop(0, log_n, _stage, x)
    return x


# -----------------------------------------------------------------------------
# One-time table preparation (not timed)
# -----------------------------------------------------------------------------

def prepare_tables(*, q, psi_powers, twiddles):
    """
    Precompute everything the butterfly loop needs:
      • omega table  (ω = ψ², primitive N-th root of unity)
      • per-stage twiddle arrays
      • per-stage butterfly even/odd index arrays
      • bit-reversal permutation
    """
    N = psi_powers.shape[0]
    log_n = int(round(math.log2(N)))
    assert 1 << log_n == N, "N must be a power of two"

    psi_powers = psi_powers.astype(jnp.uint32)

    # ---- bit-reversal permutation -------------------------------------------
    idx = jnp.arange(N, dtype=jnp.int32)
    rev = jnp.zeros(N, dtype=jnp.int32)
    for i in range(log_n):
        rev = rev | (((idx >> i) & 1) << (log_n - 1 - i))

    # ---- omega table: ω^k = ψ^{2k} for k = 0 … N-1 -------------------------
    #   ψ^N ≡ −1 (mod q), so for 2k ≥ N:  ψ^{2k} = −ψ^{2k−N}
    k = jnp.arange(N, dtype=jnp.int32)
    idx_2k = 2 * k
    omega = jnp.where(
        idx_2k < N,
        psi_powers[idx_2k % N],
        jnp.uint32(q) - psi_powers[(idx_2k - N) % N],
    )

    # ---- per-stage twiddles & butterfly indices ------------------------------
    half_n = N // 2
    flat = jnp.arange(half_n, dtype=jnp.int32)

    stage_tw  = jnp.zeros((log_n, half_n), dtype=jnp.uint32)
    even_idx  = jnp.zeros((log_n, half_n), dtype=jnp.int32)
    odd_idx   = jnp.zeros((log_n, half_n), dtype=jnp.int32)

    for s in range(log_n):
        half       = 1 << s
        block_size = half << 1
        stride     = N >> (s + 1)           # N / block_size

        group  = flat // half
        offset = flat %  half

        ei = group * block_size + offset
        oi = ei + half

        tw = omega[(offset * stride) % N]

        stage_tw = stage_tw.at[s].set(tw)
        even_idx = even_idx.at[s].set(ei)
        odd_idx  = odd_idx.at[s].set(oi)

    return psi_powers, (stage_tw, even_idx, odd_idx, rev)