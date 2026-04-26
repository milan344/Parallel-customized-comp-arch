"""
Assignment 2 student implementation.

32-bit kernels: compulsory (uses uint64, replaces % with conditional ops).
64-bit kernels: optional  (uses uint64 + overflow-safe helpers).
128-bit kernels: not implemented.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_U = jnp.uint64


# =============================================================================
# Fast conditional-branch modular helpers (no integer division)
#
# For 32-bit: a, b < q < 2^32, stored in uint64.
#   add: a + b < 2q, so result = s or s - q       (one compare + subtract)
#   sub: a - b in (-q, q), fix with conditional +q (one compare + add)
#   mul: a * b can be up to ~2^64, % is unavoidable
# =============================================================================

def _add_u64(a, b, q):
    """(a + b) mod q via conditional subtraction. Avoids integer division."""
    s = a + b
    return s - jnp.where(s >= q, q, _U(0))


def _sub_u64(a, b, q):
    """(a - b) mod q via conditional addition. Avoids integer division."""
    diff = a - b                   # wraps in uint64 if a < b
    return diff + jnp.where(a < b, q, _U(0))


# =============================================================================
# 64-bit overflow-safe helpers (same as before)
# =============================================================================

def _add_mod_u64(a, b, q):
    """(a + b) mod q, overflow-safe for uint64."""
    s = a + b
    overflow = s < a
    neg_q = _U(0) - q
    return jnp.where(overflow, s + neg_q, jnp.where(s >= q, s - q, s))


def _sub_mod_u64(a, b, q):
    """(a - b) mod q, underflow-safe for uint64."""
    diff = a - b
    return diff + jnp.where(a < b, q, _U(0))


def _mul_mod_u64(a, b, q):
    """(a * b) mod q using binary method for full uint64 range."""
    a = a % q
    shape = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    a = jnp.broadcast_to(a, shape)
    b = jnp.broadcast_to(b, shape)
    result = jnp.zeros(shape, dtype=jnp.uint64)
    for i in range(64):
        bit = (b >> _U(i)) & _U(1)
        result = jnp.where(bit, _add_mod_u64(result, a, q), result)
        a = _add_mod_u64(a, a, q)
    return result


def _sum_mod_u64(arr, q):
    """Sum array elements mod q using pairwise tree reduction."""
    while arr.shape[0] > 1:
        n = arr.shape[0]
        if n % 2 == 1:
            arr = jnp.concatenate([arr, jnp.zeros(1, dtype=jnp.uint64)])
            n += 1
        arr = _add_mod_u64(arr[:n // 2], arr[n // 2:], q)
    return arr[0]


# =============================================================================
# 32-bit primitives (compulsory)
# =============================================================================

def mod_add_32(a, b, q):
    return _add_u64(_U(a), _U(b), _U(q))


def mod_sub_32(a, b, q):
    return _sub_u64(_U(a), _U(b), _U(q))


def mod_mul_32(a, b, q):
    return (_U(a) * _U(b)) % _U(q)


# =============================================================================
# 64-bit primitives (optional)
# =============================================================================

def mod_add_64(a, b, q):
    return _add_mod_u64(_U(a), _U(b), _U(q))

def mod_sub_64(a, b, q):
    return _sub_mod_u64(_U(a), _U(b), _U(q))

def mod_mul_64(a, b, q):
    return _mul_mod_u64(_U(a), _U(b), _U(q))


# =============================================================================
# 128-bit primitives (not implemented)
# =============================================================================

def mod_add_128(a, b, q):
    raise NotImplementedError

def mod_sub_128(a, b, q):
    raise NotImplementedError

def mod_mul_128(a, b, q):
    raise NotImplementedError


# =============================================================================
# Frozen dispatch API
# =============================================================================

def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:  return mod_add_32(a, b, q)
    if int(bit_width) == 64:  return mod_add_64(a, b, q)
    if int(bit_width) == 128: return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")

def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:  return mod_sub_32(a, b, q)
    if int(bit_width) == 64:  return mod_sub_64(a, b, q)
    if int(bit_width) == 128: return mod_sub_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")

def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:  return mod_mul_32(a, b, q)
    if int(bit_width) == 64:  return mod_mul_64(a, b, q)
    if int(bit_width) == 128: return mod_mul_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# =============================================================================
# MLE update
# =============================================================================

def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    q_u = _U(q)
    z, o, t = _U(zero_eval), _U(one_eval), _U(target_eval)
    diff = _sub_u64(o, z, q_u)
    return _add_u64(z, (t * diff) % q_u, q_u)

def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    q_u = _U(q)
    z, o, t = _U(zero_eval), _U(one_eval), _U(target_eval)
    diff = _sub_mod_u64(o, z, q_u)
    return _add_mod_u64(z, _mul_mod_u64(t, diff, q_u), q_u)

def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    raise NotImplementedError

def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:  return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:  return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128: return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# =============================================================================
# Sum-check prover — 32-bit (compulsory, optimized)
#
# Optimizations over naive version:
#   1. Pre-compute lo/hi/diff once per round (not per evaluation point)
#   2. Batch all evaluation points t=0..degree into a 2D array
#   3. Interpolation via repeated addition (no mul for small t)
#   4. Conditional add/sub instead of % (avoids integer division)
#   5. Only % on multiplications (unavoidable) and final sum
# =============================================================================

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    degree = max(len(term) for term in expression)
    q_u = _U(q)
    tables = {k: jnp.uint64(v) for k, v in eval_tables.items()}
    all_round_evals = []

    for round_i in range(num_rounds):
        # --- 1. Pre-compute lo, hi, diff for all polynomials (once) ---
        splits = {}
        for name, tbl in tables.items():
            lo = tbl[0::2]
            hi = tbl[1::2]
            diff = _sub_u64(hi, lo, q_u)           # no %
            splits[name] = (lo, hi, diff)

        half = splits[next(iter(splits))][0].shape[0]

        # --- 2. Build interpolation tables for all t values ---
        # interp_all[name] has shape (degree+1, half)
        # t=0 → lo, t=1 → hi, t≥2 → lo + t*diff via repeated add (no mul)
        interp_all = {}
        for name, (lo, hi, diff) in splits.items():
            layers = [lo, hi]
            accum = diff                            # 1 * diff
            for _ in range(2, degree + 1):
                accum = _add_u64(accum, diff, q_u)  # t * diff, no mul
                layers.append(_add_u64(lo, accum, q_u))
            interp_all[name] = jnp.stack(layers)    # (degree+1, half)

        # --- 3. Evaluate expression for all t (vectorized over t and entries) ---
        total = jnp.zeros((degree + 1, half), dtype=jnp.uint64)
        for term in expression:
            prod = interp_all[term[0]]
            for f in term[1:]:
                prod = (prod * interp_all[f]) % q_u  # mul: % unavoidable
            total = _add_u64(total, prod, q_u)        # add: no %

        # --- 4. Sum over hypercube (axis=1) ---
        # max sum: 2^20 * 2^32 = 2^52, fits in uint64
        round_evals = jnp.sum(total, axis=1) % q_u   # one % per t value
        all_round_evals.append(round_evals)

        # --- 5. Bind variable with challenge (reuse pre-computed diff) ---
        if round_i < num_rounds - 1:
            r = _U(challenges[round_i])
            for name, (lo, _, diff) in splits.items():
                tables[name] = _add_u64(lo, (r * diff) % q_u, q_u)

    claim0 = _add_u64(all_round_evals[0][0], all_round_evals[0][1], q_u)
    return claim0, jnp.stack(all_round_evals)


# =============================================================================
# Sum-check prover — 64-bit (optional)
# =============================================================================

def _eval_expr_64(tables, expression, t, q_u):
    """Evaluate expression at point t, sum over remaining hypercube. 64-bit."""
    interp = {}
    for name, tbl in tables.items():
        lo = tbl[0::2]
        hi = tbl[1::2]
        if t == 0:
            interp[name] = lo
        elif t == 1:
            interp[name] = hi
        else:
            diff = _sub_mod_u64(hi, lo, q_u)
            val = lo
            for _ in range(t):
                val = _add_mod_u64(val, diff, q_u)
            interp[name] = val

    half = interp[next(iter(interp))].shape[0]
    total = jnp.zeros(half, dtype=jnp.uint64)
    for term in expression:
        prod = interp[term[0]]
        for f in term[1:]:
            prod = _mul_mod_u64(prod, interp[f], q_u)
        total = _add_mod_u64(total, prod, q_u)

    return _sum_mod_u64(total, q_u)


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    degree = max(len(term) for term in expression)
    q_u = _U(q)
    tables = {k: jnp.uint64(v) for k, v in eval_tables.items()}
    all_round_evals = []

    for round_i in range(num_rounds):
        evals = [_eval_expr_64(tables, expression, t, q_u)
                 for t in range(degree + 1)]
        all_round_evals.append(jnp.array(evals, dtype=jnp.uint64))

        if round_i < num_rounds - 1:
            r = _U(challenges[round_i])
            for name in tables:
                lo = tables[name][0::2]
                hi = tables[name][1::2]
                diff = _sub_mod_u64(hi, lo, q_u)
                tables[name] = _add_mod_u64(lo, _mul_mod_u64(r, diff, q_u), q_u)

    claim0 = _add_mod_u64(all_round_evals[0][0], all_round_evals[0][1], q_u)
    return claim0, jnp.stack(all_round_evals)


# =============================================================================
# Sum-check prover — 128-bit (not implemented)
# =============================================================================

def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    raise NotImplementedError


# =============================================================================
# Frozen dispatcher
# =============================================================================

def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    if int(bit_width) == 32:
        return sumcheck_32(eval_tables, q=q, expression=expression,
                           challenges=challenges, num_rounds=num_rounds)
    if int(bit_width) == 64:
        return sumcheck_64(eval_tables, q=q, expression=expression,
                           challenges=challenges, num_rounds=num_rounds)
    if int(bit_width) == 128:
        return sumcheck_128(eval_tables, q=q, expression=expression,
                            challenges=challenges, num_rounds=num_rounds)
    raise ValueError(f"Unsupported bit_width={bit_width}")
