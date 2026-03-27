"""
Assignment 2 student implementation reference skeleton.

This file documents the frozen student-facing API.
Only 32-bit kernels are compulsory in the base track.
64-bit and 128-bit kernels are intentionally left unimplemented here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_U = jnp.uint64


# -----------------------------------------------------------------------------
# 32-bit primitives (compulsory)
# -----------------------------------------------------------------------------

def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    return (_U(a) + _U(b)) % _U(q)


def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    return (_U(a) + _U(q) - _U(b)) % _U(q)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    return (_U(a) * _U(b)) % _U(q)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional)
# -----------------------------------------------------------------------------

def mod_add_64(a, b, q):
    raise NotImplementedError

def mod_sub_64(a, b, q):
    raise NotImplementedError

def mod_mul_64(a, b, q):
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 128-bit primitives (optional)
# -----------------------------------------------------------------------------

def mod_add_128(a, b, q):
    raise NotImplementedError

def mod_sub_128(a, b, q):
    raise NotImplementedError

def mod_mul_128(a, b, q):
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Frozen dispatch API
# -----------------------------------------------------------------------------

def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_add_32(a, b, q)
    if int(bit_width) == 64:
        return mod_add_64(a, b, q)
    if int(bit_width) == 128:
        return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_sub_32(a, b, q)
    if int(bit_width) == 64:
        return mod_sub_64(a, b, q)
    if int(bit_width) == 128:
        return mod_sub_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_mul_32(a, b, q)
    if int(bit_width) == 64:
        return mod_mul_64(a, b, q)
    if int(bit_width) == 128:
        return mod_mul_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# -----------------------------------------------------------------------------
# MLE update
# -----------------------------------------------------------------------------

def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """
    Compulsory 32-bit MLE update.

    Linear interpolation: result = zero + target * (one - zero) mod q
    """
    q_u = _U(q)
    z = _U(zero_eval)
    o = _U(one_eval)
    t = _U(target_eval)
    diff = (o + q_u - z) % q_u
    prod = (t * diff) % q_u
    return (z + prod) % q_u


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    raise NotImplementedError

def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    raise NotImplementedError


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# -----------------------------------------------------------------------------
# Sum-check prover (32-bit)
#
# Table layout (LSB convention):
#   table[i] = f(b_0, b_1, ..., b_{n-1})
#   where i = b_0 + 2*b_1 + ... + 2^{n-1}*b_{n-1}
#
# In each round, the current variable is the LSB of the table index.
#   lo = table[0::2]  (even indices, current var = 0)
#   hi = table[1::2]  (odd indices, current var = 1)
# After binding, the table halves and the next variable becomes the new LSB.
# -----------------------------------------------------------------------------

def _eval_expression_at_point(tables, expression, t, q_u):
    """
    Evaluate the composed polynomial at evaluation point t for the current
    variable, summing over all remaining variables.
    """
    t_u = _U(t)

    # Interpolate each base polynomial's table at point t
    interp = {}
    for name, tbl in tables.items():
        lo = tbl[0::2]   # even indices: current var = 0
        hi = tbl[1::2]   # odd indices:  current var = 1
        diff = (hi + q_u - lo) % q_u
        interp[name] = (lo + t_u * diff) % q_u

    # Evaluate expression: sum-of-products
    half = interp[next(iter(interp))].shape[0]
    total = jnp.zeros(half, dtype=jnp.uint64)
    for term in expression:
        prod = interp[term[0]]
        for factor_name in term[1:]:
            prod = (prod * interp[factor_name]) % q_u
        total = (total + prod) % q_u

    # Sum all entries to get g_i(t)
    return jnp.sum(total) % q_u


def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    Compulsory 32-bit sumcheck prover.

    Returns
    -------
    claim0 : jnp scalar
        The claimed sum S = g_1(0) + g_1(1).
    round_evals : jnp.ndarray, shape (num_rounds, degree+1)
        round_evals[i] = [g_{i+1}(0), g_{i+1}(1), ..., g_{i+1}(degree)].
    """
    degree = max(len(term) for term in expression)
    q_u = _U(q)

    # Work entirely in uint64
    tables = {k: jnp.uint64(v) for k, v in eval_tables.items()}

    all_round_evals = []

    for round_i in range(num_rounds):
        # Evaluate g_i(t) for t = 0, 1, ..., degree
        round_vals = []
        for t in range(degree + 1):
            val = _eval_expression_at_point(tables, expression, t, q_u)
            round_vals.append(val)

        all_round_evals.append(jnp.array(round_vals, dtype=jnp.uint64))

        # Bind the current variable to the challenge and update tables
        if round_i < num_rounds - 1:
            r = _U(challenges[round_i])
            for name in tables:
                tbl = tables[name]
                lo = tbl[0::2]
                hi = tbl[1::2]
                diff = (hi + q_u - lo) % q_u
                tables[name] = (lo + r * diff) % q_u

    # claim0 = g_1(0) + g_1(1)
    claim0 = (all_round_evals[0][0] + all_round_evals[0][1]) % q_u

    # Stack all round evaluations into (num_rounds, degree+1)
    round_evals = jnp.stack(all_round_evals)

    return claim0, round_evals


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    raise NotImplementedError

def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    raise NotImplementedError


def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
    if int(bit_width) == 32:
        return sumcheck_32(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 64:
        return sumcheck_64(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 128:
        return sumcheck_128(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    raise ValueError(f"Unsupported bit_width={bit_width}")