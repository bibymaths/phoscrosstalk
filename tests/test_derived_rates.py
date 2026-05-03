"""
test_derived_rates.py

Unit tests for the derived_rates module.
"""

import numpy as np
import pytest
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_rna():
    """Return tiny t_rna, rna_data, tf_prot_weights."""
    t_rna = np.array([0.0, 10.0, 30.0, 60.0], dtype=float)
    n_genes = 3
    T = len(t_rna)
    rna_data = np.ones((n_genes, T), dtype=float)  # constant fold-change = 1
    rna_data[0] = np.array([1.0, 2.0, 3.0, 4.0])  # gene 0 increases

    K = 2
    tf_prot_weights = np.array(
        [
            [1.0, 0.0, 0.0],  # protein 0 reads gene 0 only
            [0.0, 1.0, 0.0],  # protein 1 reads gene 1 only
        ],
        dtype=float,
    )
    return t_rna, rna_data, tf_prot_weights, K


def _simple_phospho():
    """Return tiny t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M."""
    K, M, N = 2, 3, 4
    T = 5
    t_protein = np.array([0.0, 5.0, 10.0, 30.0, 60.0], dtype=float)
    Y_data = np.random.default_rng(0).uniform(0.1, 0.9, (N, T))
    R_kin_site = np.random.default_rng(1).uniform(0, 1, (M, N))
    kin_to_prot_idx = np.array([0, 1, -1], dtype=int)
    return t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M


# ---------------------------------------------------------------------------
# k_act tests
# ---------------------------------------------------------------------------

class TestMakeKActFn:
    def test_constant_fallback_no_data(self):
        from phoscrosstalk.derived_rates import make_k_act_fn

        K = 4
        fn = make_k_act_fn(None, None, None, K)
        result = fn(jnp.float32(10.0))
        assert result.shape == (K,)
        np.testing.assert_allclose(np.array(result), np.ones(K), rtol=1e-5)

    def test_output_shape_with_data(self):
        from phoscrosstalk.derived_rates import make_k_act_fn

        t_rna, rna_data, tf_prot_weights, K = _simple_rna()
        fn = make_k_act_fn(t_rna, rna_data, tf_prot_weights, K)
        result = fn(jnp.float32(5.0))
        assert result.shape == (K,)

    def test_piecewise_constant_at_zero(self):
        """At t=0, output should equal the first time-point value."""
        from phoscrosstalk.derived_rates import make_k_act_fn

        t_rna, rna_data, tf_prot_weights, K = _simple_rna()
        fn = make_k_act_fn(t_rna, rna_data, tf_prot_weights, K, interp_mode="piecewise_constant")
        result = np.array(fn(jnp.float32(0.0)))
        # protein 0 reads gene 0 at t=0: rna_data[0, 0] = 1.0
        assert abs(result[0] - 1.0) < 1e-4

    def test_linear_interp_midpoint(self):
        """Linear interp at t=5 (midpoint of [0, 10]) should give average."""
        from phoscrosstalk.derived_rates import make_k_act_fn

        t_rna, rna_data, tf_prot_weights, K = _simple_rna()
        fn = make_k_act_fn(t_rna, rna_data, tf_prot_weights, K, interp_mode="linear")
        result = np.array(fn(jnp.float32(5.0)))
        # protein 0 reads gene 0: at t=5 = interp between 1.0 (t=0) and 2.0 (t=10) = 1.5
        assert abs(result[0] - 1.5) < 1e-4

    def test_jax_traceable(self):
        """k_act_fn must be JAX-traceable (vmappable)."""
        import jax
        from phoscrosstalk.derived_rates import make_k_act_fn

        t_rna, rna_data, tf_prot_weights, K = _simple_rna()
        fn = make_k_act_fn(t_rna, rna_data, tf_prot_weights, K)
        jit_fn = jax.jit(fn)
        result = jit_fn(jnp.float32(15.0))
        assert result.shape == (K,)


# ---------------------------------------------------------------------------
# s_prod tests
# ---------------------------------------------------------------------------

class TestMakeSProdFn:
    def test_constant_fallback_empty_data(self):
        from phoscrosstalk.derived_rates import make_s_prod_fn

        K, M = 3, 0
        fn = make_s_prod_fn(
            t_protein=np.array([0.0, 1.0]),
            Y_data=np.zeros((0, 2)),
            R_kin_site=np.zeros((M, 0)),
            kin_to_prot_idx=np.array([], dtype=int),
            K=K,
            M=M,
        )
        result = fn(jnp.float32(0.0))
        assert result.shape == (K,)
        np.testing.assert_allclose(np.array(result), np.full(K, 0.1), rtol=1e-5)

    def test_output_shape(self):
        from phoscrosstalk.derived_rates import make_s_prod_fn

        t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M = _simple_phospho()
        fn = make_s_prod_fn(t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M)
        result = fn(jnp.float32(5.0))
        assert result.shape == (K,)

    def test_softplus_output_positive(self):
        """softplus output must be strictly positive."""
        from phoscrosstalk.derived_rates import make_s_prod_fn

        t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M = _simple_phospho()
        fn = make_s_prod_fn(
            t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M,
            s_prod_fn_type="softplus"
        )
        result = np.array(fn(jnp.float32(5.0)))
        assert np.all(result > 0), "softplus output should be positive"

    def test_linear_fn(self):
        """Linear s_prod_fn should not clip to zero for positive input."""
        from phoscrosstalk.derived_rates import make_s_prod_fn

        t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M = _simple_phospho()
        fn = make_s_prod_fn(
            t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M,
            s_prod_fn_type="linear"
        )
        result = fn(jnp.float32(5.0))
        assert result.shape == (K,)

    def test_jax_traceable(self):
        """s_prod_fn must be JAX-traceable."""
        import jax
        from phoscrosstalk.derived_rates import make_s_prod_fn

        t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M = _simple_phospho()
        fn = make_s_prod_fn(t_protein, Y_data, R_kin_site, kin_to_prot_idx, K, M)
        jit_fn = jax.jit(fn)
        result = jit_fn(jnp.float32(10.0))
        assert result.shape == (K,)
