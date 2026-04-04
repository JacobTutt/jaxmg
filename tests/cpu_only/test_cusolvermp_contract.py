import jax.numpy as jnp

from jaxmg import plan_potrs_cusolvermp_contract


def test_plan_potrs_cusolvermp_contract_accepts_vector_rhs():
    b = jnp.ones((6,))

    contract = plan_potrs_cusolvermp_contract(b)

    assert contract.supported is True
    assert contract.input_nrhs == 1
    assert contract.normalized_rhs_shape == (6, 1)
    assert contract.failure_reason is None


def test_plan_potrs_cusolvermp_contract_accepts_single_column_rhs():
    b = jnp.ones((6, 1))

    contract = plan_potrs_cusolvermp_contract(b)

    assert contract.supported is True
    assert contract.input_nrhs == 1
    assert contract.normalized_rhs_shape == (6, 1)
    assert contract.failure_reason is None


def test_plan_potrs_cusolvermp_contract_rejects_multi_rhs():
    b = jnp.ones((6, 3))

    contract = plan_potrs_cusolvermp_contract(b)

    assert contract.supported is False
    assert contract.input_nrhs == 3
    assert "NRHS=1" in contract.failure_reason


def test_plan_potrs_cusolvermp_contract_rejects_rank_three_rhs():
    b = jnp.ones((2, 3, 4))

    contract = plan_potrs_cusolvermp_contract(b)

    assert contract.supported is False
    assert contract.normalized_rhs_shape is None
    assert contract.input_nrhs is None
    assert "1D or 2D" in contract.failure_reason
