import pytest

from jaxmg._block_cyclic_2d import (
    choose_process_grid,
    local_leading_dimension,
    local_matrix_shape,
    numroc,
    owning_process_col,
    owning_process_row,
)


def test_choose_process_grid_prefers_near_square_factorization():
    assert choose_process_grid(1) == (1, 1)
    assert choose_process_grid(4) == (2, 2)
    assert choose_process_grid(8) == (2, 4)
    assert choose_process_grid(12) == (3, 4)


def test_choose_process_grid_respects_explicit_dimension():
    assert choose_process_grid(8, nprow=2) == (2, 4)
    assert choose_process_grid(8, npcol=4) == (2, 4)
    assert choose_process_grid(8, nprow=2, npcol=4) == (2, 4)

    with pytest.raises(ValueError, match="not divisible"):
        choose_process_grid(8, nprow=3)
    with pytest.raises(ValueError, match="Invalid process grid"):
        choose_process_grid(8, nprow=3, npcol=3)


def test_numroc_matches_expected_1d_block_cyclic_counts():
    counts = [numroc(10, 2, proc, 0, 3) for proc in range(3)]
    assert counts == [4, 4, 2]
    assert sum(counts) == 10

    counts = [numroc(17, 4, proc, 0, 4) for proc in range(4)]
    assert counts == [5, 4, 4, 4]
    assert sum(counts) == 17


def test_local_matrix_shape_is_cartesian_product_of_row_and_col_counts():
    assert local_matrix_shape(10, 12, 2, 3, 0, 0, 2, 2) == (6, 6)
    assert local_matrix_shape(10, 12, 2, 3, 1, 1, 2, 2) == (4, 6)


def test_owning_process_helpers_follow_block_cyclic_schedule():
    assert owning_process_row(0, 2, 3) == 0
    assert owning_process_row(1, 2, 3) == 0
    assert owning_process_row(2, 2, 3) == 1
    assert owning_process_row(5, 2, 3) == 2
    assert owning_process_row(6, 2, 3) == 0

    assert owning_process_col(0, 3, 2) == 0
    assert owning_process_col(2, 3, 2) == 0
    assert owning_process_col(3, 3, 2) == 1
    assert owning_process_col(7, 3, 2) == 0


def test_local_leading_dimension_uses_blas_safe_minimum():
    assert local_leading_dimension(0) == 1
    assert local_leading_dimension(7) == 7

    with pytest.raises(ValueError, match="non-negative"):
        local_leading_dimension(-1)
