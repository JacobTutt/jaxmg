import pytest

from jaxmg._distributed_solver import get_solver_process_grid


def test_get_solver_process_grid_prefers_near_square_world_layout():
    grid = get_solver_process_grid(world_size=8, process_index=5, local_device_count=1)
    assert grid.nprow == 2
    assert grid.npcol == 4
    assert grid.process_row == 1
    assert grid.process_col == 1


def test_get_solver_process_grid_accepts_explicit_grid():
    grid = get_solver_process_grid(
        world_size=8,
        process_index=6,
        local_device_count=1,
        nprow=4,
        npcol=2,
    )
    assert grid.nprow == 4
    assert grid.npcol == 2
    assert grid.process_row == 3
    assert grid.process_col == 0


def test_get_solver_process_grid_rejects_invalid_process_index():
    with pytest.raises(ValueError, match="process_index=8"):
        get_solver_process_grid(world_size=8, process_index=8, local_device_count=1)
