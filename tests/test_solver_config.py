import diffrax
import optimistix as optx
import pytest

from phoscrosstalk import solver_config


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("tsit5", diffrax.Tsit5),
        ("dopri5", diffrax.Dopri5),
        ("dopri8", diffrax.Dopri8),
        ("bosh3", diffrax.Bosh3),
        ("kvaerno3", diffrax.Kvaerno3),
        ("kvaerno4", diffrax.Kvaerno4),
        ("kvaerno5", diffrax.Kvaerno5),
    ],
)
def test_make_diffrax_solver_supported_variants(kind, expected_type):
    solver = solver_config.make_diffrax_solver(kind, root_find_max_steps=7)
    assert isinstance(solver, expected_type)


def test_make_diffrax_solver_tsit5_scan_kind_and_invalid():
    solver = solver_config.make_diffrax_solver("tsit5", scan_kind="bounded")
    assert isinstance(solver, diffrax.Tsit5)
    with pytest.raises(ValueError, match="Unknown Diffrax ODE solver"):
        solver_config.make_diffrax_solver("bad_solver")


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("forward", diffrax.ForwardMode),
        ("recursive", diffrax.RecursiveCheckpointAdjoint),
        ("checkpoint", diffrax.RecursiveCheckpointAdjoint),
        ("direct", diffrax.DirectAdjoint),
        ("backsolve", diffrax.BacksolveAdjoint),
    ],
)
def test_make_diffrax_adjoint_supported_variants(kind, expected_type):
    adjoint = solver_config.make_diffrax_adjoint(kind)
    assert isinstance(adjoint, expected_type)


def test_make_diffrax_adjoint_none_and_invalid():
    assert solver_config.make_diffrax_adjoint(None) is None
    assert solver_config.make_diffrax_adjoint("none") is None
    with pytest.raises(ValueError, match="Unknown Diffrax adjoint"):
        solver_config.make_diffrax_adjoint("bad")


def test_make_stepsize_controller_and_ls_solver_variants():
    controller = solver_config.make_stepsize_controller(1e-6, 1e-8)
    assert isinstance(controller, diffrax.PIDController)

    assert isinstance(solver_config.make_ls_solver("lm", 1e-6, 1e-8), optx.LevenbergMarquardt)
    assert isinstance(
        solver_config.make_ls_solver("indirect_lm", 1e-6, 1e-8), optx.IndirectLevenbergMarquardt
    )
    assert isinstance(solver_config.make_ls_solver("dogleg", 1e-6, 1e-8), optx.Dogleg)
    assert isinstance(solver_config.make_ls_solver("gauss_newton", 1e-6, 1e-8), optx.GaussNewton)
    with pytest.raises(ValueError, match="Unknown least-squares solver"):
        solver_config.make_ls_solver("bad", 1e-6, 1e-8)


def test_make_optx_adjoint_variants_and_invalid():
    assert isinstance(solver_config.make_optx_adjoint("implicit"), optx.ImplicitAdjoint)
    assert isinstance(
        solver_config.make_optx_adjoint("checkpoint"), optx.RecursiveCheckpointAdjoint
    )
    with pytest.raises(ValueError, match="Unknown Optimistix adjoint"):
        solver_config.make_optx_adjoint("bad")
