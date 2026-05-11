import numpy as np
import pytest
from types import SimpleNamespace
from collections import namedtuple

from phoscrosstalk import posterior


def test_cfg_get_supports_none_and_namespace():
    assert posterior._cfg_get(None, "x", 5) == 5
    cfg = SimpleNamespace(alpha=3)
    assert posterior._cfg_get(cfg, "alpha", 0) == 3
    assert posterior._cfg_get(cfg, "missing", 7) == 7


def test_clip_theta_open_moves_to_open_interval():
    theta = np.array([0.0, 0.5, 1.0])
    xl = np.zeros(3)
    xu = np.ones(3)
    clipped = posterior._clip_theta_open(theta, xl, xu)
    assert np.all(clipped > xl)
    assert np.all(clipped < xu)


def test_resample_residual_matrix_modes():
    rng = np.random.default_rng(0)
    resid = np.array([[1.0, 2.0], [3.0, 4.0]])
    out_case = posterior._resample_residual_matrix(rng, resid, mode="case")
    out_time = posterior._resample_residual_matrix(rng, resid, mode="time")
    out_el = posterior._resample_residual_matrix(rng, resid, mode="element")
    assert out_case.shape == resid.shape
    assert out_time.shape == resid.shape
    assert out_el.shape == resid.shape


def test_summarize_samples_has_expected_columns():
    samples = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
    df = posterior._summarize_samples(samples, ["a", "b"])
    assert list(df["parameter"]) == ["a", "b"]
    assert {"mean", "std", "q2.5", "q97.5", "n_success"}.issubset(df.columns)


def _problem():
    return SimpleNamespace(
        cpu_threads="auto",
        parallel_starts="auto",
        threads_per_start="auto",
        use_physical_cores=True,
        reserve_cores=0,
        dims=SimpleNamespace(K=2, M=1, N=2),
        t=np.array([0.0, 1.0, 2.0]),
        P_data=np.array([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]]),
        A_scaled=np.array([[1.0, 1.0, 1.0]]),
        prot_idx_for_A=np.array([0]),
        W_data=np.ones((2, 3)),
        W_data_prot=np.ones((1, 3)),
        Cg=np.eye(2),
        Cl=np.eye(2),
        site_prot_idx=np.array([0, 1]),
        K_site_kin=np.ones((1, 2)),
        R=np.ones((1, 2)),
        L_alpha=np.zeros((1, 1)),
        kin_to_prot_idx=np.array([0]),
        receptor_mask_prot=np.array([False, False]),
        receptor_mask_kin=np.array([False]),
        mechanism="dist",
        lambda_net=0.1,
        reg_lambda=0.2,
        xl=np.zeros(3),
        xu=np.ones(3),
        loss_type="mse",
        pseudo_huber_delta=0.1,
        slope_lambda=0.1,
        k_act_fn=lambda t: np.ones(2),
        s_prod_fn=lambda t: np.ones(2),
    )


def test_pickle_and_runtime_helpers(monkeypatch):
    assert posterior._can_pickle({"a": 1}) is True
    bad = {"ok": 1, "bad": lambda x: x}
    assert posterior._find_non_picklable_items(bad) == ["bad"]

    cfg = SimpleNamespace(parallel_refits=3, threads_per_refit=2, reserve_cores=1)
    args = posterior._make_runtime_args_from_problem(_problem(), cfg, n_tasks=5)
    assert args.n_starts == 5
    assert args.reserve_cores == 0

    fake_plan = SimpleNamespace(
        topo=SimpleNamespace(slurm_cpus=None, logical_cpus=8, physical_cores=4, affinity_cpus=8, source="test"),
        total_available_cpus=8,
        n_parallel_runs=2,
        threads_per_run=4,
        xla_threads=4,
        blas_threads=4,
    )
    monkeypatch.setattr("phoscrosstalk.runtime_env.plan_cpu_runtime", lambda **kwargs: fake_plan)
    plan = posterior._plan_uncertainty_runtime(_problem(), cfg, n_tasks=5, label="bootstrap")
    assert plan.n_parallel_runs == 2


def test_base_refit_kwargs_and_simulate_helpers(monkeypatch, tmp_path):
    problem = _problem()
    kwargs = posterior._base_refit_kwargs_from_problem(
        problem=problem, w_phospho=1.0, w_abundance=2.0, w_reg=3.0, w_mrna=4.0
    )
    assert kwargs["w_abundance"] == 2.0
    assert "k_act_fn" in kwargs
    assert kwargs["dims"].K == 2

    class SimProblem:
        def simulate_full(self, theta):
            return {"P_sim": np.ones((2, 3)), "A_sim": np.zeros((2, 3))}

    out = posterior._simulate_full_for_uncertainty(SimProblem(), np.array([0.1]))
    assert out["P_sim"].shape == (2, 3)

    class LegacyProblem:
        def simulate(self, theta):
            return np.ones((2, 3))

    out2 = posterior._simulate_full_for_uncertainty(LegacyProblem(), np.array([0.1]))
    assert out2["P_sim"].shape == (2, 3)
    assert out2["A_sim"] is None

    class SimProblem2:
        t = np.array([0.0, 1.0, 2.0])
        t_rna = np.array([0.0, 2.0])

        def simulate_full(self, theta):
            val = float(theta[0])
            return {
                "P_sim": np.full((2, 3), val),
                "A_sim": np.full((1, 3), val),
                "R_sim": np.full((1, 2), val),
            }

    written = posterior._save_prediction_intervals(
        outdir=str(tmp_path),
        problem=SimProblem2(),
        samples=np.array([[0.1], [0.2], [0.3]]),
        theta_best=np.array([0.2]),
        max_predictions=2,
    )
    assert written["P_sim"].endswith(".tsv")
    assert (tmp_path / "bootstrap_prediction_intervals" / "P_sim_intervals.npz").exists()
    assert posterior._save_prediction_intervals(
        outdir=str(tmp_path),
        problem=SimProblem2(),
        samples=np.empty((0, 1)),
        theta_best=np.array([0.2]),
    ) is None


def test_run_optimizer_uncertainty_dispatch_and_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        posterior,
        "run_residual_bootstrap_uncertainty",
        lambda **kwargs: {"kind": "bootstrap"},
    )
    monkeypatch.setattr(
        posterior,
        "run_profile_likelihood_uncertainty",
        lambda **kwargs: {"kind": "profile"},
    )
    res = posterior.run_optimizer_uncertainty(
        outdir=str(tmp_path),
        problem=_problem(),
        theta_best=np.array([0.2, 0.3, 0.4]),
        posterior_cfg=SimpleNamespace(method="both"),
        theta_names=["a", "b", "c"],
    )
    assert res["bootstrap"]["kind"] == "bootstrap"
    assert res["profile"]["kind"] == "profile"
    with pytest.raises(ValueError, match="Unknown posterior/uncertainty method"):
        posterior.run_optimizer_uncertainty(
            outdir=str(tmp_path),
            problem=_problem(),
            theta_best=np.array([0.2, 0.3, 0.4]),
            posterior_cfg=SimpleNamespace(method="bogus"),
        )


def test_run_residual_bootstrap_uncertainty_serial(monkeypatch, tmp_path):
    problem = _problem()
    problem.rna_obs_matched = None
    monkeypatch.setattr(posterior, "_can_pickle", lambda obj: True)

    monkeypatch.setattr(
        posterior,
        "_simulate_full_for_uncertainty",
        lambda problem, theta: {
            "P_sim": np.array([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]]),
            "A_sim": np.array([[1.0, 1.0, 1.0]]),
            "R_sim": None,
        },
    )
    monkeypatch.setattr(
        posterior,
        "_plan_uncertainty_runtime",
        lambda *args, **kwargs: SimpleNamespace(n_parallel_runs=1, threads_per_run=1),
    )
    monkeypatch.setattr(
        posterior,
        "_plot_parameter_distributions",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        posterior,
        "_save_prediction_intervals",
        lambda **kwargs: {"P_sim": "written"},
    )

    WorkerResult = namedtuple(
        "WorkerResult",
        ["b", "ok", "theta_opt", "total_loss", "f1", "f2", "f3", "f4", "error"],
    )

    def fake_worker(task):
        if task[0] == 0:
            return WorkerResult(
                0, True, np.array([0.2, 0.3, 0.4]), 1.0, 0.1, 0.2, 0.3, 0.4, ""
            )
        return WorkerResult(
            1, False, None, np.nan, np.nan, np.nan, np.nan, np.nan, "RuntimeError: boom"
        )

    monkeypatch.setattr(posterior, "_bootstrap_refit_worker", fake_worker)
    result = posterior.run_residual_bootstrap_uncertainty(
        outdir=str(tmp_path),
        problem=problem,
        theta_best=np.array([0.2, 0.3, 0.4]),
        posterior_cfg=SimpleNamespace(
            n_bootstrap=2,
            save_prediction_intervals=True,
            max_prediction_samples=5,
        ),
        theta_names=["a", "b", "c"],
    )
    assert result["samples"].shape == (1, 3)
    assert result["failures"].shape[0] == 1
    assert result["prediction_intervals"]["P_sim"] == "written"
    assert (tmp_path / "uncertainty" / "bootstrap_samples.npz").exists()
    assert (tmp_path / "uncertainty" / "bootstrap_failures.tsv").exists()


def test_run_profile_likelihood_uncertainty_serial(monkeypatch, tmp_path):
    problem = _problem()
    problem.rna_obs_matched = None
    monkeypatch.setattr(posterior, "_can_pickle", lambda obj: True)
    monkeypatch.setattr(
        posterior,
        "_make_refit_residuals_fn",
        lambda **kwargs: "unused-but-built",
    )
    monkeypatch.setattr(
        posterior,
        "_plan_uncertainty_runtime",
        lambda *args, **kwargs: SimpleNamespace(n_parallel_runs=1, threads_per_run=1),
    )
    monkeypatch.setattr(
        posterior,
        "_plot_profile_likelihood",
        lambda *args, **kwargs: "plot_dir",
    )
    monkeypatch.setattr(
        posterior,
        "_profile_refit_worker",
        lambda task: {
            "task_index": int(task[0]),
            "parameter_index": int(task[1]),
            "parameter": task[-1],
            "grid_index": int(task[2]),
            "fixed_value": float(task[3]),
            "optimized_value": float(task[3]),
            "total_loss": 1.0,
            "f1": 0.1,
            "f2": 0.2,
            "f3": 0.3,
            "f4": 0.4,
            "success": True,
            "error": "",
            "theta_opt": np.array([0.2, 0.3, 0.4]),
        },
    )
    result = posterior.run_profile_likelihood_uncertainty(
        outdir=str(tmp_path),
        problem=problem,
        theta_best=np.array([0.2, 0.3, 0.4]),
        posterior_cfg=SimpleNamespace(profile_params=1, profile_points=3),
        theta_names=["a", "b", "c"],
        parameter_indices=[0],
    )
    assert result["profile"].shape[0] == 3
    assert result["theta"].shape == (3, 3)
    assert result["plot_dir"] == "plot_dir"
    assert (tmp_path / "uncertainty" / "profile_likelihood" / "profile_likelihood.tsv").exists()
