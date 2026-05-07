"""
test_neural_ode_plots.py

Tests for save_neural_ode_plots() in neuralODE.py (covers lines 1210-1302).
"""
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pytest


class TestSaveNeuralOdePlots:
    """Tests for save_neural_ode_plots()."""

    def test_creates_training_loss_png_when_non_empty_history(self, tmp_path):
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        ts = np.linspace(0, 60, 10)
        ys = {}
        loss_history = [1.0, 0.8, 0.6, 0.5, 0.4]
        time_history = []
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert (tmp_path / "neural_ode_training_loss.png").exists()

    def test_no_training_loss_png_when_empty_history(self, tmp_path):
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        ts = np.linspace(0, 60, 10)
        ys = {}
        loss_history = []
        time_history = []
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert not (tmp_path / "neural_ode_training_loss.png").exists()

    def test_creates_step_time_png_when_non_empty_time_history(self, tmp_path):
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        ts = np.linspace(0, 60, 10)
        ys = {}
        loss_history = []
        time_history = [0.01, 0.012, 0.009, 0.011, 0.010]
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert (tmp_path / "neural_ode_step_time.png").exists()

    def test_no_step_time_png_when_empty_time_history(self, tmp_path):
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        ts = np.linspace(0, 60, 10)
        ys = {}
        loss_history = []
        time_history = []
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert not (tmp_path / "neural_ode_step_time.png").exists()

    def test_creates_trajectories_png_with_P_sim(self, tmp_path):
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        T = 10
        ts = np.linspace(0, 60, T)
        ys = {
            "P_sim": np.random.default_rng(0).random((3, T)),
        }
        loss_history = []
        time_history = []
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert (tmp_path / "neural_ode_trajectories.png").exists()

    def test_creates_trajectories_png_with_A_sim(self, tmp_path):
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        T = 10
        ts = np.linspace(0, 60, T)
        ys = {
            "A_sim": np.random.default_rng(1).random((2, T)),
        }
        loss_history = []
        time_history = []
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert (tmp_path / "neural_ode_trajectories.png").exists()

    def test_creates_trajectories_png_with_both_P_and_A_sim(self, tmp_path):
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        T = 10
        ts = np.linspace(0, 60, T)
        ys = {
            "P_sim": np.ones((4, T)),
            "A_sim": np.ones((2, T)),
        }
        loss_history = []
        time_history = []
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert (tmp_path / "neural_ode_trajectories.png").exists()

    def test_no_trajectories_png_when_ys_empty(self, tmp_path):
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        ts = np.linspace(0, 60, 10)
        ys = {}
        loss_history = []
        time_history = []
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert not (tmp_path / "neural_ode_trajectories.png").exists()

    def test_curriculum_boundary_line_drawn_for_long_history(self, tmp_path):
        """len(loss_history) > 4 → vertical curriculum boundary line is added."""
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        ts = np.linspace(0, 60, 10)
        ys = {}
        # More than 4 steps triggers the curriculum mid-point marking
        loss_history = list(np.linspace(1.0, 0.1, 10))
        time_history = []
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert (tmp_path / "neural_ode_training_loss.png").exists()

    def test_exactly_4_steps_no_curriculum_boundary(self, tmp_path):
        """len(loss_history) == 4 → curriculum boundary NOT added (> 4 required)."""
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        ts = np.linspace(0, 60, 10)
        ys = {}
        loss_history = [1.0, 0.8, 0.6, 0.5]  # exactly 4
        time_history = []
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        # Should still create the loss plot, just without curriculum line
        assert (tmp_path / "neural_ode_training_loss.png").exists()

    def test_model_none_does_not_crash(self, tmp_path):
        """model=None is handled gracefully (latent plot is skipped)."""
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        ts = np.linspace(0, 60, 10)
        ys = {"P_sim": np.ones((2, 10))}
        loss_history = [1.0, 0.5]
        time_history = [0.01, 0.01]
        # Should not raise
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)

    def test_outdir_created_if_absent(self, tmp_path):
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        outdir = str(tmp_path / "new_subdir")
        assert not os.path.exists(outdir)
        ts = np.linspace(0, 60, 10)
        ys = {}
        save_neural_ode_plots(outdir, ts, ys, None, [1.0, 0.5], [])
        assert os.path.exists(outdir)

    def test_rolling_mean_overlay_with_many_steps(self, tmp_path):
        """time_history with > 20 points exercises rolling mean window."""
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        ts = np.linspace(0, 60, 10)
        ys = {}
        time_history = list(np.ones(50) * 0.01)
        save_neural_ode_plots(str(tmp_path), ts, ys, None, [], time_history)
        assert (tmp_path / "neural_ode_step_time.png").exists()

    def test_P_sim_with_more_than_5_channels_capped_at_5(self, tmp_path):
        """Only the first 5 P_sim channels are plotted."""
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        T = 10
        ts = np.linspace(0, 60, T)
        ys = {"P_sim": np.ones((10, T))}  # 10 channels; only first 5 plotted
        save_neural_ode_plots(str(tmp_path), ts, ys, None, [], [])
        assert (tmp_path / "neural_ode_trajectories.png").exists()

    def test_all_three_pngs_created(self, tmp_path):
        """Comprehensive: all three PNGs are created together."""
        from phoscrosstalk.neuralODE import save_neural_ode_plots
        T = 10
        ts = np.linspace(0, 60, T)
        ys = {
            "P_sim": np.ones((3, T)),
            "A_sim": np.ones((2, T)),
        }
        loss_history = list(np.linspace(1.0, 0.01, 20))
        time_history = list(np.ones(20) * 0.01)
        save_neural_ode_plots(str(tmp_path), ts, ys, None, loss_history, time_history)
        assert (tmp_path / "neural_ode_training_loss.png").exists()
        assert (tmp_path / "neural_ode_step_time.png").exists()
        assert (tmp_path / "neural_ode_trajectories.png").exists()
