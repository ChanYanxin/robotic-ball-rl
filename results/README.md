# Recorded experiment artifacts

These files were reconstructed from outputs saved in the September 2026 practical notebook.

- `training_log.csv`: de-duplicated Stable-Baselines3 rollout/training logs from the 100,000-transition SAC run.
- `training_curve.svg`: compact vector visualization generated directly from `training_log.csv`.

The training budget completed at 100,000 transitions. The last periodic log in the notebook is at 99,200 transitions, where `ep_rew_mean` is 131 over the current SB3 rollout window. This value is a training log statistic, not an independent success metric.

The saved notebook also contains a 50-second deterministic rollout figure showing sustained tangential motion with substantial radial drift. The public repository documents that behavior and provides evaluation code, while avoiding redistribution of course infrastructure.
