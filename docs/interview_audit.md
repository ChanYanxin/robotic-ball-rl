# Interview audit: underactuated robotic-ball RL

This document separates **recorded experimental evidence** from **theoretical
explanations** and **proposed improvements**. It does not claim that any
unrecorded baseline, revised reward, or hardware transfer was executed.

## What the experiment actually did

- **Plant:** provided Fizzy single-motor underactuated robotic ball, with
  MuJoCo/Gymnasium simulator supplied by the RWTH summer school.
- **Policy input:** 10-D IMU state (acceleration 3, angular velocity 3,
  quaternion 4); no world-frame x-y position or velocity in policy input.
- **Action:** one continuous normalized motor command in [-1, 1] at 10 Hz.
- **Training:** SB3 SAC, MLP policy, 16 vectorized environments, 100,000
  aggregate transitions, 200-step episodes, gamma=0.995, two gradient updates
  per 16 new transitions, VecNormalize for both observations and rewards.
- **Training statistic:** ep_rew_mean=131 in the final available periodic log
  at 99,200 transitions; this is neither an independent success rate nor
  an evaluation score.
- **Evaluation described in the course notebook:** deterministic rollout
  to 500 steps (50 seconds), showing circulating motion and radial drift.
- **Missing evidence:** multi-seed ablation, DDPG/TD3 head-to-head benchmark,
  a tested radius-aware reward, an independent baseline success rate, and
  successful transfer to a physical robot.

## Six common follow-ups

### 1. Why SAC for continuous control?

The control signal is a real-valued motor command, not a discrete class.
SB3's SAC supports continuous Box actions and learns a stochastic policy
with an entropy-regularized objective. Off-policy replay can reuse
experiences, and twin Q critics help reduce optimistic Q-value estimates.
It is an engineering choice, not proof that SAC is optimal for this task.

### 2. How would you choose DDPG, TD3, or SAC?

- **DDPG:** deterministic actor/critic with replay; simpler but sensitive
  to critic overestimation and exploration-noise choices.
- **TD3:** reduces those issues with twin critics, delayed actor updates,
  and target-policy smoothing; still uses a deterministic policy.
- **SAC:** stochastic entropy-regularized policy and twin critics,
  encouraging exploration. Suitable for this continuous, uncertain task.
- **Evidence boundary:** the public repository trained SAC only.

### 3. What exactly was the reward?

At planar location p=(x,y), set alpha=atan2(y,x) and a counter-clockwise
unit tangent t=(-sin(alpha),cos(alpha)). With planar velocity v, the
per-step reward is dot(t,v)/norm(v) when speed exceeds epsilon, and 0
when stationary. It lies in [-1,1] and encourages *direction*, not
fixed radius or nonzero speed.

The empirical radial drift is consistent with a reward-design limitation:
a velocity can be tangential while its trajectory gradually migrates
radially over time. A high rollout reward therefore need not indicate
good circular tracking.

### 4. Why does partial observation matter?

The reward reads privileged simulation x-y position and velocity, but
the policy only sees IMU signals. Different trajectories can share
similar instantaneous IMU readings. An MLP without explicit history
cannot, in general, infer the world-frame radius and heading uniquely.
Possible research directions: observation history, recurrent policy,
state estimation, or additional external position sensing.

### 5. Can we deploy directly on the real robot?

Not without testing and adaptation. Possible gaps include wheel/surface
friction, inertia, motor response, sensor axis conventions, IMU bias,
orientation conventions, timing, and latency. Reward evaluation on
hardware additionally requires world position/velocity estimates,
such as camera tracking, or a redesigned onboard-only objective.
The recorded materials do not demonstrate a completed real-robot run.

### 6. What do 16 environments and 100,000 timesteps mean?

16 independent simulators collect transitions for one shared SAC model.
One simultaneous vector step provides 16 transitions. Total timesteps
counts *aggregate interactions*, not 100,000 optimizer updates and not
100,000 steps per simulator. A 200-step episode is 20 simulated seconds
at 10 Hz. The training configuration in this repository performs
2 gradient updates for each block of 16 collected transitions.

## Newly added independent trajectory diagnostics

`evaluate.py` now writes `evaluation_metrics.json`, including
mean radius, radial standard deviation and coefficient of variation,
signed start-to-end radial drift, radius range, path length, signed
counter-clockwise turns, and raw rollout reward totals. These are
geometry and reward diagnostics; no artificial success threshold has
been imposed.

**Important:** These metrics are implemented as evaluation *capabilities*.
No new policy has been trained and no fresh experiment has been run by
adding this code. The public repository intentionally does not contain
the non-redistributable simulator or model checkpoint.

## Next controlled experiment: radius-aware reward (proposal only)

With desired radius R>0 and radial velocity v_r = dot(p,v)/norm(p),
consider

`r2 = w_t * tangent_alignment - w_r * abs(norm(p)-R)/R - w_v * abs(v_r)/max(norm(v),eps)`.

Weights should be tuned, and a consistent convention is needed near
the origin. Evaluate against the original reward at multiple fixed
seeds, and compare **radial tracking error, signed turns, episode return,
and success criteria defined before training**. Because the policy
lacks x-y position, consider a history encoder or state estimator too;
reward shaping alone cannot repair missing observability.

## Engineering notes

- The YAML in `configs/` documents settings but is **not parsed** by
  `train.py`; do not claim the full experiment is YAML-config driven.
- `VecNormalize` uses its own gamma default unless explicitly supplied;
  the current training code sets SAC gamma=0.995 but does not pass gamma
  to `VecNormalize`. A follow-up run could align the two, clearly marked
  as a changed setting rather than historical replication.
- The original reward wrapper deliberately suppresses base-task
  termination; only the outer TimeLimit ends an episode. This preserves
  the saved task formulation but requires care if the external environment
  has independent safety termination conditions.
