# Experiment notes

## System

Fizzy is an underactuated robotic ball with a single motor. In the supplied summer-school environment, the controller acts at 10 Hz with one normalized motor command.

## Policy observation

The course environment exposes a 10-value IMU state consisting of:

- accelerometer: 3 values;
- gyroscope: 3 values;
- orientation quaternion: 4 values.

The original goal-conditioned task also exposes a 2-D target displacement, but the spiral reward experiment intentionally filters the policy observation down to the IMU state.

## Reward used in the recorded run

For planar position `p = [x, y]` and velocity `v = [vx, vy]`:

```text
alpha = atan2(y, x)
tangent = [-sin(alpha), cos(alpha)]
reward = dot(tangent, v) / ||v||
```

If speed is nearly zero, reward is zero. This rewards counter-clockwise tangential motion but does **not** penalize radial drift. The recorded trajectory therefore expands and oscillates in radius rather than converging to a fixed circle.

## Training configuration

- algorithm: Soft Actor-Critic (SAC);
- policy: MLP policy;
- total transition budget: 100,000;
- parallel environments: 16;
- episode length: 200 steps (20 s at 10 Hz);
- gamma: 0.995;
- gradient steps: 2 per 16 newly collected transitions;
- VecNormalize: observation and reward normalization enabled;
- seed: 42.

The saved rollout figure evaluates the learned policy for up to 500 steps (50 s), which makes the radial drift visible beyond the 20-second training horizon.
