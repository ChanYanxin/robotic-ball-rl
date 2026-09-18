# Sim-to-real notes

The summer-school material exposes a useful separation between simulation training and a physical Fizzy interface. The physical robot exposes IMU-derived state and one motor command, while simulation can additionally access world position and velocity for reward computation.

For the spiral experiment, the policy input is IMU state only, but the reward uses privileged planar position and velocity. A physical deployment would therefore need either:

1. an external tracking system (for example camera-based x-y position and velocity) during learning/evaluation, or
2. a reward/task redesign that can be computed from onboard sensing.

A second issue is sensor convention and calibration: accelerometer/gyro scale, axis conventions, gravity handling, quaternion convention, latency, and control rate must match sufficiently between simulation and hardware.

## Evidence boundary

The uploaded notebook contains the optional hardware interface scaffold, but the sensor-collection and policy-transfer tasks are not completed in the saved file. This repository therefore **does not claim a successful real-robot policy deployment**. The verified project evidence is simulation training/evaluation plus analysis of the transfer requirements.
