# Provenance and public-repository boundary

This repository is a **portfolio refactor** of work completed during the September 2026 RWTH Aachen / IRT *Artificial Intelligence for Robotics* summer school.

## What is represented as my work

- understanding and using the provided MuJoCo/Gymnasium robotic-ball environment;
- formulating the tangential-velocity reward used for the spiral-motion experiment;
- running SAC training with vectorized environments and observation/reward normalization;
- evaluating the learned policy and interpreting trajectory/radius behavior;
- engineering the experiment into reusable training, evaluation, testing, configuration, and documentation code after the course.

## What is not represented as my work

The following were supplied as course/research infrastructure and are **not redistributed** here:

- the Fizzy MuJoCo environment implementation and MJCF/XML model;
- the original summer-school notebooks and slides;
- pretrained SAC policies and normalization statistics;
- the VMC baseline implementation;
- the real-robot UDP/Gymnasium adapter;
- introductory programming/ML exercise solutions.

The earlier programming-exercise package explicitly states RWTH/IAT copyright and “All rights reserved.” The 2026 practical package does not contain a public software license in the material provided to me, so this repository takes the conservative approach of not republishing its source files either.

## Reproducing locally

If you are an authorized participant with the original course files, place the `fizzy_modules` package on `PYTHONPATH` (or install it into the active environment). The public code imports it dynamically at run time. The CI tests only the original reward-design code and therefore do not require the private course package.
