"""
0_lead_dynamics.py

Simple lead-vehicle dynamics demonstration for students.

This script simulates a lead vehicle using the components provided in
`elements.py`: a `SpeedProfile` that defines a time-varying target speed,
a simple `Driver` controller that produces throttle and brake commands to
track the target speed, and `Car2Dynamics` which integrates the vehicle's
longitudinal dynamics.

Simulation details:
- Discrete time step: `dt = 0.05` seconds
- Number of iterations: 1000 (looped calls to `step` on the models)
- Logged signals: actual speed, target speed, throttle, brake, and time

Outputs:
- A two-panel plot showing (1) actual vs target speed and (2) throttle
    and brake signals over time.

Learning goals for students:
- Observe closed-loop behavior of a simple driver-vehicle system.
- See how a predefined speed profile is tracked and how actuator signals
    (throttle/brake) change during tracking.
- Practice running discrete-time simulations and visualizing results.
"""

import matplotlib.pyplot as plt

from elements import Car2Dynamics, SpeedProfile, Driver

speed_profile = SpeedProfile((10, 20, 30, 45), (10, 30, 5, 0))
driver = Driver()
v_current = 0
x_current = 0
cd = Car2Dynamics(0.05, 0, v_current)

v_current_storage = []
v_target_storage = []
throttle = []
brake = []
times = []
dt = 0.05
for i in range(1000):
    v_target = speed_profile.step(dt)
    th, br = driver.step(dt, v_current, v_target)
    x_current, v_current, t = cd.step(th, br)
    v_current_storage.append(v_current)
    v_target_storage.append(v_target)
    throttle.append(th)
    brake.append(br)
    times.append(t)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.plot(times, v_current_storage, label="Actual Speed")
ax1.plot(times, v_target_storage, label="Expected Speed")
ax1.set_ylabel("Speed (m/s)")
ax1.legend()
ax2.plot(times, throttle, label="Throttle")
ax2.plot(times, brake, label="Brake")
ax2.legend()

plt.xlabel("Time (s)")
plt.show()
