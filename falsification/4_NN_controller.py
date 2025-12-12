import torch

from elements import Car2Dynamics, SpeedProfile, Driver, SimpleACC, ACCNetFull

model_acc = ACCNetFull(in_dim=5)  # create model instance
model_acc.load_state_dict(torch.load("model.pth"))
model_acc.eval()

# Lead car
lead_speed_profile = SpeedProfile((5, 10, 15, 20), (5, 10, 5, 0))

lead_driver = Driver()
lead_v_current = 0
lead_x_current = 30
lead_car = Car2Dynamics(0.05, lead_x_current, lead_v_current)

# ego car
ego_v_set = 20
ego_acc = SimpleACC(v_set=ego_v_set)
ego_v_current = 0
ego_x_current = 0
ego_car = Car2Dynamics(0.05, ego_x_current, ego_v_current)

t = 0
dt = 0.05
N = lead_speed_profile.get_total_s()
K = int(N // dt)

v_ego = []
v_lead = []
times = []
actual_distance = []
des_distance = []

for i in range(K):
    t += dt
    lead_v_target = lead_speed_profile.step(dt)
    lead_th, lead_br = lead_driver.step(dt, lead_v_current, lead_v_target)
    lead_x_current, lead_v_current, _ = lead_car.step(lead_th, lead_br)
    if ego_v_current - lead_v_current > 0.0 and lead_x_current - ego_x_current > 0:
        ttc = (lead_x_current - ego_x_current) / max(ego_v_current - lead_v_current, 1e-3)  # avoid division by zero
    else:
        ttc = 10

    thbr = model_acc(
        torch.Tensor([ego_v_current, lead_v_current, ego_x_current, lead_x_current, ttc])).detach().numpy()
    d_des = 5 + 1.8 * max(ego_v_current, 0.0)

    ego_x_current, ego_v_current, _ = ego_car.step(thbr[0], thbr[1])
    v_lead.append(lead_v_current)
    v_ego.append(ego_v_current)
    des_distance.append(d_des)
    actual_distance.append(lead_x_current - ego_x_current)
    times.append(t)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

ax1.plot(times, v_lead, label="Lead Speed")
ax1.plot(times, v_ego, label="Ego Speed")
ax1.axhline(ego_v_set, label="Ego v_set", c="k")
ax1.set_ylabel("Speed (m/s)")
ax1.legend()

ax2.plot(times, actual_distance, label="Actual Distance")
ax2.plot(times, des_distance, label="Desired Distance")
ax2.legend()

plt.xlabel("Time (s)")
plt.show()
