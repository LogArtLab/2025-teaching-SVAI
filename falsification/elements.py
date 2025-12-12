from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CarParams:
    mass: float = 1500.0  # kg
    max_engine_force: float = 4000.0  # N
    max_brake_force: float = 8000.0  # N
    Crr: float = 0.015  # rolling resistance
    Cd: float = 0.29  # aero drag coefficient
    frontal_area: float = 2.2  # m^2
    rho_air: float = 1.225  # kg/m^3


class Driver:
    def __init__(self, a_max=2.0, a_min=-6.0,
                 Kp_v=0.8, Ki_v=0.1):
        self.a_max = a_max
        self.a_min = a_min
        self.Kp_v = Kp_v
        self.Ki_v = Ki_v
        self.int_err_v = 0.0

    def step(self, dt, v_current, v_target):
        v_error = v_target - v_current
        self.int_err_v += v_error * dt
        a_cmd = self.Kp_v * v_error + self.Ki_v * self.int_err_v
        a_cmd = float(np.clip(a_cmd, self.a_min, self.a_max))

        if a_cmd >= 0:
            throttle = a_cmd / self.a_max if self.a_max > 0 else 0.0
            brake = 0.0
        else:
            throttle = 0.0
            brake = -a_cmd / (-self.a_min) if self.a_min < 0 else 0.0

        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))
        return throttle, brake


class SpeedProfile:

    def __init__(self, switches, speeds):
        super().__init__()
        self.t = 0
        self.speeds = speeds
        self.switches = switches

    def step(self, dt):
        i = 0
        self.t += dt
        while i < len(self.switches) and self.t > self.switches[i]:
            i += 1
        if i - 1 < len(self.switches):
            return self.speeds[i - 1]
        return self.speeds[-1]

    def get_total_s(self):
        return self.switches[-1]


class Car2Dynamics:

    def __init__(self, dt, x0, v0):
        self.params = CarParams()
        self.t = 0
        self.dt = dt
        self.x = x0
        self.v = v0

    def step(self, th, br):
        state_lead = (self.x, self.v)
        deriv_lead = self.derivatives((self.x, self.v), th, br)
        self.x, self.v = state_lead + self.dt * deriv_lead
        self.t = self.t + self.dt
        return self.x, self.v, self.t

    def derivatives(self, state, throttle, brake):
        x, v = state
        p = self.params

        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        # sign_v = np.sign(v) if v != 0 else 1.0
        sign_v = int(v >= 0)

        F_engine = p.max_engine_force * throttle
        F_brake = p.max_brake_force * brake * sign_v
        F_drag = 0.5 * p.rho_air * p.Cd * p.frontal_area * v ** 2 * sign_v
        F_roll = p.mass * 9.81 * p.Crr * sign_v
        F_net = F_engine - F_brake - F_drag - F_roll
        a = F_net / p.mass

        return np.array([v, a], dtype=float)


class SimpleACC:
    def __init__(self,
                 v_set=25.0,
                 T_headway=1.8,
                 d_min=5.0,
                 a_max=2.0,
                 a_min=-5.0,
                 Kp_v=0.8,
                 Ki_v=0.1,
                 Kp_d=1.5,
                 ttc_emergency=2.0):  # [s] threshold for hard brake
        self.v_set = v_set
        self.T_headway = T_headway
        self.d_min = d_min
        self.a_max = a_max
        self.a_min = a_min
        self.Kp_v = Kp_v
        self.Ki_v = Ki_v
        self.Kp_d = Kp_d
        self.ttc_emergency = ttc_emergency
        self.int_err_v = 0.0

    def compute(self, dt, v_ego, v_lead, x_ego, x_lead):
        gap = x_lead - x_ego
        d_des = self.d_min + self.T_headway * max(v_ego, 0.0)
        d_error = gap - d_des

        # relative speed (ego closing in if v_rel > 0)
        v_rel = v_ego - v_lead
        if v_rel > 0.0:
            ttc = gap / max(v_rel, 1e-3)  # avoid division by zero
        else:
            ttc = np.inf

        # ----- speed / distance target -----
        v_target = self.v_set

        # if lead is slower, don't try to exceed it
        if v_lead < v_target:
            v_target = v_lead

        # if too close, reduce v_target even more
        if d_error < 0.0:
            v_target += 0.1 * d_error  # d_error < 0 -> v_target decreases

        # ----- PI on speed -----
        v_error = v_target - v_ego

        self.int_err_v += v_error * dt

        a_cmd = self.Kp_v * v_error + self.Ki_v * self.int_err_v

        # extra braking when too close
        if d_error < 0.0:
            a_cmd += self.Kp_d * d_error

        # ----- TTC-based emergency override -----
        if ttc < self.ttc_emergency:
            # force near-max braking if collision is imminent
            a_cmd = self.a_min

        # saturate
        a_cmd = float(np.clip(a_cmd, self.a_min, self.a_max))

        # map to throttle / brake
        if a_cmd >= 0.0:
            throttle = a_cmd / self.a_max if self.a_max > 0 else 0.0
            brake = 0.0
        else:
            throttle = 0.0
            brake = -a_cmd / (-self.a_min) if self.a_min < 0 else 0.0

        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        return throttle, brake, d_des, ttc, v_target


def inner_model(t0: float, t1: float, t2: float, v0: float, v1: float, v2: float):
    lead_speed_profile = SpeedProfile((0, t0, t0 + t1, t0 + t1 + t2), (v0, v1, v2, 0))
    lead_driver = Driver()
    lead_v_current = 0
    lead_x_current = 30
    lead_car = Car2Dynamics(0.05, lead_x_current, lead_v_current)

    # ego car
    ego_v_set = 25
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
    ego_brs = []
    ego_ths = []
    ttcs = []

    for i in range(K):
        t += dt
        lead_v_target = lead_speed_profile.step(dt)
        lead_th, lead_br = lead_driver.step(dt, lead_v_current, lead_v_target)
        lead_x_current, lead_v_current, _ = lead_car.step(lead_th, lead_br)

        ego_th, ego_br, d_des, ttc, ego_v_target = ego_acc.compute(dt, ego_v_current, lead_v_current, ego_x_current,
                                                                   lead_x_current)
        ego_x_current, ego_v_current, _ = ego_car.step(ego_th, ego_br)
        v_lead.append(lead_v_current)
        v_ego.append(ego_v_current)
        des_distance.append(d_des)
        actual_distance.append(lead_x_current - ego_x_current)
        ego_brs.append(ego_br)
        ego_ths.append(ego_th)
        ttcs.append(min(10,ttc))
        times.append(t)
    return actual_distance, des_distance, v_ego, v_lead, ego_ths, ego_brs, ttcs, times


class ACCNetFull(nn.Module):
    """
    NN controller:
      input:  [v_ego, gap, v_rel, d_des, d_error, ttc, a_ego]
      output: [throttle, brake] in [0,1]^2
    """
    def __init__(self, in_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # x: (batch_size, in_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x