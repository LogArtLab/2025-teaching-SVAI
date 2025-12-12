import logging
from typing import Any, Final

import numpy as np
import torch
from staliro import TestOptions, staliro
from staliro.models import Blackbox, Trace, blackbox
from staliro.optimizers import DualAnnealing
from staliro.specifications import rtamt
from elements import ACCNetFull

model_acc = ACCNetFull(in_dim=5)  # create model instance
model_acc.load_state_dict(torch.load("model.pth"))
model_acc.eval()


def inner_model(t0: float, t1: float, t2: float, v0: float, v1: float, v2: float) -> tuple[
    list[Any], list[Any], list[Any]]:
    lead_speed_profile = SpeedProfile((0, t0, t0 + t1, t0 + t1 + t2), (v0, v1, v2, 0))
    lead_driver = Driver()
    lead_v_current = 0
    lead_x_current = 30
    lead_car = Car2Dynamics(0.05, lead_x_current, lead_v_current)

    # ego car
    ego_v_set = 50
    ego_acc = model_acc
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
        if ego_v_current - lead_v_current > 0.0:
            ttc = (lead_x_current - ego_x_current) / max(ego_v_current - lead_v_current, 1e-3)  # avoid division by zero
        else:
            ttc = 10

        thbr = ego_acc(
            torch.Tensor([ego_v_current, lead_v_current, ego_x_current, lead_x_current, ttc])).detach().numpy()
        d_des = 5 + 1.8 * max(ego_v_current, 0.0)
        ego_x_current, ego_v_current, _ = ego_car.step(thbr[0], thbr[1])
        v_lead.append(lead_v_current)
        v_ego.append(ego_v_current)
        des_distance.append(d_des)
        actual_distance.append(lead_x_current - ego_x_current)
        times.append(t)
    return actual_distance, des_distance, times


from elements import SpeedProfile, Driver, Car2Dynamics, ACCNetFull

TSPAN: Final[tuple[float, float]] = (0, 60)


@blackbox()
def model(inputs: Blackbox.Inputs) -> Trace[list[float]]:
    t0 = inputs.static["t0"]
    t1 = inputs.static["t1"]
    t2 = inputs.static["t2"]
    v0 = inputs.static["v0"]
    v1 = inputs.static["v1"]
    v2 = inputs.static["v2"]

    # Lead car
    actual_distance, des_distance, times = inner_model(t0, t1, t2, v0, v1, v2)
    # print(min(np.array(actual_distance)))
    return Trace(times=times, states=np.array([actual_distance, des_distance]).T.tolist())


spec = rtamt.parse_dense("always (ad > 0)", {"ad": 0, "dd": 1})
optimizer = DualAnnealing()
options = TestOptions(
    runs=1,
    iterations=120,
    tspan=TSPAN,
    static_inputs={
        "t0": (0.0, 20.0),
        "t1": (0, 20),
        "t2": (0, 20),
        "v0": (0, 36),
        "v1": (0, 36),
        "v2": (0, 36),
    },
)

logging.basicConfig(level=logging.DEBUG)
runs = staliro(model, spec, optimizer, options)
run = runs[0]
min_cost_eval = min(run.evaluations, key=lambda e: e.cost)
min_cost_trace = min_cost_eval.extra.trace
print(min_cost_eval.cost)
print(min_cost_eval.sample.values)
