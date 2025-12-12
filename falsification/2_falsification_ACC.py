import logging
from typing import Final, Any

import numpy as np
from staliro import TestOptions, Trace, staliro
from staliro.models import blackbox, Blackbox
from staliro.optimizers import DualAnnealing
from staliro.specifications import rtamt

from elements import SpeedProfile, Driver, Car2Dynamics, SimpleACC

TSPAN: Final[tuple[float, float]] = (0, 60)
dt = 0.05


@blackbox()
def model(inputs: Blackbox.Inputs) -> Trace[list[float]]:
    t0 = inputs.static["t0"]
    t1 = inputs.static["t1"]
    t2 = inputs.static["t2"]
    v0 = inputs.static["v0"]
    v1 = inputs.static["v1"]
    v2 = inputs.static["v2"]

    # Lead car
    actual_distance, des_distance, times = scenario_evolution(t0, t1, t2, v0, v1, v2)
    # print(min(np.array(actual_distance)))
    return Trace(times=times, states=np.array([actual_distance, des_distance]).T.tolist())


def scenario_evolution(t0: float, t1: float, t2: float, v0: float, v1: float, v2: float) -> tuple[
    list[Any], list[Any], list[Any]]:
    lead_speed_profile = SpeedProfile((0, t0, t0 + t1, t0 + t1 + t2), (v0, v1, v2, 0))
    lead_driver = Driver()
    lead_v_current = 0
    lead_x_current = 30
    lead_car = Car2Dynamics(dt, lead_x_current, lead_v_current)

    # ego car
    ego_v_set = 50
    ego_acc = SimpleACC(v_set=ego_v_set)
    ego_v_current = 0
    ego_x_current = 0
    ego_car = Car2Dynamics(dt, ego_x_current, ego_v_current)
    t = 0

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

        ego_th, ego_br, d_des, _, ego_v_target = ego_acc.compute(dt, ego_v_current, lead_v_current, ego_x_current,
                                                                 lead_x_current)
        ego_x_current, ego_v_current, _ = ego_car.step(ego_th, ego_br)
        v_lead.append(lead_v_current)
        v_ego.append(ego_v_current)
        des_distance.append(d_des)
        actual_distance.append(lead_x_current - ego_x_current)
        times.append(t)
    return actual_distance, des_distance, times


spec = rtamt.parse_dense("always (ad > 0)", {"ad": 0, "rd": 1})
optimizer = DualAnnealing()
options = TestOptions(
    runs=1,  # number of indipendent optimizations
    iterations=120,
    tspan=TSPAN,
    static_inputs={
        "t0": (0.0, 20.0),
        "t1": (0.0, 20.0),
        "t2": (0.0, 20.0),
        "v0": (0.0, 36.0),
        "v1": (0.0, 36.0),
        "v2": (0.0, 36.0),
    },
)

logging.basicConfig(level=logging.DEBUG)
runs = staliro(model, spec, optimizer, options)
run = runs[0]
min_cost_eval = min(run.evaluations, key=lambda e: e.cost)
min_cost_trace = min_cost_eval.extra.trace
print(min_cost_eval.cost)
print(min_cost_eval.sample.values)
