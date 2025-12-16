"""
3_train_NN_controller.py

This script demonstrates supervised learning of a neural network controller
to replace the rule-based SimpleACC system. The goal is to learn a mapping
from system observations to vehicle control actions (throttle and brake).

Training process:
1. Generate a diverse dataset by collecting observations and actions from
   many simulated episodes with random lead vehicle speed profiles.
2. Train a neural network (ACCNetFull) to predict throttle/brake commands
   from system state observations.
3. Evaluate the trained network on a separate test dataset.
4. Save the trained model weights to "model.pth" for later use.

Dataset:
- Observations (5 features): actual distance, desired distance, ego velocity,
  lead velocity, and time-to-collision (TTC).
- Actions (2 outputs): throttle and brake commands.
- Training: 300 episodes × ~1000 samples per episode = ~300k samples.
- Testing: 100 episodes × ~1000 samples per episode = ~100k samples.

Network architecture:
- `ACCNetFull`: A fully connected neural network defined in `elements.py`.
- Loss function: Mean Squared Error (MSE).
- Optimizer: Adam with learning rate 1e-3.
- Training: 40 epochs with batch size 256.

Outputs:
- Training and test loss curves (printed to console).
- Saved model weights in "model.pth".

Learning goals for students:
- Understand data-driven control and supervised learning for CPS.
- Learn how to collect training data from simulation.
- Practice training and evaluating neural network models in PyTorch.
- See how learned controllers can replace hand-crafted controllers.
"""

import random

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from elements import inner_model, ACCNetFull


def create_dataset(num_episodes=200):
    all_obs = []
    all_act = []
    for ep in range(num_episodes):
        obs, act = simulate_episode_collect()
        all_obs.append(obs)
        all_act.append(act)
        print(f"Episode {ep + 1}/{num_episodes} -> {len(obs)} samples")
    X = np.concatenate(all_obs, axis=0)  # (N_total, 7)
    Y = np.concatenate(all_act, axis=0)  # (N_total, 2)
    return X, Y


def simulate_episode_collect():
    t0 = random.uniform(0, 20)
    t1 = random.uniform(0, 20)
    t2 = random.uniform(0, 20)
    v0 = random.uniform(0, 36)
    v1 = random.uniform(0, 36)
    v2 = random.uniform(0, 36)
    actual_distance, des_distance, v_ego, v_lead, th, br, ttcs, times = inner_model(t0, t1, t2, v0, v1, v2)
    obs = np.array([actual_distance, des_distance, v_ego, v_lead, ttcs], dtype=np.float32).T
    act = np.array([th, br], dtype=np.float32).T
    return obs, act


X, y = create_dataset(300)


def train_accnet_full(X, Y, epochs=20, batch_size=256):
    obs_tensor = torch.from_numpy(X)
    act_tensor = torch.from_numpy(Y)

    dataset = TensorDataset(obs_tensor, act_tensor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = ACCNetFull(in_dim=X.shape[1])
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            pred = net(batch_obs)
            loss = criterion(pred, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_obs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[Train] Epoch {epoch + 1}/{epochs}, loss = {avg_loss:.6f}")

    return net


@torch.no_grad()
def evaluate_performance(net, test_dataset, batch_size=256):
    net.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    total_loss = 0.0
    n = len(test_dataset)

    for obs, act in test_loader:
        pred = net(obs)
        loss = criterion(pred, act)
        total_loss += loss.item() * obs.size(0)

    avg_loss = total_loss / n
    print(f"[Test] MSE Loss = {avg_loss:.6f}")

    return avg_loss


net = train_accnet_full(X, y, epochs=40, batch_size=256)

X, y = create_dataset(100)
obs_tensor = torch.from_numpy(X)
act_tensor = torch.from_numpy(y)
dataset = TensorDataset(obs_tensor, act_tensor)
evaluate_performance(net, dataset)

torch.save(net.state_dict(), "model.pth")
