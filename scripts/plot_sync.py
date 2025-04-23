"""*******************************************************************************
* HumARConoid
*
* Advanced Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
*******************************************************************************"""

"* Authors: Sol Choi (Jennifer) *"

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from humarcscripts.color_code import *

torque_log = "torque_log.npy"
labels = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint',
          'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_knee_joint', 'right_knee_joint',
          'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint']

plot_max_len = 400
update_interval = 0.01

fig, axs = plt.subplots(6, 2, figsize=(12, 10), sharex=True)

lines = []

for i in range(len(labels)):
    l3, = axs[i // 2, i % 2].plot([], [], label=r'$\tau_{cur}$', linewidth=2, color='#ef7159')
    axs[i // 2, i % 2].set_ylabel(f'{labels[i]}')
    axs[i // 2, i % 2].legend()
    axs[i // 2, i % 2].grid(True)
    # axs[i // 2, i % 2].set_xlim(0, plot_max_len)
    lines.append((l3,))

axs[-1, -1].set_xlabel("Time Step")
plt.suptitle("Current Torque")
plt.tight_layout()

while True:
    if os.path.exists(torque_log):
        try:
            tau_data = np.load(torque_log)
        except Exception:
            continue

        min_len = len(tau_data)
        if min_len == 0:
            continue
        if min_len > plot_max_len:
            start = min_len - plot_max_len
        else:
            start = 0
        t = np.arange(start, min_len)

        for i in range(len(labels)):
            lines[i][0].set_data(t, tau_data[start:min_len, i])
            axs[i // 2, i % 2].set_xlim(t[-1] - plot_max_len, t[-1])
            axs[i // 2][i % 2].relim()
            axs[i // 2][i % 2].autoscale_view()

        plt.pause(update_interval)
    else:
        print(f"{RED}[ERROR] {torque_log} does not exist!{RESET}")
        print(f"[INFO] Waiting for {torque_log}...")
        time.sleep(0.5)
