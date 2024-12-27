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

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_df(data, smoothing=100):
    data.columns = ['Wall_time', 'Step', 'Value']
    
    # 스무딩 적용
    data['Value_RollingMean'] = data['Value'].rolling(window=smoothing).mean()
    data['Value_RollingStd'] = data['Value'].rolling(window=smoothing).std()
    return data

def preprocess_dff(data, smoothing=100):
    data.columns = ['timestep', 'command', 'action']
    
    # 스무딩 적용
    data['action_RollingMean'] = data['action'].rolling(window=smoothing).mean()
    data['action_RollingStd'] = data['action'].rolling(window=smoothing).std()
    return data

def draw_plot(data1, data2, label1="Dataset 1", label2="Dataset 2", figure_number=None):
    font_size = 14
    if figure_number is not None:
        plt.figure(figure_number, figsize=(8, 6))
        ax = plt.gca()
    
    ax.fill_between(data1['Step'],
                    data1['Value_RollingMean'] - data1['Value_RollingStd'],
                    data1['Value_RollingMean'] + data1['Value_RollingStd'],
                    color='r', alpha=0.2, label=f'{label1} ±1 Std Dev')

    ax.plot(data1['Step'], data1['Value_RollingMean'], label=f'{label1} Mean', color='r')
    # ax.plot(data2['Step'], data2['Value_RollingMean'], label=f'{label2} Mean', color='g')

    ax.set_xlabel('Steps', fontsize=font_size)
    ax.set_ylabel('Value', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)
    ax.legend(fontsize=font_size, loc='upper left')
    ax.grid(True)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    plt.title('Step vs Value with Mean and Std Dev', fontsize=font_size)
    plt.tight_layout()


def draw_plott(data1, data2, label1="Dataset 1", label2="Dataset 2", figure_number=None):
    font_size = 14
    if figure_number is not None:
        plt.figure(figure_number, figsize=(8, 6))
        ax = plt.gca()
    
    ax.fill_between(data1['timestep'],
                    data1['action_RollingMean'] - data1['action_RollingStd'],
                    data1['action_RollingMean'] + data1['action_RollingStd'],
                    color='g', alpha=0.2, label=f'{label1} ±1 Std Dev')
    # ax.fill_between(data2['Step'],
    #                 data2['Value_RollingMean'] - data2['Value_RollingStd'],
    #                 data2['Value_RollingMean'] + data2['Value_RollingStd'],
    #                 color='g', alpha=0.2, label=f'{label2} ±1 Std Dev')

    ax.plot(data1['timestep'], data1['command'], label=f'{label1} command', color='r')
    ax.plot(data1['timestep'], data1['action_RollingMean'], label=f'{label1} Action Mean', color='g')
    # ax.plot(data2['Step'], data2['Value_RollingMean'], label=f'{label2} Mean', color='g')

    ax.set_xlabel('timestep', fontsize=font_size)
    ax.set_ylabel('Value', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)
    ax.legend(fontsize=font_size, loc='upper left')
    ax.grid(True)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)

    plt.title('Step vs Value with Mean and Std Dev', fontsize=font_size)
    plt.tight_layout()


import os

current_dir = os.getcwd()
print(f"Current Directory: {current_dir}")

file_path1 = os.path.join(current_dir, '2024-12-06_20-50-01.csv')
file_path2 = os.path.join(current_dir, '2024-12-08_21-59-04.csv')

file_path3 = os.path.join(current_dir, 'simulation_dataf.csv')
file_path4 = os.path.join(current_dir, 'simulation_datar.csv')

if not os.path.exists(file_path1):
    print(f"File not found: {file_path1}")
else:
    print(f"File found: {file_path1}")
    
data1 = preprocess_df(load_csv(file_path1), smoothing=20)
data2 = preprocess_df(load_csv(file_path2), smoothing=20)

data3 = preprocess_dff(load_csv(file_path3), smoothing=30)
data4 = preprocess_dff(load_csv(file_path4), smoothing=30)

draw_plot(data1, data2, "Reward", figure_number=0, )
plt.savefig(os.path.join(current_dir, "reward_0.svg"), format="svg")
draw_plot(data2, data2, "Reward", figure_number=1, )
plt.savefig(os.path.join(current_dir, "reward_1.svg"), format="svg")

draw_plott(data3, data2, "Command", figure_number=2, )
plt.savefig(os.path.join(current_dir, "reward_2.svg"), format="svg")

draw_plott(data4, data2, "Command", figure_number=3, )
plt.savefig(os.path.join(current_dir, "reward_3.svg"), format="svg")
plt.show()
