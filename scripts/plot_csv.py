# """*******************************************************************************
# * HumARConoid
# *
# * Advanced Humanoid Locomotion Strategy using Reinforcement Learning
# *
# *     https://github.com/S-CHOI-S/HumARConoid.git
# *
# * Advanced Robot Control Lab. (ARC)
# * 	  @ Korea Institute of Science and Technology
# *
# *	  https://sites.google.com/view/kist-arc
# *
# *******************************************************************************"""

# "* Authors: Sol Choi (Jennifer) *"

# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter

# def load_csv(file_path):
#     data = pd.read_csv(file_path)
#     return data

# def preprocess_df(data, smoothing=100):
#     data.columns = ['Wall_time', 'Step', 'Value']
    
#     # 스무딩 적용
#     data['Value_RollingMean'] = data['Value'].rolling(window=smoothing).mean()
#     data['Value_RollingStd'] = data['Value'].rolling(window=smoothing).std()
#     return data

# def preprocess_dff(data, smoothing=100):
#     data.columns = ['timestep', 'command', 'action']
    
#     # 스무딩 적용
#     data['action_RollingMean'] = data['action'].rolling(window=smoothing).mean()
#     data['action_RollingStd'] = data['action'].rolling(window=smoothing).std()
#     return data

# def draw_plot(data1, data2, label1="Dataset 1", label2="Dataset 2", figure_number=None):
#     font_size = 14
#     if figure_number is not None:
#         plt.figure(figure_number, figsize=(8, 6))
#         ax = plt.gca()
    
#     ax.fill_between(data1['Step'],
#                     data1['Value_RollingMean'] - data1['Value_RollingStd'],
#                     data1['Value_RollingMean'] + data1['Value_RollingStd'],
#                     color='r', alpha=0.2, label=f'{label1} ±1 Std Dev')

#     ax.plot(data1['Step'], data1['Value_RollingMean'], label=f'{label1} Mean', color='r')
#     # ax.plot(data2['Step'], data2['Value_RollingMean'], label=f'{label2} Mean', color='g')

#     ax.set_xlabel('Steps', fontsize=font_size)
#     ax.set_ylabel('Value', fontsize=font_size)
#     ax.tick_params(axis='both', which='major', labelsize=font_size-2)
#     ax.legend(fontsize=font_size, loc='upper left')
#     ax.grid(True)

#     formatter = ScalarFormatter(useMathText=True)
#     formatter.set_scientific(True)
#     formatter.set_powerlimits((-1, 1))
#     ax.xaxis.set_major_formatter(formatter)

#     plt.title('Step vs Value with Mean and Std Dev', fontsize=font_size)
#     plt.tight_layout()
#     plt.show()


# def draw_plott(data1, data2, label1="Dataset 1", label2="Dataset 2", figure_number=None):
#     font_size = 14
#     if figure_number is not None:
#         plt.figure(figure_number, figsize=(8, 6))
#         ax = plt.gca()
    
#     ax.fill_between(data1['timestep'],
#                     data1['action_RollingMean'] - data1['action_RollingStd'],
#                     data1['action_RollingMean'] + data1['action_RollingStd'],
#                     color='g', alpha=0.2, label=f'{label1} ±1 Std Dev')
#     # ax.fill_between(data2['Step'],
#     #                 data2['Value_RollingMean'] - data2['Value_RollingStd'],
#     #                 data2['Value_RollingMean'] + data2['Value_RollingStd'],
#     #                 color='g', alpha=0.2, label=f'{label2} ±1 Std Dev')

#     ax.plot(data1['timestep'], data1['command'], label=f'{label1} command', color='r')
#     ax.plot(data1['timestep'], data1['action_RollingMean'], label=f'{label1} Action Mean', color='g')
#     # ax.plot(data2['Step'], data2['Value_RollingMean'], label=f'{label2} Mean', color='g')

#     ax.set_xlabel('timestep', fontsize=font_size)
#     ax.set_ylabel('Value', fontsize=font_size)
#     ax.tick_params(axis='both', which='major', labelsize=font_size-2)
#     ax.legend(fontsize=font_size, loc='upper left')
#     ax.grid(True)

#     formatter = ScalarFormatter(useMathText=True)
#     formatter.set_scientific(True)
#     formatter.set_powerlimits((-1, 1))
#     ax.xaxis.set_major_formatter(formatter)

#     plt.title('Step vs Value with Mean and Std Dev', fontsize=font_size)
#     plt.tight_layout()


# import os

# current_dir = os.getcwd()
# print(f"Current Directory: {current_dir}")

# file_path1 = os.path.join(current_dir, '2025-01-09_12-38-08.csv')
# file_path2 = os.path.join(current_dir, '2025-01-09_12-38-08.csv')

# # file_path3 = os.path.join(current_dir, 'simulation_dataf.csv')
# # file_path4 = os.path.join(current_dir, 'simulation_datar.csv')

# if not os.path.exists(file_path1):
#     print(f"File not found: {file_path1}")
# else:
#     print(f"File found: {file_path1}")
    
# data1 = preprocess_df(load_csv(file_path1), smoothing=20)
# data2 = preprocess_df(load_csv(file_path2), smoothing=20)

# # data3 = preprocess_dff(load_csv(file_path3), smoothing=30)
# # data4 = preprocess_dff(load_csv(file_path4), smoothing=30)

# draw_plot(data1, data2, "Reward", figure_number=0, )
# # plt.savefig(os.path.join(current_dir, "reward_0.svg"), format="svg")
# draw_plot(data2, data2, "Reward", figure_number=1, )
# # plt.savefig(os.path.join(current_dir, "reward_1.svg"), format="svg")

# # draw_plott(data3, data2, "Command", figure_number=2, )
# # plt.savefig(os.path.join(current_dir, "reward_2.svg"), format="svg")

# # draw_plott(data4, data2, "Command", figure_number=3, )
# # plt.savefig(os.path.join(current_dir, "reward_3.svg"), format="svg")
# # plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CSV 파일 경로 설정
csv_file_path = '2025-01-09_12-38-08.csv'  # 여기에 실제 파일 경로를 입력하세요.

# 데이터 로드 및 전처리
def preprocess_data(file_path, smoothing=20):
    data = pd.read_csv(file_path)
    data.columns = ['Wall_time', 'Step', 'Value']
    
    # 스무딩 처리
    data['Value_RollingMean'] = data['Value'].rolling(window=smoothing).mean()
    data['Value_RollingStd'] = data['Value'].rolling(window=smoothing).std()
    
    return data

# 데이터 로드 및 전처리 실행
data = preprocess_data(csv_file_path, smoothing=20)

# Seaborn 스타일 설정
sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)

# Seaborn을 사용한 시각화
plt.figure(figsize=(8, 6))

# 라인 플롯과 표준편차 영역 추가
sns.lineplot(x='Step', y='Value_RollingMean', data=data, label='Mean Value', color='b', linewidth=2)
plt.fill_between(data['Step'], 
                 data['Value_RollingMean'] - data['Value_RollingStd'], 
                 data['Value_RollingMean'] + data['Value_RollingStd'], 
                 alpha=0.3, color='b', label='±1 Std Dev')

# 그래프 제목 및 축 레이블 설정
plt.title('Step vs Value (Smoothed with ±1 Std Dev)', fontsize=20, weight='bold', pad=20)
plt.xlabel('Step', fontsize=16, weight='bold', labelpad=10)
plt.ylabel('Value', fontsize=16, weight='bold', labelpad=10)

# 눈금 조정
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 범례 설정
plt.legend(fontsize=14, loc='upper left', frameon=True, shadow=True)

# 그래프 출력
plt.tight_layout()
plt.show()