import os
import csv
import shutil
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


path_root = './data/argoverse_with_LRP/val'
file_names = os.listdir(path_root)

# +
fde_list = []
ade_list = []
low_fde_list = []
low_ade_list = []
agent_lrp_low_x = []
agent_lrp_low_y = []
agent_lrp_high_x = []
agent_lrp_high_y = []
diff_lrp_low_fde_x = []
diff_lrp_low_fde_y = []
diff_lrp_high_fde_x = []
diff_lrp_high_fde_y = []

for file_name in file_names:
  # print(file_name)  
  df = pd.read_pickle(path_root + "/" +file_name)
  fde = df['metric']['fde']
  ade = df['metric']['ade']
  fde_list.append(fde)
  ade_list.append(ade)
  if fde < 1:
    # low_fde_list.append(df['file_path'])
    # df['file_path']
    agent_lines_lrp = sum(df['agent_oracle_centerline_lrp'])
    agent_features_lrp = sum(df['agent_features_lrp'])
    lrp_diff = abs(agent_lines_lrp - agent_features_lrp)
    agent_lrp_low_x.append(agent_features_lrp[0])
    agent_lrp_low_y.append(agent_features_lrp[1])

    diff_lrp_low_fde_x.append(lrp_diff[0])
    diff_lrp_low_fde_y.append(lrp_diff[1])

  else : 
    agent_lines_lrp = sum(df['agent_oracle_centerline_lrp'])
    agent_features_lrp = sum(df['agent_features_lrp'])
    lrp_diff = abs(agent_lines_lrp - agent_features_lrp)

    agent_lrp_high_x.append(agent_features_lrp[0])
    agent_lrp_high_y.append(agent_features_lrp[1])
    diff_lrp_high_fde_x.append(lrp_diff[0])
    diff_lrp_high_fde_y.append(lrp_diff[1])

  # if ade < 1:
  #   low_ade_list.append(df['file_path'])
# -

# fig = plt.figure()
# ax0 = fig.add_subplot(2, 1, 1)
# ax0 = plt.hist(fde_list,  range=(0, 10), bins=100, alpha=0.5, density=True, stacked=True)
# ax1 = fig.add_subplot(2, 1, 2)
# ax1 = plt.hist(ade_list,  range=(0, 10), bins=100, alpha=0.5, density=True, stacked=True)

len(agent_lrp_high_x), len(agent_lrp_low_x)

# +
fig = plt.figure(figsize=(15,15))

import numpy as np

ax0 = fig.add_subplot(3, 4, 1)
ax0 = plt.hist(diff_lrp_low_fde_x,  range=(0, 3),   bins=100, alpha=0.5, density=True, stacked=True)
ax1 = fig.add_subplot(3, 4, 2)
ax1 = plt.hist(diff_lrp_low_fde_y,   range=(0, 3),bins=100, alpha=0.5, density=True, stacked=True)
ax2 = fig.add_subplot(3, 4, 3)
ax2 = plt.hist(agent_lrp_low_x,   range=(0, 3),bins=100, alpha=0.5, density=True, stacked=True)
ax3 = fig.add_subplot(3, 4, 4)
ax3 = plt.hist(agent_lrp_low_y,   range=(0, 1),bins=100, alpha=0.5, density=True, stacked=True)


ax4 = fig.add_subplot(3, 4, 5)
ax4 = plt.hist(diff_lrp_high_fde_x,     range=(0, 3),bins=100, alpha=0.5, density=True, stacked=True)
ax5 = fig.add_subplot(3, 4, 6)
ax5 = plt.hist(diff_lrp_high_fde_y,   range=(0, 3),bins=100, alpha=0.5, density=True, stacked=True)
ax6 = fig.add_subplot(3, 4, 7)
ax6 = plt.hist(agent_lrp_high_x,   range=(0, 3),bins=100, alpha=0.5, density=True, stacked=True)
ax7 = fig.add_subplot(3, 4, 8)
ax7 = plt.hist(agent_lrp_high_y,   range=(0, 1),bins=100, alpha=0.5, density=True, stacked=True)


print(np.mean(diff_lrp_high_fde_x) - np.mean(diff_lrp_low_fde_x))
print(np.mean(diff_lrp_high_fde_y)- np.mean(diff_lrp_low_fde_y))
print(np.mean(agent_lrp_high_x) - np.mean(agent_lrp_low_x))
print(np.mean(agent_lrp_high_y) - np.mean(agent_lrp_low_y))

plt.savefig('diff.png')



# +
# 0.1 / 5.0
# 0.20173857
# 0.052210152
# -0.07049544
# 0.012250347
# -







