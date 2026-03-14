# %% 导入所需模块
import os
import glob
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import find_events, Epochs

# %% 设置参数
data_dir = "batch_preprocessing_results"
reject_criteria = dict(mag=10e-12)
tmin, tmax = -0.1, 0.2
baseline = (-0.1, 0)

# 输出目录（将所有保存的文件放到此目录）
output_dir = "conference"
os.makedirs(output_dir, exist_ok=True)

# %% 查找所有fif文件
fif_pattern = os.path.join(data_dir, "*", "*.fif")
fif_files = glob.glob(fif_pattern)
fif_files = sorted(fif_files)

print(f"找到 {len(fif_files)} 个fif文件:")
for i, file in enumerate(fif_files):
    print(f"{i+1}. {os.path.basename(file)}")

# %% 读取数据1
if len(fif_files) > 0:
    raw1 = mne.io.read_raw_fif(fif_files[0], preload=True, verbose=False)
    filename1 = os.path.splitext(os.path.basename(fif_files[0]))[0]
    events1 = find_events(raw1, stim_channel='Trigger', verbose=False)
    epochs1 = Epochs(raw1, events1, 1000000000,
                     tmin=tmin, baseline=baseline, detrend=1,
                     tmax=tmax, reject=reject_criteria, preload=True, verbose=False)
    evoked1 = epochs1.average()
    print(f"数据1 ({filename1}): {len(epochs1)} epochs")

# %% 读取数据2
if len(fif_files) > 1:
    raw2 = mne.io.read_raw_fif(fif_files[1], preload=True, verbose=False)
    filename2 = os.path.splitext(os.path.basename(fif_files[1]))[0]
    events2 = find_events(raw2, stim_channel='Trigger', verbose=False)
    epochs2 = Epochs(raw2, events2, 1000000000,
                     tmin=tmin, baseline=baseline, detrend=1,
                     tmax=tmax, reject=reject_criteria, preload=True, verbose=False)
    evoked2 = epochs2.average()
    print(f"数据2 ({filename2}): {len(epochs2)} epochs")

# %% 读取数据3
if len(fif_files) > 2:
    raw3 = mne.io.read_raw_fif(fif_files[2], preload=True, verbose=False)
    filename3 = os.path.splitext(os.path.basename(fif_files[2]))[0]
    events3 = find_events(raw3, stim_channel='Trigger', verbose=False)
    epochs3 = Epochs(raw3, events3, 1000000000,
                     tmin=tmin, baseline=baseline, detrend=1,
                     tmax=tmax, reject=reject_criteria, preload=True, verbose=False)
    evoked3 = epochs3.average()
    print(f"数据3 ({filename3}): {len(epochs3)} epochs")

# %% 读取数据4
if len(fif_files) > 3:
    raw4 = mne.io.read_raw_fif(fif_files[3], preload=True, verbose=False)
    filename4 = os.path.splitext(os.path.basename(fif_files[3]))[0]
    events4 = find_events(raw4, stim_channel='Trigger', verbose=False)
    epochs4 = Epochs(raw4, events4, 1000000000,
                     tmin=tmin, baseline=baseline, detrend=1,
                     tmax=tmax, reject=reject_criteria, preload=True, verbose=False)
    evoked4 = epochs4.average()
    print(f"数据4 ({filename4}): {len(epochs4)} epochs")

# %% 读取数据5
if len(fif_files) > 4:
    raw5 = mne.io.read_raw_fif(fif_files[4], preload=True, verbose=False)
    filename5 = os.path.splitext(os.path.basename(fif_files[4]))[0]
    events5 = find_events(raw5, stim_channel='Trigger', verbose=False)
    epochs5 = Epochs(raw5, events5, 1000000000,
                     tmin=tmin, baseline=baseline, detrend=1,
                     tmax=tmax, reject=reject_criteria, preload=True, verbose=False)
    evoked5 = epochs5.average()
    print(f"数据5 ({filename5}): {len(epochs5)} epochs")

# %% 读取数据6
import numpy as np
import math
from typing import List, Tuple

if len(fif_files) > 5:
    raw6 = mne.io.read_raw_fif(fif_files[5], preload=True, verbose=False)
    filename6 = os.path.splitext(os.path.basename(fif_files[5]))[0]
    events6 = find_events(raw6, stim_channel='Trigger', verbose=False)
    epochs6 = Epochs(raw6, events6, 1000000000,
                     tmin=tmin, baseline=baseline, detrend=1,
                     tmax=tmax, reject=reject_criteria, preload=True, verbose=False)
    evoked6 = epochs6.average()
    print(f"数据6 ({filename6}): {len(epochs6)} epochs")
    

# %% 创建变量列表用于后续处理
# 根据实际文件数量创建列表
available_raws = []
available_epochs = []
available_evokeds = []
filenames = []

if len(fif_files) > 0:
    available_raws.append(raw1)
    available_epochs.append(epochs1)
    available_evokeds.append(evoked1)
    filenames.append(filename1)

if len(fif_files) > 1:
    available_raws.append(raw2)
    available_epochs.append(epochs2)
    available_evokeds.append(evoked2)
    filenames.append(filename2)

if len(fif_files) > 2:
    available_raws.append(raw3)
    available_epochs.append(epochs3)
    available_evokeds.append(evoked3)
    filenames.append(filename3)

if len(fif_files) > 3:
    available_raws.append(raw4)
    available_epochs.append(epochs4)
    available_evokeds.append(evoked4)
    filenames.append(filename4)

if len(fif_files) > 4:
    available_raws.append(raw5)
    available_epochs.append(epochs5)
    available_evokeds.append(evoked5)
    filenames.append(filename5)

if len(fif_files) > 5:
    available_raws.append(raw6)
    available_epochs.append(epochs6)
    available_evokeds.append(evoked6)
    filenames.append(filename6)

print(f"\n已加载 {len(available_evokeds)} 个数据集")
print("可用变量:")
for i in range(len(available_evokeds)):
    print(f"  raw{i+1}, epochs{i+1}, evoked{i+1} - {filenames[i]}")

# ------------------------------------------------------------------
# 可选：只处理/绘制特定被试（按1-based编号）。
# 例如只处理被试 2,3,5,6 -> selected_subjs = [2,3,5,6]
# 如果不需要子集处理，可以将 selected_subjs 设为 None 或空列表。
# ------------------------------------------------------------------
original_filenames = filenames.copy()
selected_subjs = [ 1, 3, 4, 5]  # 1-based indices of subjects to keep
if selected_subjs:
    # 转换为 0-based 索引并筛选出存在的索引
    selected_idx = [s - 1 for s in selected_subjs if (s - 1) < len(filenames)]
    if len(selected_idx) == 0:
        print("Warning: no selected subjects available; keeping all datasets.")
    else:
        available_raws = [available_raws[i] for i in selected_idx]
        available_epochs = [available_epochs[i] for i in selected_idx]
        available_evokeds = [available_evokeds[i] for i in selected_idx]
        filenames = [filenames[i] for i in selected_idx]
        print(f"Selected subjects (1-based): {selected_subjs}")
        print(f"Using datasets: {filenames}")


# %% 数据2：使用 evoked = epoch[:N].average()，在2x2子图中展示 N=50/100/150/200
if len(available_epochs) > 1:
    ep2 = available_epochs[1]
    total2 = len(ep2)
    targets = [50, 100, 150, 200]
    # 保留不超过总epoch数的目标
    valid = [n for n in targets if n <= total2]
    if len(valid) == 0:
        print(f"数据2 ({filenames[1]}) 的epoch数量为 {total2}，不足以绘制指定的分段平均。")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
        axes = axes.flatten()
        for i, n in enumerate(targets):
            ax = axes[i]
            if n <= total2:
                ev = ep2[:n].average()
                # 使用 MNE 的 evoked.plot 在指定轴上绘制
                ev.plot(axes=ax, spatial_colors=True, show=False, titles=None, time_unit='s')
                # 清理默认文本并添加自定义标题/标记
                for text in list(ax.texts):
                    text.remove()
                picks = mne.pick_types(ev.info, meg=True, exclude='bads')
                ax.set_title(f'{filenames[1]} - first {n} epochs\n({len(picks)} channels, {ev.nave} epochs)', fontsize=10)
                ax.axvline(0, color='red', linestyle='--', linewidth=1)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_visible(False)

        fig.suptitle(f"{filenames[1]} - Evoked for different N (total {total2})", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
        outname = f"{filenames[1]}_evoked_by_N_epochs_grid.png"
        outpath = os.path.join(output_dir, outname)
        fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"已保存: {outpath}")
        plt.show()

# %% 为 selected_subjs 中的（最多）4 个被试绘制 4x4 子图：每行一个被试，每列为不同 N (50,100,150,200)
targets = [50, 100, 150, 200]
# available_epochs / filenames 已在之前可能被筛选为 selected_subjs 的子集
n_subjects = min(4, len(available_epochs))
if n_subjects > 0:
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 14), facecolor='white')
    axes = axes.reshape((nrows, ncols))

    for i in range(nrows):
        if i < n_subjects:
            epochs_obj = available_epochs[i]
            subject_label = f'Subject{i+1}'
            total_n = len(epochs_obj)
            for j, n in enumerate(targets):
                ax = axes[i, j]
                if n <= total_n:
                    try:
                        ev = epochs_obj[:n].average()
                        ev.plot(axes=ax, spatial_colors=True, show=False, titles=None, time_unit='s')
                        for text in list(ax.texts):
                            text.remove()
                        ax.set_title(f'{subject_label} - N={n}\n(N={ev.nave} epochs)', fontsize=9)
                        ax.axvline(0, color='red', linestyle='--', linewidth=1)
                        ax.grid(True, alpha=0.3)
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center', transform=ax.transAxes)
                        ax.set_xticks([]); ax.set_yticks([])
                else:
                    # 若该被试 epoch 不足，隐藏该子图
                    ax.set_visible(False)
        else:
            # 整行没有数据，隐藏整行子图
            for j in range(ncols):
                axes[i, j].set_visible(False)

    fig.suptitle('Evoked plots of different subjects and different test counts', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    outname = os.path.join(output_dir, 'selected_subjs_evoked_grid_4x4.png')
    fig.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {outname}")
    plt.show()

# %% 绘制单个evoked响应对比图
n_plots = len(available_evokeds)

if n_plots > 0:
    # 动态选择网格：4 -> 2x2, 6 -> 2x3, else自适应
    if n_plots == 4:
        nrows, ncols = 2, 2
        figsize = (12, 10)
    elif n_plots == 6:
        nrows, ncols = 2, 3
        figsize = (18, 10)
    else:
        if n_plots <= 2:
            nrows, ncols = 1, n_plots
            figsize = (6*ncols, 5)
        elif n_plots <= 4:
            nrows, ncols = 2, 2
            figsize = (12, 10)
        else:
            ncols = 3
            nrows = (n_plots + ncols - 1) // ncols
            figsize = (18, 5*nrows)

    fig1, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                             facecolor='white', edgecolor='black')
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 绘制每个evoked响应
    for i, (evoked, filename) in enumerate(zip(available_evokeds, filenames)):
        ax = axes[i]
        
        # 使用MNE的evoked.plot方法
        evoked.plot(axes=ax, spatial_colors=True, show=False, titles=None, time_unit='s')
        
        # 清理默认文本
        for text in list(ax.texts):
            text.remove()
        
        # 添加自定义标题
        picks = mne.pick_types(evoked.info, meg=True, exclude='bads')
        ax.set_title(f'{filename}\n({len(picks)} channels, {evoked.nave} epochs)', 
                    fontsize=10, pad=10)
        
        # 添加刺激标记
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvspan(baseline[0], baseline[1], alpha=0.1, color='blue')
        ax.grid(True, alpha=0.3)
        
        # # 第一个子图添加图例
        # if i == 0:
        #     ax.legend(['MEG channels', 'Stimulus', 'Zero line', 'Baseline'], 
        #              loc='upper right', fontsize=8)
    
    # 隐藏多余子图
    for i in range(len(available_evokeds), len(axes)):
        axes[i].set_visible(False)
    
    # 添加总标题
    fig1.suptitle('MEG Evoked Responses Comparison (Individual Responses)', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
    outpath = os.path.join(output_dir, "evoked_comparison_individual.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {outpath}")
    plt.show()

# %%
