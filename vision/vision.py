# %% 导入所需模块
import mne
import scipy.io as scio
import numpy as np

def read_meg_data(data_path,sensor_path):
    fs = 1000  # 采样率1000Hz
    n_record_chans = 66 # 记录通道数
    file_id = open(data_path, "rb")  # 二进制文件
    baseDate_data_0 = np.fromfile(file_id, dtype=np.float32)  # 读取二进制文件
    baseDate_data = baseDate_data_0[512:]
    General_Time_In_Seconds = len(baseDate_data) // n_record_chans // fs  # 读取数据的轮数
    Single_Sensor_Data_Length = General_Time_In_Seconds * fs  # 单探头的数据量
    file_id.close()  # 关闭文件
    read_raw_data = np.zeros((n_record_chans, Single_Sensor_Data_Length))  # 创建一个空数组，用于存放拼接后的数据
    for channel_index in range(n_record_chans):  # 遍历探头（通道数）
        for time_seconds in range(General_Time_In_Seconds):  # 遍历记录轮数(每秒的数据）
            # 将baseDate_data中对应位置的数据赋值给All_Channel_Data中对应位置
            read_raw_data[channel_index, time_seconds * fs:(time_seconds + 1) * fs] = baseDate_data[channel_index * fs + (time_seconds * n_record_chans * fs):(channel_index + 1) * fs + (time_seconds * n_record_chans * fs)]

    use_chans = 65
    raw_data = read_raw_data[:use_chans, :]
    raw_data[:-1, :] = raw_data[:-1, :] * 1e-12

    num_chans_data = 64
    sensor_info = scio.loadmat(sensor_path)

    label = list(sensor_info['ch_names'])
    label = [lab.strip() for lab in label]
    pos = sensor_info['pos']
    ori = sensor_info['ori']

    sfreq = 1000
    num_chan = 64
    raw_info = mne.create_info(
    ch_names = label + ['Trigger'],
    ch_types = ['eeg' for i in range(num_chan)] + ['stim'],sfreq=sfreq)

    raw = mne.io.RawArray(raw_data, raw_info)

    dic = {label[i]: pos[i,:] for i in range(num_chans_data)}
    montage = mne.channels.make_dig_montage(ch_pos=dic, coord_frame='head')
    raw = raw.set_montage(montage)

    for j,ch_name in enumerate(raw.info['ch_names']):
        if ch_name != 'Trigger':
            raw.info['chs'][j]['kind'] = mne.io.constants.FIFF.FIFFV_MEG_CH # 通道类型
            raw.info['chs'][j]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_T # 单位tesla
            raw.info['chs'][j]['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2 # Qusqpin类型
            raw.info['chs'][j]['loc'][3:12] = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.]) # 旋转矩阵
            Z_orient =  mne._fiff.tag._loc_to_coil_trans(raw.info['chs'][j]['loc'])[:3, :3]
            find_Rotation = mne.transforms._find_vector_rotation(Z_orient[:, 2], ori[j, :])
            raw.info['chs'][j]['loc'][3:12] = np.dot(find_Rotation, Z_orient).T.ravel()

    return raw,read_raw_data,raw_data

##
import os
import matplotlib.pyplot as plt
from mne import find_events, Epochs

data_path = r"/Users/luckyyoung/Desktop/data/vision/20241214 104044.basedata"
sensor_path = r'/Users/luckyyoung/Desktop/data/vision/sensors_mecg64.mat'
raw,read_raw_data,raw_data = read_meg_data(data_path,sensor_path)
raw.filter(l_freq=1.5, h_freq=40, method='iir', verbose=True)

# %% 参数
tmin, tmax = -0.1, 0.6
baseline = (-0.1, 0)
reject_criteria = dict(mag=3e-9)
output_dir = "conference"
os.makedirs(output_dir, exist_ok=True)

# %% 查看事件，自动获取 event_id
events = find_events(raw, stim_channel='Trigger', verbose=True)
print(f"事件总数: {len(events)}，事件类型: {np.unique(events[:, 2])}")
event_id = int(np.unique(events[:, 2])[0])

# %% 切 epoch & evoked
epochs = Epochs(raw, events, event_id,
                tmin=tmin, tmax=tmax,
                baseline=baseline, detrend=1,
                reject=reject_criteria,
                preload=True, verbose=False)
print(f"有效 epoch 数: {len(epochs)}")
evoked = epochs.average()
picks  = mne.pick_types(evoked.info, meg=True, exclude='bads')

# %% 总 evoked 图
fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
evoked.plot(axes=ax, spatial_colors=True, show=False, titles=None, time_unit='s')
for text in list(ax.texts):
    text.remove()
ax.set_title(f"Hearing Evoked  ({len(picks)} ch, {evoked.nave} epochs)", fontsize=12)
ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.axvspan(baseline[0], baseline[1], alpha=0.1, color='blue')
ax.grid(True, alpha=0.3)
plt.tight_layout()
outpath = os.path.join(output_dir, "hearing_evoked.png")
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f"已保存: {outpath}")
plt.show()

# %% 不同叠加数 N 的 2x2 对比图
targets = [50, 100, 150, 200]
total_n = len(epochs)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
axes = axes.flatten()
for i, n in enumerate(targets):
    ax = axes[i]
    if n <= total_n:
        ev = epochs[:n].average()
        ev.plot(axes=ax, spatial_colors=True, show=False, titles=None, time_unit='s')
        for text in list(ax.texts):
            text.remove()
        ax.set_title(f"First {n} epochs  (N={ev.nave})", fontsize=10)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f"N={n}\n(共 {total_n} 个 epoch)",
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Hearing Evoked — 不同叠加数对比", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
outpath = os.path.join(output_dir, "hearing_evoked_by_N.png")
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f"已保存: {outpath}")
plt.show()

# %% SNR
data_ev = evoked.data[picks]
times   = evoked.times
def rms(mask):
    return np.mean(np.sqrt(np.mean(data_ev[:, mask] ** 2, axis=1)))
task_snr = rms((times >= 0) & (times <= tmax)) / rms((times >= baseline[0]) & (times < 0))
print(f"SNR (task / baseline) = {task_snr:.4f}")
