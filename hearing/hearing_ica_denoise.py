# %% 导入与配置
import matplotlib
matplotlib.use('TkAgg')

import os
import numpy as np
import mne
import scipy.io as scio
from mne import find_events, Epochs
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "20241214 105123.basedata")
SENSOR_PATH  = os.path.join(SCRIPT_DIR, "sensors_mecg64.mat")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "ica_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TMIN, TMAX   = -0.1, 0.4
BASELINE     = (-0.1, 0)
REJECT       = dict(mag=3e-9)
N_COMPONENTS = 15
ICA_METHOD   = 'infomax'
RANDOM_STATE = 42


# %% 辅助函数
def read_meg_data(data_path, sensor_path):
    fs = 1000
    n_record_chans = 66
    file_id = open(data_path, "rb")
    baseDate_data_0 = np.fromfile(file_id, dtype=np.float32)
    baseDate_data = baseDate_data_0[512:]
    General_Time_In_Seconds = len(baseDate_data) // n_record_chans // fs
    Single_Sensor_Data_Length = General_Time_In_Seconds * fs
    file_id.close()
    read_raw_data = np.zeros((n_record_chans, Single_Sensor_Data_Length))
    for channel_index in range(n_record_chans):
        for time_seconds in range(General_Time_In_Seconds):
            read_raw_data[channel_index, time_seconds * fs:(time_seconds + 1) * fs] = \
                baseDate_data[channel_index * fs + (time_seconds * n_record_chans * fs):
                              (channel_index + 1) * fs + (time_seconds * n_record_chans * fs)]

    use_chans = 65
    raw_data = read_raw_data[:use_chans, :]
    raw_data[:-1, :] = raw_data[:-1, :] * 1e-12

    num_chans_data = 64
    sensor_info = scio.loadmat(sensor_path)
    label = list(sensor_info['ch_names'])
    label = [lab.strip() for lab in label]
    pos = sensor_info['pos']
    ori = sensor_info['ori']

    raw_info = mne.create_info(
        ch_names=label + ['Trigger'],
        ch_types=['eeg' for _ in range(64)] + ['stim'],
        sfreq=1000)

    raw = mne.io.RawArray(raw_data, raw_info)
    dic = {label[i]: pos[i, :] for i in range(num_chans_data)}
    montage = mne.channels.make_dig_montage(ch_pos=dic, coord_frame='head')
    raw = raw.set_montage(montage)

    for j, ch_name in enumerate(raw.info['ch_names']):
        if ch_name != 'Trigger':
            raw.info['chs'][j]['kind'] = mne.io.constants.FIFF.FIFFV_MEG_CH
            raw.info['chs'][j]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_T
            raw.info['chs'][j]['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2
            raw.info['chs'][j]['loc'][3:12] = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
            Z_orient = mne._fiff.tag._loc_to_coil_trans(raw.info['chs'][j]['loc'])[:3, :3]
            find_Rotation = mne.transforms._find_vector_rotation(Z_orient[:, 2], ori[j, :])
            raw.info['chs'][j]['loc'][3:12] = np.dot(find_Rotation, Z_orient).T.ravel()

    return raw, read_raw_data, raw_data


def build_epochs_and_evoked(raw, tmin, tmax, baseline, reject):
    events = find_events(raw, stim_channel='Trigger', verbose=False)
    events[:, 0] += 400  # trigger 比实际听觉刺激早 0.4s，平移对齐
    event_id = int(np.unique(events[:, 2])[0])
    epochs = Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                    baseline=baseline, detrend=1, reject=reject,
                    preload=True, verbose=False)
    return epochs, epochs.average()


def compute_snr_from_evoked(evoked, sig_tmin=0.0, sig_tmax=0.4, noise_tmin=-0.1, noise_tmax=0.0):
    picks = mne.pick_types(evoked.info, meg=True, exclude='bads')
    data = evoked.data[picks]
    times = evoked.times

    def rms(mask):
        return np.mean(np.sqrt(np.mean(data[:, mask] ** 2, axis=1)))

    rms_signal = rms((times >= sig_tmin) & (times <= sig_tmax))
    rms_noise  = rms((times >= noise_tmin) & (times < noise_tmax))
    snr_linear = rms_signal / rms_noise if rms_noise > 0 else np.nan
    snr_db     = 20 * np.log10(snr_linear) if snr_linear > 0 else np.nan
    return snr_linear, snr_db, rms_noise


def compute_sf(noise_rms_before, noise_rms_after):
    sf_linear = noise_rms_before / noise_rms_after if noise_rms_after > 0 else np.nan
    sf_db     = 20 * np.log10(sf_linear) if sf_linear > 0 else np.nan
    return sf_linear, sf_db


# %% 读取原始数据
print("读取原始数据...")
raw, _, _ = read_meg_data(DATA_PATH, SENSOR_PATH)
print(raw)


# %% 去噪前：带通 1.5–40 Hz → epoch → evoked → SNR
raw_before = raw.copy()
raw_before.filter(1.5, 40.0, picks='meg', verbose=False)

_, evoked_before = build_epochs_and_evoked(raw_before, TMIN, TMAX, BASELINE, REJECT)

snr_lin_before, snr_db_before, noise_rms_before = compute_snr_from_evoked(evoked_before)
print(f"SNR before : {snr_lin_before:.4f}  ({snr_db_before:.2f} dB)")


# %% 拟合 ICA（1–100 Hz，ICLabel 要求）
raw_ica_fit = raw.copy()
raw_ica_fit.filter(1.0, 100.0, picks='meg', verbose=False)

picks_mag = mne.pick_types(raw_ica_fit.info, meg=True, exclude='bads')
n_comp = min(N_COMPONENTS, len(picks_mag) - 1)
ica = ICA(n_components=n_comp, method=ICA_METHOD,
          fit_params=dict(extended=True), random_state=RANDOM_STATE, verbose=False)
ica.fit(raw_ica_fit, picks=picks_mag, verbose=False)
print(f"ICA 拟合完成，成分数={n_comp}")


# %% 手动选择 ICA 成分
print("请在弹出的窗口中点击要排除的成分，关闭窗口后继续...")
ica.plot_components()        # 地形图，点击标记
ica.plot_sources(raw_ica_fit)  # 时间序列，点击标记
plt.show(block=True)

exclude = list(ica.exclude)
print(f"已选择排除成分 ({len(exclude)} 个): {exclude}")


# %% 应用 ICA → 去噪后 evoked → SNR
raw_after = raw_before.copy()
ica.apply(raw_after, verbose=False)

_, evoked_after = build_epochs_and_evoked(raw_after, TMIN, TMAX, BASELINE, REJECT)
snr_lin_after, snr_db_after, noise_rms_after = compute_snr_from_evoked(evoked_after)

sf_linear, sf_db = compute_sf(noise_rms_before, noise_rms_after)

print(f"SNR before : {snr_lin_before:.4f}  ({snr_db_before:.2f} dB)")
print(f"SNR after  : {snr_lin_after:.4f}  ({snr_db_after:.2f} dB)")
print(f"Delta SNR  : {snr_db_after - snr_db_before:+.2f} dB")
print(f"SF         : {sf_linear:.4f}  ({sf_db:.2f} dB)")


# %% 绘图：诱发响应对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, evoked, title in zip(axes,
                              [evoked_before, evoked_after],
                              ['ICA 前', 'ICA 后']):
    evoked.plot(axes=ax, spatial_colors=True, show=False, titles=None, time_unit='s')
    for text in list(ax.texts):
        text.remove()
    ax.set_title(title, fontsize=12)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvspan(BASELINE[0], BASELINE[1], alpha=0.1, color='blue')
    ax.grid(True, alpha=0.3)

fig.suptitle('听觉 MEG 诱发响应：ICA 去噪前后对比', fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'hearing_evoked_before_after_ica.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"已保存: {out_path}")


# %% 绘图：被排除的 ICA 成分
if exclude:
    try:
        fig = ica.plot_components(picks=exclude, show=False)
        if not isinstance(fig, list):
            fig = [fig]
        out_path = os.path.join(OUTPUT_DIR, 'hearing_ica_excluded_components.png')
        fig[0].savefig(out_path, dpi=150)
        for f in fig:
            plt.close(f)
        print(f"已保存: {out_path}")
    except Exception as e:
        print(f"[警告] 无法保存 ICA 成分图: {e}")


# %% 保存清洁数据
out_fif = os.path.join(OUTPUT_DIR, 'hearing_ica_clean-raw.fif')
raw_after.save(out_fif, overwrite=True, verbose=False)
print(f"已保存: {out_fif}")
