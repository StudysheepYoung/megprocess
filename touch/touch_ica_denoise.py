# %% 导入与配置
import matplotlib
matplotlib.use('Agg')

import os
import glob
import json
import numpy as np
import mne
from mne import find_events, Epochs
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "batch_preprocessing_results")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "ica_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENT_ID     = 1000000000
TMIN, TMAX   = -0.1, 0.4
BASELINE     = (-0.1, 0)
REJECT       = dict(mag=10e-12)
N_COMPONENTS = 15
ICA_METHOD   = 'infomax'
RANDOM_STATE = 42


# %% 辅助函数
def build_epochs_and_evoked(raw, tmin, tmax, baseline, reject):
    events = find_events(raw, stim_channel='Trigger', verbose=False)
    epochs = Epochs(raw, events, EVENT_ID, tmin=tmin, tmax=tmax,
                    baseline=baseline, detrend=1, reject=reject,
                    preload=True, verbose=False)
    return epochs, epochs.average()


def compute_snr_from_evoked(evoked, sig_tmin=0.0, sig_tmax=0.4,
                             noise_tmin=-0.1, noise_tmax=0.0):
    picks = mne.pick_types(evoked.info, meg=True, exclude='bads')
    data  = evoked.data[picks]
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


# %% 扫描文件列表
fif_files = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.fif")))
print(f"找到 {len(fif_files)} 个 .fif 文件")
for f in fif_files:
    print(f"  {f}")

summary = {}


# %% 批处理主循环
count = 0
for fif_path in fif_files:
    if count > 0:
        break
    count += 1
    subj_name = os.path.splitext(os.path.basename(fif_path))[0]
    print(f"\n{'='*60}")
    print(f"处理: {subj_name}")

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # 去噪前：带通 1.5–40 Hz → epoch → evoked → SNR
    raw_before = raw.copy()
    raw_before.filter(1.5, 40.0, picks='meg', verbose=False)
    _, evoked_before = build_epochs_and_evoked(raw_before, TMIN, TMAX, BASELINE, REJECT)
    snr_lin_before, snr_db_before, noise_rms_before = compute_snr_from_evoked(evoked_before)
    print(f"  SNR before : {snr_lin_before:.4f}  ({snr_db_before:.2f} dB)")

    # 拟合 ICA（1–40 Hz 副本）
    raw_ica_fit = raw.copy()
    raw_ica_fit.filter(1.0, 40.0, picks='meg', verbose=False)
    picks_mag = mne.pick_types(raw_ica_fit.info, meg=True, exclude='bads')
    n_comp = min(N_COMPONENTS, len(picks_mag) - 1)
    ica = ICA(n_components=n_comp, method=ICA_METHOD,
              fit_params=dict(extended=True), random_state=RANDOM_STATE, verbose=False)
    ica.fit(raw_ica_fit, picks=picks_mag, verbose=False)
    print(f"  ICA 拟合完成，成分数={n_comp}")

    # 自动识别伪影成分
    exclude = []
    try:
        ecg_idx, _ = ica.find_bads_ecg(raw_ica_fit, method='correlation', verbose=False)
        exclude += ecg_idx
    except Exception:
        pass
    try:
        eog_idx, _ = ica.find_bads_eog(raw_ica_fit, verbose=False)
        exclude += eog_idx
    except Exception:
        pass
    exclude = list(set(exclude))

    if len(exclude) == 0:
        sources   = ica.get_sources(raw_ica_fit).get_data()
        variances = np.var(sources, axis=1)
        z_scores  = (variances - variances.mean()) / variances.std()
        print(f"  各成分方差 z-score: {np.round(z_scores, 2)}")
        exclude = list(np.where(z_scores > 1.5)[0])
        if exclude:
            print(f"  [兜底] 方差 z-score 检测到 {len(exclude)} 个成分: {exclude}")

    ica.exclude = exclude
    print(f"  排除成分: {exclude}")

    # 应用 ICA → 去噪后 evoked → SNR
    raw_after = raw_before.copy()
    ica.apply(raw_after, verbose=False)
    _, evoked_after = build_epochs_and_evoked(raw_after, TMIN, TMAX, BASELINE, REJECT)
    snr_lin_after, snr_db_after, noise_rms_after = compute_snr_from_evoked(evoked_after)
    sf_linear, sf_db = compute_sf(noise_rms_before, noise_rms_after)

    print(f"  SNR before : {snr_lin_before:.4f}  ({snr_db_before:.2f} dB)")
    print(f"  SNR after  : {snr_lin_after:.4f}  ({snr_db_after:.2f} dB)")
    print(f"  Delta SNR  : {snr_db_after - snr_db_before:+.2f} dB")
    print(f"  SF         : {sf_linear:.4f}  ({sf_db:.2f} dB)")

    # 绘图：诱发响应对比
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
    fig.suptitle(f'触觉 MEG 诱发响应：ICA 去噪前后对比 — {subj_name}', fontsize=13)
    plt.tight_layout()
    evoked_path = os.path.join(OUTPUT_DIR, f'{subj_name}_evoked_before_after_ica.png')
    fig.savefig(evoked_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {evoked_path}")

    # 绘图：被排除的 ICA 成分
    if exclude:
        try:
            fig = ica.plot_components(picks=exclude, show=False)
            if not isinstance(fig, list):
                fig = [fig]
            comp_path = os.path.join(OUTPUT_DIR, f'{subj_name}_ica_excluded_components.png')
            fig[0].savefig(comp_path, dpi=150)
            for f in fig:
                plt.close(f)
            print(f"  已保存: {comp_path}")
        except Exception as e:
            print(f"  [警告] 无法保存 ICA 成分图: {e}")

    # 保存清洁数据
    out_fif = os.path.join(OUTPUT_DIR, f'{subj_name}_clean-raw.fif')
    raw_after.save(out_fif, overwrite=True, verbose=False)
    print(f"  已保存: {out_fif}")

    summary[subj_name] = {
        "snr_before_linear": float(snr_lin_before) if not np.isnan(snr_lin_before) else None,
        "snr_before_db":     float(snr_db_before)  if not np.isnan(snr_db_before)  else None,
        "snr_after_linear":  float(snr_lin_after)  if not np.isnan(snr_lin_after)  else None,
        "snr_after_db":      float(snr_db_after)   if not np.isnan(snr_db_after)   else None,
        "delta_snr_db":      float(snr_db_after - snr_db_before),
        "sf_linear":         float(sf_linear)      if not np.isnan(sf_linear)      else None,
        "sf_db":             float(sf_db)           if not np.isnan(sf_db)          else None,
        "n_ica_excluded":    len(exclude),
        "excluded_components": [int(i) for i in exclude],
    }


# %% 汇总输出
print(f"\n{'='*60}")
print(f"{'Subject':<30} {'SNR Before':>12} {'SNR After':>12} {'Delta':>8} {'SF':>8}")
print('-' * 72)
for name, v in summary.items():
    sb = f"{v['snr_before_db']:.2f} dB" if v['snr_before_db'] is not None else "  N/A"
    sa = f"{v['snr_after_db']:.2f} dB"  if v['snr_after_db']  is not None else "  N/A"
    dd = f"{v['delta_snr_db']:+.2f} dB"
    sf = f"{v['sf_db']:.2f} dB"         if v['sf_db']         is not None else "  N/A"
    print(f"{name:<30} {sb:>12} {sa:>12} {dd:>8} {sf:>8}")

summary_path = os.path.join(OUTPUT_DIR, "snr_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n已保存汇总: {summary_path}")
