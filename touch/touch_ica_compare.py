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
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "ica_compare_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TMIN, TMAX   = -0.1, 0.6
BASELINE     = (-0.1, 0)
REJECT       = dict(mag=10e-12)
EVENT_ID     = 1000000000
N_COMPONENTS = 15
RANDOM_STATE = 42

METHODS = ['fastica', 'infomax', 'picard', 'jade', 'sobi', 'amuse']


# %% 辅助函数
def build_epochs_and_evoked(raw, tmin, tmax, baseline, reject):
    events = find_events(raw, stim_channel='Trigger', verbose=False)
    epochs = Epochs(raw, events, EVENT_ID, tmin=tmin, tmax=tmax,
                    baseline=baseline, detrend=1, reject=reject,
                    preload=True, verbose=False)
    return epochs, epochs.average()


def compute_snr_from_evoked(evoked, sig_tmin=0.0, sig_tmax=0.1, noise_tmin=-0.1, noise_tmax=0.0):
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


def amuse_numpy(X, n_comp, lag=1):
    from scipy.linalg import eigh
    _, T = X.shape
    cov = X @ X.T / T
    vals, vecs = eigh(cov)
    idx = np.argsort(vals)[::-1][:n_comp]
    D = np.diag(1.0 / np.sqrt(vals[idx]))
    vecs_sub = vecs[:, idx]
    W_white = D @ vecs_sub.T
    Z = W_white @ X

    R_lag = Z[:, lag:] @ Z[:, :-lag].T / (T - lag)
    R_sym = (R_lag + R_lag.T) / 2
    _, V_mat = eigh(R_sym)
    order = np.argsort(np.abs(np.linalg.eigvalsh(R_sym)))[::-1]
    V = V_mat[:, order]

    S = V.T @ Z
    return V, vecs_sub, D, S


def build_ica_from_amuse(raw_fit, picks, n_comp, lag=1):
    data = raw_fit.get_data(picks=picks)
    V, vecs_sub, D, _ = amuse_numpy(data, n_comp, lag)
    ica = ICA(n_components=n_comp, method='fastica', random_state=42, verbose=False)
    ica.fit(raw_fit, picks=picks, verbose=False)
    D_inv = np.diag(1.0 / np.diag(D))
    ica.pca_components_  = vecs_sub.T
    ica.unmixing_matrix_ = V.T @ D
    ica.mixing_matrix_   = D_inv @ V
    ica.pca_mean_        = data.mean(axis=1)
    ica.pre_whitener_    = np.ones((len(picks), 1))
    return ica


def sobi_numpy(X, n_comp, n_lags=100):
    from scipy.linalg import eigh
    _, T = X.shape
    cov = X @ X.T / T
    vals, vecs = eigh(cov)
    idx = np.argsort(vals)[::-1][:n_comp]
    D = np.diag(1.0 / np.sqrt(vals[idx]))
    vecs_sub = vecs[:, idx]
    W_white = D @ vecs_sub.T
    Z = W_white @ X

    lags = np.arange(1, n_lags + 1)
    Rs = []
    for lag in lags:
        R = Z[:, lag:] @ Z[:, :-lag].T / (T - lag)
        Rs.append((R + R.T) / 2)

    V = np.eye(n_comp)
    for _ in range(200):
        for p in range(n_comp - 1):
            for q in range(p + 1, n_comp):
                num = sum(2 * R[p, q] * (R[p, p] - R[q, q]) for R in Rs)
                den = sum((R[p, p] - R[q, q])**2 - 4 * R[p, q]**2 for R in Rs)
                theta = 0.5 * np.arctan2(num, den + 1e-12)
                c, s = np.cos(theta), np.sin(theta)
                G = np.eye(n_comp)
                G[p, p] = c; G[q, q] = c
                G[p, q] = s; G[q, p] = -s
                V = V @ G
                Rs = [G.T @ R @ G for R in Rs]

    S = V.T @ Z
    return V, vecs_sub, D, S


def build_ica_from_sobi(raw_fit, picks, n_comp, n_lags=100):
    data = raw_fit.get_data(picks=picks)
    V, vecs_sub, D, _ = sobi_numpy(data, n_comp, n_lags)
    ica = ICA(n_components=n_comp, method='fastica', random_state=42, verbose=False)
    ica.fit(raw_fit, picks=picks, verbose=False)
    D_inv = np.diag(1.0 / np.diag(D))
    ica.pca_components_  = vecs_sub.T
    ica.unmixing_matrix_ = V.T @ D
    ica.mixing_matrix_   = D_inv @ V
    ica.pca_mean_        = data.mean(axis=1)
    ica.pre_whitener_    = np.ones((len(picks), 1))
    return ica


def jade_numpy(X, n_comp):
    from scipy.linalg import eigh
    _, T = X.shape
    cov = X @ X.T / T
    vals, vecs = eigh(cov)
    idx = np.argsort(vals)[::-1][:n_comp]
    D = np.diag(1.0 / np.sqrt(vals[idx]))
    vecs_sub = vecs[:, idx]
    W_white = D @ vecs_sub.T
    Z = W_white @ X

    CM = np.zeros((n_comp, n_comp * n_comp))
    for p in range(n_comp):
        for q in range(n_comp):
            zp, zq = Z[p], Z[q]
            cum = (zp * zq * Z) @ Z.T / T - np.outer(zp @ zq.T / T * np.ones(n_comp), np.ones(n_comp))
            cum -= np.eye(n_comp) * (zp @ zq.T / T)
            CM[:, p * n_comp + q] = cum[:, p]

    V = np.eye(n_comp)
    for _ in range(100):
        for p in range(n_comp - 1):
            for q in range(p + 1, n_comp):
                g = np.array([CM[p, p * n_comp + p] - CM[q, q * n_comp + q],
                               CM[p, p * n_comp + q] + CM[q, q * n_comp + p]])
                ton = g[0]; toff = g[1]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton**2 + toff**2))
                c, s = np.cos(theta), np.sin(theta)
                G = np.eye(n_comp)
                G[p, p] = c; G[q, q] = c
                G[p, q] = s; G[q, p] = -s
                V = V @ G
                CM = G.T @ CM.reshape(n_comp, n_comp, n_comp).transpose(1, 0, 2).reshape(n_comp, n_comp * n_comp)
                CM = (G.T @ CM.reshape(n_comp, n_comp, n_comp)).reshape(n_comp, n_comp * n_comp)

    W = V.T @ W_white
    S = W @ X
    return V, vecs_sub, D, S


def build_ica_from_jade(raw_fit, picks, n_comp):
    data = raw_fit.get_data(picks=picks)
    V, vecs_sub, D, _ = jade_numpy(data, n_comp)
    ica = ICA(n_components=n_comp, method='fastica', random_state=42, verbose=False)
    ica.fit(raw_fit, picks=picks, verbose=False)
    D_inv = np.diag(np.sqrt(np.diag(np.linalg.inv(D @ D))))
    ica.pca_components_  = vecs_sub.T
    ica.unmixing_matrix_ = V.T @ D
    ica.mixing_matrix_   = D_inv @ V
    ica.pca_mean_        = data.mean(axis=1)
    ica.pre_whitener_    = np.ones((len(picks), 1))
    return ica


# %% 读取数据（只处理第一个 FIF 文件）
fif_files = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.fif")))
print(f"找到 {len(fif_files)} 个 .fif 文件")
if not fif_files:
    raise FileNotFoundError(f"未找到 FIF 文件: {DATA_DIR}")

fif_path = fif_files[0]
subj_name = os.path.splitext(os.path.basename(fif_path))[0]
print(f"使用: {fif_path}")

raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

# 去噪前基线：FIF 已预处理，直接 copy，不再做 1.5–40 Hz 滤波
raw_before = raw.copy()
_, evoked_before = build_epochs_and_evoked(raw_before, TMIN, TMAX, BASELINE, REJECT)
snr_lin_before, snr_db_before, noise_rms_before = compute_snr_from_evoked(evoked_before)
print(f"SNR before: {snr_lin_before:.4f} ({snr_db_before:.2f} dB)\n")

picks_mag = mne.pick_types(raw_before.info, meg=True, exclude='bads')
n_comp = min(N_COMPONENTS, len(picks_mag) - 1)


# %% 逐方法运行 ICA，收集指标和 evoked
results = {}
evokeds = {'before': evoked_before}

for method in METHODS:
    print(f"=== {method.upper()} ===")
    if method == 'jade':
        try:
            ica = build_ica_from_jade(raw_before, picks_mag, n_comp)
        except Exception as e:
            print(f"  拟合失败: {e}")
            continue
    elif method == 'sobi':
        try:
            ica = build_ica_from_sobi(raw_before, picks_mag, n_comp)
        except Exception as e:
            print(f"  拟合失败: {e}")
            continue
    elif method == 'amuse':
        try:
            ica = build_ica_from_amuse(raw_before, picks_mag, n_comp)
        except Exception as e:
            print(f"  拟合失败: {e}")
            continue
    else:
        fit_params = dict(extended=True) if method == 'infomax' else {}
        ica = ICA(n_components=n_comp, method=method,
                  fit_params=fit_params, random_state=RANDOM_STATE, verbose=False)
        try:
            ica.fit(raw_before, picks=picks_mag, verbose=False)
        except Exception as e:
            print(f"  拟合失败: {e}")
            continue

    # 手动选择排除成分：弹出交互界面，关闭窗口后继续
    matplotlib.use('TkAgg')
    ica.plot_components(picks=range(n_comp), show=True)
    plt.show(block=True)
    ica.plot_sources(raw_before, show=True, block=True)
    exclude = ica.exclude[:]
    matplotlib.use('Agg')
    print(f"  排除成分: {exclude}")

    raw_after = raw_before.copy()
    ica.apply(raw_after, verbose=False)
    _, evoked_after = build_epochs_and_evoked(raw_after, TMIN, TMAX, BASELINE, REJECT)
    snr_lin, snr_db, noise_rms_after = compute_snr_from_evoked(evoked_after)
    picks_ev = mne.pick_types(evoked_before.info, meg=True, exclude='bads')
    data_before = evoked_before.data[picks_ev]
    data_after  = evoked_after.data[picks_ev]
    removed = data_before - data_after
    sf_linear = np.mean(removed**2) / np.mean(data_before**2) if np.mean(data_before**2) > 0 else np.nan
    sf_db = 10 * np.log10(sf_linear) if sf_linear > 0 else np.nan

    print(f"  SNR after : {snr_lin:.4f} ({snr_db:.2f} dB)  |  Delta SNR: {snr_db - snr_db_before:+.2f} dB  |  SF: {sf_linear:.4f} ({sf_db:.2f} dB)")
    results[method] = dict(snr_lin=snr_lin, snr_db=snr_db,
                           delta_snr_db=snr_db - snr_db_before,
                           sf_linear=sf_linear, sf_db=sf_db,
                           n_excluded=len(exclude))
    evokeds[method] = evoked_after


# %% 保存指标 JSON
results['before'] = dict(snr_lin=snr_lin_before, snr_db=snr_db_before)
with open(os.path.join(OUTPUT_DIR, 'ica_compare_metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\n已保存指标 JSON")


# %% 对比图：各方法 evoked
n_plots = 1 + len([m for m in METHODS if m in evokeds])
fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), facecolor='white')
plot_items = [('before', 'ICA 前')] + [(m, m.upper()) for m in METHODS if m in evokeds]

for ax, (key, title) in zip(axes, plot_items):
    evokeds[key].plot(axes=ax, spatial_colors=True, show=False, titles=None, time_unit='s')
    for text in list(ax.texts):
        text.remove()
    if key in results and key != 'before':
        r = results[key]
        subtitle = f"SNR {r['snr_db']:.1f} dB  ΔSNR {r['delta_snr_db']:+.1f} dB\nSF {r['sf_db']:.1f} dB  排除 {r['n_excluded']} 个"
    else:
        subtitle = f"SNR {snr_db_before:.1f} dB"
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.axvspan(BASELINE[0], BASELINE[1], alpha=0.1, color='blue')
    ax.grid(True, alpha=0.3)

fig.suptitle('触觉 MEG — ICA 方法对比', fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'ica_compare_evoked.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"已保存: {out_path}")


# %% 柱状图：SNR 和 SF 对比
methods_done = [m for m in METHODS if m in results]
x = np.arange(len(methods_done))
snr_vals    = [results[m]['snr_db'] for m in methods_done]
delta_vals  = [results[m]['delta_snr_db'] for m in methods_done]
sf_vals     = [results[m]['sf_db'] for m in methods_done]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='white')
for ax, vals, ylabel, title in zip(
        axes,
        [snr_vals, delta_vals, sf_vals],
        ['SNR (dB)', 'ΔSNR (dB)', 'SF (dB)'],
        ['去噪后 SNR', 'SNR 提升量', '噪声抑制因子 SF']):
    bars = ax.bar(x, vals, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'][:len(methods_done)])
    ax.axhline(0, color='gray', linewidth=0.8)
    if title == '去噪后 SNR':
        ax.axhline(snr_db_before, color='red', linestyle='--', linewidth=1, label=f'去噪前 {snr_db_before:.1f} dB')
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods_done])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'ica_compare_metrics.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"已保存: {out_path}")
