#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Time-frequency decomposition of the epoched signal
"""

from preprocessing.config import *








# batch_data = data[10:20, :, :]
# tfr = mne.time_frequency.tfr_array_multitaper(batch_data, SAMPLING_RATE, freqs=freqlist, n_cycles=cycles, time_bandwidth=3.0, use_fft=True, decim=8, n_jobs=-1, output='complex', verbose=False)

# power = np.abs(tfr) ** 2
# power_avg = np.mean(power, axis=2) 
# power_db = 10 * np.log10(power_avg)

# baseline = np.mean(power_db, axis=-1, keepdims=True)
# power_baseline = power_db - baseline
# lim = np.max((np.abs(power_baseline.min()), np.abs(power_baseline.max())))

# for i in range(10) :
#     lim = np.max((np.abs(power_baseline[i, 4, :, :].min()), np.abs(power_baseline[i, 4, :, :].max())))
#     plt.imshow(power_baseline[i, 8, :, :], aspect='auto', origin='lower', interpolation='nearest', cmap='jet', vmin=-lim, vmax=lim)
#     plt.colorbar()
#     plt.show()


# phase = np.angle(tfr)
# phase_avg_tapers = np.angle(np.mean(np.exp(1j * phase), axis=2))  # Moyenne complexe des phases
# itpc = np.abs(np.mean(np.exp(1j * phase_avg_tapers), axis=0))  # Inter-Trial Phase Coherence


# # 2. Visualiser l'ITPC pour un canal et une fréquence spécifiques
# channel_idx = 0  # Premier canal
# freq_idx = 30    # 21ème fréquence (vérifiez sa valeur avec freqlist[freq_idx])


# plt.figure(figsize=(10, 5))
# plt.plot(epochs.times[::8], np.mean(itpc[channel_idx, freq_idx, :], axis=0))
# plt.axvline(x=0, color='r', linestyle='--')  # Marquer le temps zéro
# plt.title(f'ITPC - Canal {clean_electrode["name"].values[channel_idx]}, Freq: {freqlist[freq_idx]:.1f} Hz')
# plt.xlabel('Temps (s)')
# plt.ylabel('ITPC')
# plt.grid(True)
# plt.show()


# # 3. Calculer la différence de phase entre deux canaux (PLV - Phase Locking Value)
# channel1 = 0
# channel2 = 1
# phase_diff = phase_avg_tapers[:, channel1, :, :] - phase_avg_tapers[:, channel2, :, :]
# # Normaliser sur le cercle trigonométrique
# phase_diff = np.angle(np.exp(1j * phase_diff))
# # Calculer la PLV (moyenner la différence de phase à travers les essais)
# plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
# # plv est maintenant de dimensions (n_freqs, n_times)
# plt.imshow(plv, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
# plt.colorbar(label='Phase Locking Value')
# plt.show()
# # 4. Calculer la moyenne de phase ou la direction préférée
# mean_phase = np.angle(np.mean(np.exp(1j * phase_avg_tapers), axis=0))
# # mean_phase est de dimensions (n_channels, n_freqs, n_times)
# plt.figure(figsize=(10, 5))
# for i in range(10) :
#     plt.subplot(2, 5, i+1)
#     plt.imshow(mean_phase[i, :, :], aspect='auto', origin='lower', interpolation='nearest', cmap='jet')
#     plt.title(f'Canal {i}')
#     plt.colorbar(label='Phase (radians)')
# plt.tight_layout()
# plt.show()

# # 5. Analyser le reset de phase après un événement
# # Sélectionner une bande de fréquence (par exemple, alpha)
# theta_idx = np.where((freqlist >= 4) & (freqlist <= 8))[0]
# # Calculer l'ITPC moyen dans la bande alpha
# alpha_itpc = np.mean(itpc[:, theta_idx, :], axis=1)

# # 6. Visualiser l'ITPC dans une carte temps-fréquence
# plt.figure(figsize=(12, 6))
# plt.imshow(itpc[channel_idx], aspect='auto', origin='lower', 
#           extent=[epochs.times[0], epochs.times[-1], freqlist[0], freqlist[-1]],
#           cmap='viridis')
# plt.colorbar(label='ITPC')
# plt.axvline(x=0, color='r', linestyle='--')  # Marquer le temps zéro
# plt.title(f'ITPC - Canal {clean_electrode["name"].values[channel_idx]}')
# plt.xlabel('Temps (s)')
# plt.ylabel('Fréquence (Hz)')
# plt.ylim(2, 80)  # Limiter la plage de fréquences affichées
# plt.yscale('log')  # Échelle logarithmique pour les fréquences
# plt.show()

# # 7. Extraire la phase à un moment spécifique (par exemple, au stimulus)
# # Trouver l'index du temps 0
# zero_idx = np.where(np.abs(epochs.times[::8]) == np.min(np.abs(epochs.times[::8])))[0][0]
# # Phase au moment du stimulus pour tous les canaux et fréquences
# stim_phase = phase[:, :, :, zero_idx]

# phase_diff_plv = phase[:, channel1, :, :] - phase[:, channel2, :, :]
# pli = np.abs(np.mean(np.sign(np.sin(phase_diff_plv)), axis=0))


# plt.imshow(phase_avg_tapers[0, 0, :, :], aspect='auto', origin='lower', interpolation='nearest', cmap='jet')
# plt.show()


# # Moyenne sur tous les essais et conditions pour un canal spécifique
# channel_idx = 0  # Premier canal
# power_tf_map = np.mean(np.mean(power_baseline[:, channel_idx, :, :, :], axis=0), axis=0)  # Forme: (80, 3073)

# # Visualisation améliorée
# plt.figure(figsize=(12, 6))
# plt.imshow(power_tf_map, aspect='auto', origin='lower', 
#            extent=[epochs.times[0], epochs.times[-1], freqlist[0], freqlist[-1]],
#            cmap='jet')
# plt.colorbar(label='Power (dB)')
# plt.axvline(x=0, color='white', linestyle='--')  # Marquer le temps zéro
# plt.title(f'TF Map - Canal {clean_electrode["name"].values[channel_idx]}')
# plt.xlabel('Temps (s)')
# plt.ylabel('Fréquence (Hz)')
# plt.ylim(2, 80)  # Limiter la plage de fréquences affichées
# plt.yscale('log')  # Échelle logarithmique pour les fréquences
# plt.tight_layout()
# plt.show()



# high_gamma_idx = np.where((freqlist >= 70) & (freqlist <= 150))[0]
# # Moyenne sur tous les essais et canaux pour la bande alpha
# alpha_power = np.mean(np.mean(np.mean(power_baseline[:, :, :, high_gamma_idx, :], axis=0), axis=0), axis=1)  # Forme: (3073,)

# plt.figure(figsize=(10, 5))
# plt.plot(epochs.times[::8], alpha_power[0])
# plt.axvline(x=0, color='red', linestyle='--')  # Marquer le temps zéro
# plt.title('Puissance moyenne dans la bande alpha')
# plt.xlabel('Temps (s)')
# plt.ylabel('Puissance (dB)')
# plt.grid(True)
# plt.show()