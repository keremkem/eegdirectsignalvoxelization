import numpy as np
from scipy.io import savemat

def create_simple_dipole_simulation(coord_file, output_file='test_dipole_sim.mat'):
    """Basit ama kontrollü dipol simülasyonu"""
    
    # Koordinatları oku
    coords = {}
    with open(coord_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                name, x, y, z = parts
                coords[name] = np.array([float(x), float(y), float(z)])
    
    channel_names = list(coords.keys())
    channel_positions = np.array(list(coords.values()))
    n_channels = len(channel_names)
    
    # Parametreler
    sfreq = 256
    duration = 5.0
    n_samples = int(duration * sfreq)
    times = np.arange(n_samples) / sfreq
    
    # Gerçek dipol konumları
    true_dipoles = np.array([
        [0, -50, 30],    # Pz yakını
        [30, 20, 50]     # FC2/C4 yakını
    ])
    
    # EEG sinyali oluştur
    eeg_data = np.zeros((n_channels, n_samples))
    
    # Her dipol için
    for d, dipole in enumerate(true_dipoles):
        # Forward model: 1/r decay
        distances = np.linalg.norm(channel_positions - dipole, axis=1)
        leadfield = 1 / (distances + 20)  # +20 singularity önleme
        leadfield = leadfield / np.max(leadfield)  # Normalize
        
        # Zaman sinyali (10Hz sinüs + gürültü)
        signal = np.sin(2 * np.pi * 10 * times + d * np.pi/2)
        signal += 0.1 * np.random.randn(n_samples)
        
        # EEG'ye ekle
        eeg_data += leadfield[:, np.newaxis] * signal * 50
    
    # Biraz pink noise ekle
    for ch in range(n_channels):
        pink = np.random.randn(n_samples)
        # Basit low-pass filter
        for i in range(1, n_samples):
            pink[i] = 0.9 * pink[i-1] + 0.1 * pink[i]
        eeg_data[ch] += pink * 5
    
    # Kontrol için en aktif kanalları yazdır
    channel_power = np.mean(eeg_data**2, axis=1)
    top_10 = np.argsort(channel_power)[-10:]
    
    print("\n🔍 Simülasyon Kontrolü:")
    print("En aktif 10 kanal (beklenen: Pz, POz, FC2, C4 civarı):")
    for idx in reversed(top_10):
        print(f"  {channel_names[idx]}: Power={channel_power[idx]:.1f}")
    
    # Kaydet
    mat_dict = {
        'dataRest': eeg_data.astype(np.float32),
        'chanlocs': np.array(channel_names, dtype=object),
        'channel_coords': channel_positions,
        'true_dipole_locations': true_dipoles,
        'srate': sfreq,
        'times': times
    }
    
    savemat(output_file, mat_dict)
    print(f"\n✓ Yeni simülasyon kaydedildi: {output_file}")
    
    return eeg_data, channel_names

# Çalıştır
create_simple_dipole_simulation('kanalkoordinatlarson.txt')
