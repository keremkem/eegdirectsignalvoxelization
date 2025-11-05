# check_csv_content.py
"""
CSV dosyalarının içeriğini kontrol et
"""

import pandas as pd
from pathlib import Path

DATA_PATH = r"C:\Users\kerem\Downloads\eegyedek\eckfoldgroup"

# Bir örnek dosya aç
test_files = [
    "S10_alpha_phys_metrics_v33.csv",
    "S10_beta_phys_metrics_v33.csv", 
    "S10_theta_phys_metrics_v33.csv"
]

for filename in test_files:
    filepath = Path(DATA_PATH) / filename
    
    if not filepath.exists():
        print(f"❌ {filename} bulunamadı")
        continue
    
    print(f"\n{'='*70}")
    print(f"📄 {filename}")
    print(f"{'='*70}")
    
    df = pd.read_csv(filepath)
    
    print("\nSütunlar:")
    print(df.columns.tolist())
    
    print("\nİlk 10 satır:")
    print(df.head(10))
    
    print("\n'Metric' sütunundaki değerler:")
    if 'Metric' in df.columns:
        print(df['Metric'].unique())
    else:
        print("⚠️  'Metric' sütunu yok!")
    
    print("\n'post_ant' içeren satırlar:")
    if 'Metric' in df.columns:
        matches = df[df['Metric'].str.contains('post', case=False, na=False)]
        print(matches)
    
    print("\n" + "="*70)