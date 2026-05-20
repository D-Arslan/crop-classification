import pandas as pd
import numpy as np

np.random.seed(42)

ark_files = [
    'Arkansas_Z1_final.csv',
    'Arkansas_Z2_final.csv',
    'Arkansas_Z3_final.csv',
    'Arkansas_Z4_final.csv',
    'Arkansas_Z5_final.csv'
]

ark = pd.concat([pd.read_csv(f) for f in ark_files], ignore_index=True)
ark = ark.drop(columns=['system:index', '.geo'], errors='ignore')

print(f"Arkansas avant sous-échantillonnage: {len(ark)} points")
ark_sampled = ark.sample(n=10000, random_state=42).reset_index(drop=True)

print(f"Arkansas après sous-échantillonnage: {len(ark_sampled)} points")
print("\nDistribution des classes Arkansas:")
label_map_ark = {0: 'Corn', 1: 'Cotton', 2: 'Rice', 3: 'Soybeans', 4: 'Others'}
for lab, name in label_map_ark.items():
    count = (ark_sampled['label'] == lab).sum()
    pct = count / len(ark_sampled) * 100
    print(f"  {lab} ({name:10s}): {count:5d} ({pct:5.1f}%)")

ark_sampled.to_csv('Arkansas_10k.csv', index=False)
print("\n-> Fichier sauvegardé : Arkansas_10k.csv")

cal_files = [
    'California_Z1_final.csv',
    'California_Z2_final.csv',
    'California_Z3_final.csv',
    'California_Z4_final.csv',
    'California_Z5_final.csv',
    'California_Z6_final.csv',
    'California_Z7_final.csv',
    'California_Z8_final.csv'
]

cal = pd.concat([pd.read_csv(f) for f in cal_files], ignore_index=True)
cal = cal.drop(columns=['system:index', '.geo'], errors='ignore')

print(f"\n{'='*50}")
print(f"California avant sous-échantillonnage: {len(cal)} points")
cal_sampled = cal.sample(n=10000, random_state=42).reset_index(drop=True)

print(f"California après sous-échantillonnage: {len(cal_sampled)} points")
print("\nDistribution des classes California:")
label_map_cal = {0: 'Rice', 1: 'Alfalfa', 2: 'Grapes', 3: 'Almonds', 4: 'Pistachios', 5: 'Others'}
for lab, name in label_map_cal.items():
    count = (cal_sampled['label'] == lab).sum()
    pct = count / len(cal_sampled) * 100
    print(f"  {lab} ({name:12s}): {count:5d} ({pct:5.1f}%)")

cal_sampled.to_csv('California_10k.csv', index=False)
print("\n-> Fichier sauvegardé : California_10k.csv")
