"""
train.py — Entraînement MCTNet
Wang et al., 2024

Usage :
    # depuis le dossier "Point 5 — Model Implementation"

    # Scale 30m
    python train.py --region Arkansas   --data_dir ../../data/preprocessed/scale30
    python train.py --region California --data_dir ../../data/preprocessed/scale30

    # Scale 20m
    python train.py --region Arkansas   --data_dir ../../data/preprocessed/scale20
    python train.py --region California --data_dir ../../data/preprocessed/scale20

Sorties :
    best_Arkansas_scale30.pth    — meilleur modèle Arkansas scale 30m (selon F1 val)
    best_California_scale30.pth  — meilleur modèle Californie scale 30m
    best_Arkansas_scale20.pth    — meilleur modèle Arkansas scale 20m
    best_California_scale20.pth  — meilleur modèle Californie scale 20m
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.mctnet import MCTNet


# ---------------------------------------------------------------------------
# Configuration — hyperparamètres (Table 3 de l'article)
# ---------------------------------------------------------------------------

CONFIG = {
    'lr':         0.001,
    'batch_size': 32,
    'epochs':     200,
    'n_head':     5,
    'kernel_size': 3,
    'dropout':    0.1,
    'print_every': 10,   # afficher un résumé toutes les N époques
}

N_CLASSES = {
    'Arkansas':   5,
    'California': 6,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    """
    Charge les fichiers .npy produits par le preprocessing (Sarah).

    Args:
        data_dir : dossier contenant les fichiers .npy
        region   : 'Arkansas' ou 'California'
        split    : 'train', 'val', ou 'test'
    """

    def __init__(self, data_dir: str, region: str, split: str):
        prefix = os.path.join(data_dir, f'{region}_{split}')
        self.X    = torch.from_numpy(np.load(f'{prefix}_input1.npy'))  # (N, 10, 36)
        self.mask = torch.from_numpy(np.load(f'{prefix}_input2.npy'))  # (N, 36)
        self.y    = torch.from_numpy(np.load(f'{prefix}_labels.npy'))  # (N,)

        # Garantit les bons types même si les fichiers ont été régénérés
        self.X    = self.X.float()
        self.mask = self.mask.float()
        self.y    = self.y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Entraînement — une époque
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Entraîne le modèle sur un DataLoader pendant une époque.

    Returns:
        loss_mean : loss moyenne sur l'époque
    """
    model.train()
    total_loss = 0.0

    for X, mask, y in loader:
        X, mask, y = X.to(device), mask.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X, mask)          # (B, N_classes)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Évaluation — OA / Kappa / F1
# ---------------------------------------------------------------------------

def evaluate(model, loader, criterion, device):
    """
    Évalue le modèle sur un DataLoader.

    Returns:
        loss  : loss moyenne
        oa    : Overall Accuracy
        kappa : Cohen's Kappa
        f1    : F1 macro-averaged
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X, mask, y in loader:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            logits = model(X, mask)
            loss   = criterion(logits, y)
            total_loss += loss.item() * len(y)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    oa    = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1    = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    loss  = total_loss / len(loader.dataset)

    return loss, oa, kappa, f1


# ---------------------------------------------------------------------------
# Main — boucle complète
# ---------------------------------------------------------------------------

def main(region: str, data_dir: str):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes  = N_CLASSES[region]

    # Déduire le nom de la scale depuis le chemin data_dir pour nommer le fichier sauvegardé
    if 'scale20' in data_dir:
        scale_tag = 'scale20'
    elif 'scale30' in data_dir:
        scale_tag = 'scale30'
    else:
        scale_tag = 'scale_unknown'
    save_path  = f'best_{region}_{scale_tag}.pth'

    print('=' * 60)
    print(f'MCTNet — {region}  ({n_classes} classes)  [{scale_tag}]')
    print(f'Device : {device}')
    print(f'Hyperparametres : {CONFIG}')
    print(f'Data dir : {data_dir}')
    print('=' * 60)

    # --- Données ---
    train_set = CropDataset(data_dir, region, 'train')
    val_set   = CropDataset(data_dir, region, 'val')
    test_set  = CropDataset(data_dir, region, 'test')

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=CONFIG['batch_size'], shuffle=False)

    print(f'\nDonnees : {len(train_set)} train | {len(val_set)} val | {len(test_set)} test')

    # --- Modèle ---
    model = MCTNet(
        n_classes=n_classes,
        n_head=CONFIG['n_head'],
        kernel_size=CONFIG['kernel_size'],
        dropout=CONFIG['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parametres : {total_params:,}  (article Table 6 : 55 059)\n')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # --- Boucle d'entraînement ---
    best_f1    = 0.0
    best_epoch = 0
    t0         = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_oa, val_kappa, val_f1 = evaluate(model, val_loader, criterion, device)

        # Sauvegarde du meilleur modèle (selon F1 val)
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        # Affichage toutes les N époques
        if epoch % CONFIG['print_every'] == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f'Epoch {epoch:3d}/{CONFIG["epochs"]} '
                f'| train_loss={train_loss:.4f} '
                f'| val_loss={val_loss:.4f} '
                f'| val_OA={val_oa:.4f} '
                f'| val_Kappa={val_kappa:.4f} '
                f'| val_F1={val_f1:.4f} '
                f'| best_F1={best_f1:.4f} (ep{best_epoch}) '
                f'| {elapsed:.0f}s'
            )

    # --- Évaluation finale sur le test set ---
    print(f'\nChargement du meilleur modele (epoch {best_epoch}, val_F1={best_f1:.4f})...')
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, test_oa, test_kappa, test_f1 = evaluate(model, test_loader, criterion, device)

    print('\n' + '=' * 60)
    print(f'RESULTATS FINAUX — {region}')
    print('=' * 60)
    print(f'  OA    : {test_oa:.4f}  (article : {"0.968" if region == "Arkansas" else "0.852"})')
    print(f'  Kappa : {test_kappa:.4f}  (article : {"0.951" if region == "Arkansas" else "0.806"})')
    print(f'  F1    : {test_f1:.4f}  (article : {"0.933" if region == "Arkansas" else "0.829"})')
    print(f'\nModele sauvegarde : {save_path}')


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrainement MCTNet')
    parser.add_argument(
        '--region',
        choices=['Arkansas', 'California'],
        default='Arkansas',
        help='Dataset a utiliser (defaut: Arkansas)',
    )
    parser.add_argument(
        '--data_dir',
        default='../../data/preprocessed/scale30',
        help='Chemin vers les fichiers .npy (defaut: ../../data/preprocessed/scale30)',
    )
    args = parser.parse_args()
    main(args.region, args.data_dir)
