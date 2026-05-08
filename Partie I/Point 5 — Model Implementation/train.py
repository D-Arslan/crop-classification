"""
train.py — Entraînement MCTNet / GatedMCTNet
Wang et al., 2024

Usage :
    # depuis le dossier "Point 5 — Model Implementation"

    python train.py --region Arkansas   --data_dir ../../data/preprocessed/partie1
    python train.py --region California --data_dir ../../data/preprocessed/partie1

    # Partie 3 — GatedMCTNet
    python train.py --region Arkansas   --data_dir ../../data/preprocessed/partie1 --model gated
    python train.py --region California --data_dir ../../data/preprocessed/partie1 --model gated

Sorties :
    best_{region}_{model}.pth  — meilleur modèle selon F1 val
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.mctnet import MCTNet, GatedMCTNet


# ---------------------------------------------------------------------------
# Configuration — hyperparamètres (Table 3 de l'article)
# ---------------------------------------------------------------------------

CONFIG = {
    'lr':           0.001,
    'weight_decay': 1e-4,
    'batch_size':   32,
    'epochs':       200,
    'n_head':       5,
    'kernel_size':  3,
    'dropout':      0.1,
    'print_every':  10,
    'patience':     20,   # early stopping
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

def main(region: str, data_dir: str, model_name: str):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = N_CLASSES[region]
    save_path = f'best_{region}_{model_name}.pth'

    print('=' * 60)
    print(f'{model_name.upper()} -- {region}  ({n_classes} classes)')
    print(f'Device : {device}')
    print(f'Data dir : {data_dir}')
    print('=' * 60)

    # --- Données ---
    train_set = CropDataset(data_dir, region, 'train')
    val_set   = CropDataset(data_dir, region, 'val')
    test_set  = CropDataset(data_dir, region, 'test')

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=CONFIG['batch_size'], shuffle=False)

    print(f'Donnees : {len(train_set)} train | {len(val_set)} val | {len(test_set)} test')

    # --- Modèle ---
    model_cls = GatedMCTNet if model_name == 'gated' else MCTNet
    model = model_cls(
        n_classes=n_classes,
        n_head=CONFIG['n_head'],
        kernel_size=CONFIG['kernel_size'],
        dropout=CONFIG['dropout'],
    ).to(device)

    print(f'Parametres : {sum(p.numel() for p in model.parameters()):,}\n')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # --- Boucle d'entraînement ---
    best_f1        = 0.0
    best_epoch     = 0
    epochs_no_impr = 0
    t0             = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_oa, val_kappa, val_f1 = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if val_f1 > best_f1:
            best_f1        = val_f1
            best_epoch     = epoch
            epochs_no_impr = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_impr += 1

        if epoch % CONFIG['print_every'] == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f'Ep {epoch:3d}/{CONFIG["epochs"]} '
                f'| train={train_loss:.4f} '
                f'| val={val_loss:.4f} '
                f'| OA={val_oa:.4f} '
                f'| F1={val_f1:.4f} '
                f'| best={best_f1:.4f}(ep{best_epoch}) '
                f'| {elapsed:.0f}s'
            )

        if epochs_no_impr >= CONFIG['patience']:
            print(f'\nEarly stopping a l\'epoch {epoch} (best ep={best_epoch})')
            break

    # --- Évaluation finale sur le test set ---
    print(f'\nChargement meilleur modele (epoch {best_epoch}, val_F1={best_f1:.4f})...')
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, test_oa, test_kappa, test_f1 = evaluate(model, test_loader, criterion, device)

    print('\n' + '=' * 60)
    print(f'RESULTATS FINAUX -- {region} [{model_name}]')
    print('=' * 60)
    print(f'  OA    : {test_oa:.4f}')
    print(f'  Kappa : {test_kappa:.4f}')
    print(f'  F1    : {test_f1:.4f}')
    print(f'\nModele sauvegarde : {save_path}')


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrainement MCTNet / GatedMCTNet')
    parser.add_argument(
        '--region',
        choices=['Arkansas', 'California'],
        default='Arkansas',
    )
    parser.add_argument(
        '--data_dir',
        default='../../data/preprocessed/partie1',
    )
    parser.add_argument(
        '--model',
        choices=['mctnet', 'gated'],
        default='gated',
        help='mctnet = MCTNet original | gated = GatedMCTNet (Partie 3)',
    )
    args = parser.parse_args()
    main(args.region, args.data_dir, args.model)
