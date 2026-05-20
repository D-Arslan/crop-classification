"""
train.py — Entraînement MCTNet
(early stopping)
Wang et al., 2024
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.mctnet import MCTNet


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

CONFIG = {
    'lr': 0.001,
    'batch_size': 32,
    'epochs': 200,
    'n_head': 5,
    'kernel_size': 3,
    'dropout': 0.1,
    'print_every': 10,
    'es_patience': 30,
}

N_CLASSES = {
    'Arkansas': 5,
    'California': 6,
}


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class CropDataset(Dataset):

    def __init__(self, data_dir: str, region: str, split: str):

        prefix = os.path.join(data_dir, f'{region}_{split}')

        self.X = torch.from_numpy(
            np.load(f'{prefix}_input1.npy')
        ).float()

        self.mask = torch.from_numpy(
            np.load(f'{prefix}_input2.npy')
        ).float()

        self.y = torch.from_numpy(
            np.load(f'{prefix}_labels.npy')
        ).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.mask[idx],
            self.y[idx]
        )


# ---------------------------------------------------------------------------
# TRAIN ONE EPOCH
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    for X, mask, y in loader:

        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X, mask)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(
    model,
    loader,
    criterion,
    device,
    return_preds=False,
):

    model.eval()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for X, mask, y in loader:

            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(X, mask)

            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)

            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    oa = accuracy_score(all_labels, all_preds)

    kappa = cohen_kappa_score(all_labels, all_preds)

    f1 = f1_score(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0,
    )

    loss = total_loss / len(loader.dataset)

    if return_preds:

        return (
            loss,
            oa,
            kappa,
            f1,
            all_preds,
            all_labels,
        )

    return loss, oa, kappa, f1


# ---------------------------------------------------------------------------
# PLOT CURVES
# ---------------------------------------------------------------------------

def plot_curves(history, region, scale_tag):

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 5)
    )

    # LOSS
    axes[0].plot(
        epochs,
        history['train_loss'],
        label='Train'
    )

    axes[0].plot(
        epochs,
        history['val_loss'],
        label='Validation'
    )

    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # OA
    axes[1].plot(
        epochs,
        history['val_oa']
    )

    axes[1].set_title('Validation OA')
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    # KAPPA
    axes[2].plot(
        epochs,
        history['val_kappa']
    )

    axes[2].set_title('Validation Kappa')
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)

    # F1
    axes[3].plot(
        epochs,
        history['val_f1']
    )

    axes[3].set_title('Validation F1')
    axes[3].set_ylim(0, 1)
    axes[3].grid(alpha=0.3)

    fig.suptitle(
        f'MCTNet - {region} - {scale_tag}'
    )

    plt.tight_layout()

    save_name = f'curves_{region}_{scale_tag}.png'

    plt.savefig(save_name, dpi=150)

    plt.show()

    print(f'Courbes sauvegardées : {save_name}')


# ---------------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------------

def plot_confusion(
    labels,
    preds,
    region,
    scale_tag,
):

    cm = confusion_matrix(
        labels,
        preds,
        normalize='true',
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    disp = ConfusionMatrixDisplay(cm)

    disp.plot(
        ax=ax,
        cmap='Blues',
        colorbar=True,
        values_format='.2f',
    )

    ax.set_title(
        f'Matrice de confusion - {region}'
    )

    save_name = f'confusion_{region}_{scale_tag}.png'

    plt.savefig(save_name, dpi=150)

    plt.show()

    print(f'Matrice sauvegardée : {save_name}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(region: str, data_dir: str):

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    n_classes = N_CLASSES[region]

    # Détection scale
    if 'scale20' in data_dir:

        scale_tag = 'scale20'

    elif 'scale30' in data_dir:

        scale_tag = 'scale30'

    else:

        scale_tag = 'scale_unknown'

    save_path = f'best_{region}_{scale_tag}.pth'

    print('=' * 60)
    print(f'MCTNet — {region}')
    print(f'Device : {device}')
    print(f'Data dir : {data_dir}')
    print('=' * 60)

    # ----------------------------------------------------------------------
    # DATASETS
    # ----------------------------------------------------------------------

    train_set = CropDataset(
        data_dir,
        region,
        'train'
    )

    val_set = CropDataset(
        data_dir,
        region,
        'val'
    )

    test_set = CropDataset(
        data_dir,
        region,
        'test'
    )

    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
    )

    print(
        f'\nDonnees : '
        f'{len(train_set)} train | '
        f'{len(val_set)} val | '
        f'{len(test_set)} test'
    )

    # ----------------------------------------------------------------------
    # MODEL
    # ----------------------------------------------------------------------

    model = MCTNet(
        n_classes=n_classes,
        n_head=CONFIG['n_head'],
        kernel_size=CONFIG['kernel_size'],
        dropout=CONFIG['dropout'],
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f'Parametres : {total_params:,}')

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['lr']
    )

    # ----------------------------------------------------------------------
    # HISTORY
    # ----------------------------------------------------------------------

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_oa': [],
        'val_kappa': [],
        'val_f1': [],
    }

    # ----------------------------------------------------------------------
    # EARLY STOPPING
    # ----------------------------------------------------------------------

    es_patience = CONFIG.get(
        'es_patience',
        30
    )

    es_counter = 0

    best_val_loss = float('inf')

    # ----------------------------------------------------------------------
    # TRAIN LOOP
    # ----------------------------------------------------------------------

    best_f1 = 0.0
    best_epoch = 0

    t0 = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):

        # TRAIN
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # VALIDATION
        val_loss, val_oa, val_kappa, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # HISTORY
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_oa'].append(val_oa)
        history['val_kappa'].append(val_kappa)
        history['val_f1'].append(val_f1)

        # SAVE BEST MODEL
        if val_f1 > best_f1:

            best_f1 = val_f1
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                save_path
            )

        # ------------------------------------------------------------------
        # EARLY STOPPING
        # ------------------------------------------------------------------

        if val_loss < best_val_loss - 1e-4:

            best_val_loss = val_loss

            es_counter = 0

        else:

            es_counter += 1

            if es_counter >= es_patience:

                print(
                    f'\nEarly stopping à l\'epoch {epoch} '
                    f'(best epoch={best_epoch}, '
                    f'best F1={best_f1:.4f})'
                )

                break

        # PRINT
        if epoch % CONFIG['print_every'] == 0 or epoch == 1:

            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{CONFIG['epochs']} "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| OA={val_oa:.4f} "
                f"| Kappa={val_kappa:.4f} "
                f"| F1={val_f1:.4f} "
                f"| best_F1={best_f1:.4f} "
                f"| {elapsed:.0f}s"
            )

    # ----------------------------------------------------------------------
    # LOAD BEST MODEL
    # ----------------------------------------------------------------------

    print(
        f'\nChargement du meilleur modèle '
        f'(epoch {best_epoch}, F1={best_f1:.4f})'
    )

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
        )
    )

    (
        _,
        test_oa,
        test_kappa,
        test_f1,
        preds,
        labels,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
        return_preds=True,
    )

    # ----------------------------------------------------------------------
    # RESULTS
    # ----------------------------------------------------------------------

    print('\n' + '=' * 60)

    print(f'RESULTATS FINAUX — {region}')

    print('=' * 60)

    print(f'OA     : {test_oa:.4f}')
    print(f'Kappa : {test_kappa:.4f}')
    print(f'F1     : {test_f1:.4f}')

    print(f'\nModèle sauvegardé : {save_path}')

    # ----------------------------------------------------------------------
    # PLOTS
    # ----------------------------------------------------------------------

    plot_curves(
        history,
        region,
        scale_tag,
    )

    plot_confusion(
        labels,
        preds,
        region,
        scale_tag,
    )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Entrainement MCTNet'
    )

    parser.add_argument(
        '--region',
        choices=['Arkansas', 'California'],
        default='Arkansas',
    )

    parser.add_argument(
        '--data_dir',
        default='../../data/preprocessed/scale30',
    )

    args = parser.parse_args()

    main(
        args.region,
        args.data_dir,
    )