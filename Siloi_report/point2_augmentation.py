"""
exercise1_augmentation.py — Exercise 1, Point 2: Data Augmentation
Implements 2a (data reduction) and 2b (Gaussian noise augmentation)
using the best model found in Point 1 (report_fcnn.py).

Imports model, dataset and training utilities directly from report_fcnn.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
import scipy.special
import optuna

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

# Import groupmate's utilities from Point 1
from report_fcnn import (
    data_load_clean,
    CancerDataset,
    OptimizedGPANet,
    train_model,
    DATA_PATH,
    MODEL_DB_PATH,
    CRITERION
)

SEED      = 42
N_REPEATS = 3   # repeat each experiment N times and average


# -----------------------------------------------------------------------
# HELPER: load best hyperparameters from Point 1 SQLite study
# -----------------------------------------------------------------------

def load_best_params(db_path=MODEL_DB_PATH):
    """Load best hyperparameters found by Optuna in Point 1."""
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")
    study   = optuna.load_study(study_name="cancer_model_tuning", storage=storage)
    print(f"Best params from Point 1: {study.best_params}")
    return study.best_params


# -----------------------------------------------------------------------
# HELPER: build and train a fresh model, return F1 on test set
# -----------------------------------------------------------------------

def build_and_evaluate(X_tr, y_tr, X_val, y_val, X_te, y_te,
                        p_drop, lr, input_size,
                        num_epochs=150, patience=15):
    """
    Train a fresh OptimizedGPANet with given hyperparameters.
    Returns (f1_macro, best_val_loss).
    Uses BCEWithLogitsLoss + sigmoid threshold=0.5 for binary classification.
    """
    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1)
        )
        return DataLoader(ds, batch_size=32, shuffle=shuffle)

    tr_loader  = make_loader(X_tr,  y_tr,  shuffle=True)
    val_loader = make_loader(X_val, y_val)
    te_loader  = make_loader(X_te,  y_te)

    model     = OptimizedGPANet(input_size=input_size, p_drop=p_drop)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train_model returns best_val_loss
    best_val_loss = train_model(model, num_epochs=num_epochs,
                                train_loader=tr_loader, test_loader=val_loader,
                                optimizer=optimizer, criterion=CRITERION,
                                patience=patience, verbose=False)

    # Evaluate F1 on test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for s, l in te_loader:
            probs = scipy.special.expit(model(s).numpy())
            all_preds.extend((probs > 0.5).astype(int).flatten())
            all_labels.extend(l.numpy().flatten().astype(int))

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return f1, best_val_loss


# -----------------------------------------------------------------------
# PART 2a — Effect of Reducing Training Set Size
# -----------------------------------------------------------------------

def run_reduction(X_train, y_train, X_val, y_val, X_test, y_test,
                  p_drop, lr, input_size):
    """
    Train the best model on progressively smaller fractions of the training set.
    Repeats each fraction N_REPEATS times and averages.
    """
    fractions = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00]
    results   = []

    print("\n=== Part 2a — Data Reduction ===")
    for frac in fractions:
        f1s, val_losses = [], []
        n_samples = int(frac * len(X_train))

        for rep in range(N_REPEATS):
            if frac < 1.0:
                X_tr, _, y_tr, _ = train_test_split(
                    X_train, y_train,
                    train_size=frac,
                    random_state=SEED + rep,
                    stratify=y_train
                )
            else:
                X_tr, y_tr = X_train, y_train

            f1, val_loss = build_and_evaluate(
                X_tr, y_tr, X_val, y_val, X_test, y_test,
                p_drop, lr, input_size
            )
            f1s.append(f1)
            val_losses.append(val_loss)

        results.append({
            'fraction': frac,
            'n_samples': n_samples,
            'f1_mean':       np.mean(f1s),
            'f1_std':        np.std(f1s),
            'val_loss_mean': np.mean(val_losses),
            'val_loss_std':  np.std(val_losses),
        })
        print(f"  {frac:.0%}  N={n_samples:4d}  "
              f"F1={np.mean(f1s):.4f} +- {np.std(f1s):.4f}  "
              f"ValLoss={np.mean(val_losses):.4f} +- {np.std(val_losses):.4f}")

    return pd.DataFrame(results)


# -----------------------------------------------------------------------
# PART 2b — Data Augmentation with Gaussian Noise
# -----------------------------------------------------------------------

def augment_samples(X_tr, y_tr, multiplier, noise_std=0.05, seed=SEED):
    """
    Generate (multiplier-1)*N synthetic samples by adding Gaussian noise
    to randomly selected real training samples.
    noise_std=0.05 is small relative to StandardScaler unit variance.
    Both classes augmented proportionally — class ratio preserved.
    Returns: augmented X and y (originals + synthetics), shuffled.
    """
    rng = np.random.default_rng(seed)
    N   = len(X_tr)
    n_s = (multiplier - 1) * N
    idx = rng.integers(0, N, size=n_s)
    noise     = rng.normal(0, noise_std, size=(n_s, X_tr.shape[1]))
    X_aug = np.vstack([X_tr, X_tr[idx] + noise])
    y_aug = np.concatenate([y_tr, y_tr[idx]])
    perm  = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


def run_augmentation(X_train, y_train, X_val, y_val, X_test, y_test,
                     p_drop, lr, input_size):
    """
    Train the best model with augmented training sets.
    Multipliers: x1 (baseline), x2, x3, x5.
    Repeats each N_REPEATS times and averages.
    """
    multipliers = [1, 2, 3, 5]
    results     = []

    print("\n=== Part 2b — Data Augmentation ===")
    for mult in multipliers:
        f1s, val_losses, ns = [], [], []

        for rep in range(N_REPEATS):
            if mult == 1:
                X_tr_aug, y_tr_aug = X_train, y_train
            else:
                X_tr_aug, y_tr_aug = augment_samples(
                    X_train, y_train, mult, seed=SEED + rep
                )
            ns.append(len(X_tr_aug))

            f1, val_loss = build_and_evaluate(
                X_tr_aug, y_tr_aug, X_val, y_val, X_test, y_test,
                p_drop, lr, input_size
            )
            f1s.append(f1)
            val_losses.append(val_loss)

        results.append({
            'multiplier':    mult,
            'n_train':       int(np.mean(ns)),
            'f1_mean':       np.mean(f1s),
            'f1_std':        np.std(f1s),
            'val_loss_mean': np.mean(val_losses),
            'val_loss_std':  np.std(val_losses),
        })
        print(f"  x{mult}  N={int(np.mean(ns)):5d}  "
              f"F1={np.mean(f1s):.4f} +- {np.std(f1s):.4f}  "
              f"ValLoss={np.mean(val_losses):.4f} +- {np.std(val_losses):.4f}")

    return pd.DataFrame(results)


# -----------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------

def plot_results(red_df, aug_df):
    """Generate and save all plots for 2a and 2b."""

    # --- Figure 1: F1 score (2a and 2b side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.errorbar(red_df['n_samples'], red_df['f1_mean'], yerr=red_df['f1_std'],
                marker='o', capsize=4, color='steelblue', label='Test F1 (macro)')
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('F1 Score (macro, mean +- std)')
    ax.set_title('2a — Effect of Reducing Training Set Size')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.errorbar(aug_df['multiplier'], aug_df['f1_mean'], yerr=aug_df['f1_std'],
                 marker='s', capsize=4, color='darkorange', label='Test F1 (macro)')
    ax2.set_xlabel('Augmentation multiplier')
    ax2.set_ylabel('F1 Score (macro, mean +- std)')
    ax2.set_title('2b — Effect of Data Augmentation')
    ax2.set_xticks(aug_df['multiplier'])
    ax2.set_xticklabels([f'x{m}' for m in aug_df['multiplier']])
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('augmentation_f1.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Plot saved to augmentation_f1.png")

    # --- Figure 2: Validation loss vs training set size (2a) ---
    fig2, ax3 = plt.subplots(figsize=(9, 5))
    ax3.errorbar(red_df['n_samples'], red_df['val_loss_mean'], yerr=red_df['val_loss_std'],
                 marker='o', capsize=4, color='steelblue', label='Validation Loss')
    ax3.set_xlabel('Number of training samples')
    ax3.set_ylabel('Best Validation Loss (BCEWithLogitsLoss, mean +- std)')
    ax3.set_title('2a — Validation Loss vs Training Set Size')
    ax3.legend(); ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('val_loss_reduction.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Plot saved to val_loss_reduction.png")

    # --- Figure 3: Validation loss vs augmentation multiplier (2b) ---
    fig3, ax4 = plt.subplots(figsize=(9, 5))
    ax4.errorbar(aug_df['multiplier'], aug_df['val_loss_mean'], yerr=aug_df['val_loss_std'],
                 marker='s', capsize=4, color='darkorange', label='Validation Loss')
    ax4.set_xlabel('Augmentation multiplier')
    ax4.set_ylabel('Best Validation Loss (BCEWithLogitsLoss, mean +- std)')
    ax4.set_title('2b — Validation Loss vs Augmentation Multiplier')
    ax4.set_xticks(aug_df['multiplier'])
    ax4.set_xticklabels([f'x{m}' for m in aug_df['multiplier']])
    ax4.legend(); ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('val_loss_augmentation.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Plot saved to val_loss_augmentation.png")

    # --- Figure 4: Real vs Augmented (F1 comparison) ---
    fig4, ax5 = plt.subplots(figsize=(9, 5))
    ax5.errorbar(red_df['n_samples'], red_df['f1_mean'], yerr=red_df['f1_std'],
                 marker='o', capsize=4, color='steelblue', label='Reduced (real samples)')
    ax5.errorbar(aug_df['n_train'], aug_df['f1_mean'], yerr=aug_df['f1_std'],
                 marker='s', capsize=4, color='darkorange', linestyle='--',
                 label='Augmented (synthetic)')
    ax5.set_xlabel('Number of training samples')
    ax5.set_ylabel('Test F1 Score (macro)')
    ax5.set_title('2a vs 2b — Real vs Augmented Samples')
    ax5.legend(); ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('real_vs_augmented.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Plot saved to real_vs_augmented.png")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

if __name__ == "__main__":

    # --- Load data (same preprocessing as report_fcnn.py) ---
    df       = data_load_clean(DATA_PATH)
    features = df.columns[2:-1].tolist()
    target   = 'diagnosis'

    X = df[features].values
    y = df[target].values

    input_size = len(features)

    # --- Train / val / test split (stratified) ---
    # 60% train, 20% val, 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval
    )

    # Scale features (fit on training only)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")
    print(f"Input features: {input_size}")

    # --- Load best hyperparameters from Point 1 ---
    best_params = load_best_params(MODEL_DB_PATH)
    p_drop      = best_params['dropout_rate']
    lr          = best_params['lr']
    print(f"Using: dropout={p_drop}, lr={lr:.6f}")

    # --- Run experiments ---
    red_df = run_reduction(X_train, y_train, X_val, y_val, X_test, y_test,
                           p_drop, lr, input_size)

    aug_df = run_augmentation(X_train, y_train, X_val, y_val, X_test, y_test,
                              p_drop, lr, input_size)

    # --- Summary tables ---
    print("\n=== 2a — Reduction summary ===")
    print(red_df.round(4).to_string(index=False))

    print("\n=== 2b — Augmentation summary ===")
    print(aug_df.round(4).to_string(index=False))

    # --- Plots ---
    plot_results(red_df, aug_df)
