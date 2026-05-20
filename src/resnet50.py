"""
src/models/resnet50.py
Definición, entrenamiento y evaluación del modelo ResNet-50 para clasificación
de enfermedades en plantas (PlantVillage).

Extraído de: notebooks/02_cnn_training.ipynb
Proyecto Final IA · EAFIT 2026-1
"""

import copy
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# Hiperparámetros por defecto
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS_FASE_A: int   = 10      # Entrenar solo la cabeza FC (base congelada)
EPOCHS_FASE_B: int   = 10      # Fine-tuning layer4 + FC
EPOCHS_TOTAL:  int   = EPOCHS_FASE_A + EPOCHS_FASE_B

LR_FASE_A:     float = 1e-3    # LR para la cabeza clasificadora
LR_FASE_B:     float = 1e-4    # LR para fine-tuning
WEIGHT_DECAY:  float = 1e-4    # Regularización L2
PATIENCE:      int   = 5       # Early stopping


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del modelo
# ─────────────────────────────────────────────────────────────────────────────
def build_resnet50(n_classes: int, freeze_base: bool = True) -> nn.Module:
    """
    Construye ResNet-50 preentrenado (ImageNet) con cabeza de clasificación
    personalizada para n_classes clases.

    Estrategia de transfer learning en 2 fases:
      - Fase A: freeze_base=True  → solo entrena la FC nueva
      - Fase B: unfreeze_layer4() → fine-tuning de layer4 + FC

    Args:
        n_classes:   Número de clases de salida.
        freeze_base: Si True, congela todos los parámetros del backbone.

    Returns:
        Modelo nn.Module listo para mover a device y entrenar.

    Ejemplo:
        >>> model = build_resnet50(n_classes=8, freeze_base=True)
        >>> model = model.to(device)
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model   = models.resnet50(weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Reemplazar capa FC original (in_features=2048)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, n_classes),
    )
    return model


def load_resnet50(model_path: str | Path, n_classes: int, device: torch.device) -> nn.Module:
    """
    Carga un ResNet-50 previamente entrenado desde un archivo .pth.

    Args:
        model_path: Ruta al archivo de pesos (best_resnet50.pth).
        n_classes:  Número de clases (debe coincidir con el entrenamiento).
        device:     Dispositivo destino (cpu / cuda).

    Returns:
        Modelo cargado en modo eval(), listo para inferencia.

    Ejemplo:
        >>> model = load_resnet50('models/checkpoints/best_resnet50.pth', 8, device)
    """
    model = build_resnet50(n_classes=n_classes, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Descongelar capa4 para Fase B
# ─────────────────────────────────────────────────────────────────────────────
def unfreeze_layer4(model: nn.Module) -> None:
    """
    Descongela layer4 + avgpool + fc para fine-tuning profundo (Fase B).
    Llamar justo antes de iniciar la Fase B del entrenamiento.

    Args:
        model: Modelo ResNet-50 con base congelada.
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in ["layer4", "avgpool", "fc"]):
            param.requires_grad = True


def count_params(model: nn.Module) -> dict:
    """
    Cuenta parámetros totales, entrenables y congelados del modelo.

    Returns:
        Dict con keys: 'total', 'trainable', 'frozen'.

    Ejemplo:
        >>> p = count_params(model)
        >>> print(f"Entrenables: {p['trainable']:,} / {p['total']:,}")
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ─────────────────────────────────────────────────────────────────────────────
# Loop de entrenamiento / validación
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device:    torch.device = torch.device("cpu"),
    phase:     str = "train",
) -> tuple[float, float]:
    """
    Ejecuta una época completa de entrenamiento o validación.

    Args:
        model:     Modelo PyTorch.
        loader:    DataLoader (train, val o test).
        criterion: Función de pérdida (CrossEntropyLoss).
        optimizer: Optimizador. Si None, corre en modo inferencia (val/test).
        device:    Dispositivo donde corre el modelo.
        phase:     'train' activa gradientes; cualquier otro valor los desactiva.

    Returns:
        Tupla (epoch_loss, epoch_accuracy) como floats.

    Ejemplo:
        >>> loss, acc = run_epoch(model, loader_train, criterion, optimizer,
        ...                       device, phase='train')
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds      = outputs.max(1)
            correct       += preds.eq(labels).sum().item()
            total         += imgs.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento completo (Fase A + Fase B)
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model:         nn.Module,
    loader_train:  DataLoader,
    loader_val:    DataLoader,
    device:        torch.device,
    save_path:     str | Path,
    epochs_fase_a: int   = EPOCHS_FASE_A,
    epochs_fase_b: int   = EPOCHS_FASE_B,
    lr_fase_a:     float = LR_FASE_A,
    lr_fase_b:     float = LR_FASE_B,
    weight_decay:  float = WEIGHT_DECAY,
    patience:      int   = PATIENCE,
    save_to_drive_fn=None,  # función opcional: save_to_drive(path) para Drive
) -> dict:
    """
    Entrena ResNet-50 en dos fases con early stopping y checkpoint automático.

    Fase A (épocas 1-epochs_fase_a):
        Solo la cabeza FC entrena (base congelada). LR alto.

    Fase B (épocas epochs_fase_a+1 en adelante):
        Se descongela layer4 para fine-tuning profundo. LR bajo.

    El mejor modelo (según val_accuracy) se guarda en `save_path`.

    Args:
        model:          Modelo construido con build_resnet50().
        loader_train:   DataLoader de entrenamiento.
        loader_val:     DataLoader de validación.
        device:         Dispositivo (cpu / cuda).
        save_path:      Ruta donde guardar best_resnet50.pth.
        epochs_fase_a:  Épocas de la Fase A.
        epochs_fase_b:  Épocas de la Fase B.
        lr_fase_a:      Learning rate Fase A.
        lr_fase_b:      Learning rate Fase B.
        weight_decay:   Regularización L2.
        patience:       Épocas sin mejora antes de early stopping (solo Fase B).
        save_to_drive_fn: Función f(path) para backup en Drive (opcional).

    Returns:
        Dict 'history' con listas: train_loss, val_loss, train_acc, val_acc, lr.

    Ejemplo:
        >>> history = train(model, loader_train, loader_val, device,
        ...                 save_path='models/checkpoints/best_resnet50.pth')
    """
    save_path   = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_total = epochs_fase_a + epochs_fase_b
    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_fase_a, weight_decay=weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [], "lr": [],
    }
    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve     = 0

    print("Iniciando entrenamiento...")
    print("=" * 70)

    for epoch in range(1, epochs_total + 1):

        # ── Transición a Fase B ───────────────────────────────────────────────
        if epoch == epochs_fase_a + 1:
            print("\n" + "─" * 70)
            print(f"  FASE B: descongelando layer4 — LR → {lr_fase_b}")
            print("─" * 70)
            unfreeze_layer4(model)
            p = count_params(model)
            print(f"  Parámetros entrenables ahora: {p['trainable']:,}")
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_fase_b, weight_decay=weight_decay,
            )
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, loader_train, criterion, optimizer, device, "train")
        val_loss,   val_acc   = run_epoch(model, loader_val,   criterion, None,      device, "val")
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        fase   = "A" if epoch <= epochs_fase_a else "B"
        marker = ""

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            if save_to_drive_fn:
                save_to_drive_fn(save_path)   # backup silencioso en Drive
            no_improve = 0
            marker     = " ✅ MEJOR"
        else:
            no_improve += 1

        print(
            f"[{fase}] Epoch {epoch:02d}/{epochs_total} "
            f"| Loss: {train_loss:.4f}/{val_loss:.4f} "
            f"| Acc: {train_acc:.4f}/{val_acc:.4f} "
            f"| LR: {current_lr:.2e} "
            f"| {elapsed:.0f}s{marker}"
        )

        if no_improve >= patience and epoch > epochs_fase_a:
            print(f"\n⏹  Early stopping en época {epoch} (sin mejora por {patience} épocas)")
            break

    print("\n" + "=" * 70)
    print(f"Entrenamiento finalizado. Mejor val_acc: {best_val_acc:.4f}")
    print(f"Modelo guardado en: {save_path}")

    # Restaurar mejores pesos al modelo en memoria
    model.load_state_dict(best_model_wts)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación en test set
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    model:       nn.Module,
    loader_test: DataLoader,
    device:      torch.device,
    class_names: list[str],
) -> dict:
    """
    Evalúa el modelo en el test set y devuelve métricas detalladas.

    Args:
        model:       Modelo entrenado (en modo eval).
        loader_test: DataLoader del test set.
        device:      Dispositivo (cpu / cuda).
        class_names: Lista de nombres de clases en orden de índice.

    Returns:
        Dict con:
          - 'accuracy'     : float
          - 'f1_macro'     : float
          - 'f1_weighted'  : float
          - 'auc_roc'      : float (o None si falla)
          - 'all_preds'    : np.ndarray de predicciones
          - 'all_labels'   : np.ndarray de etiquetas reales
          - 'all_probs'    : np.ndarray de probabilidades (N × n_classes)

    Ejemplo:
        >>> metrics = evaluate(model, loader_test, device, class_names)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader_test:
            imgs    = imgs.to(device)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    return {
        "accuracy"   : float(acc),
        "f1_macro"   : float(f1),
        "f1_weighted": float(f1_w),
        "auc_roc"    : float(auc) if auc is not None else None,
        "all_preds"  : all_preds,
        "all_labels" : all_labels,
        "all_probs"  : all_probs,
    }
