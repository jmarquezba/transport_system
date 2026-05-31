"""
Módulo 2 — Clasificación de Conducción Distractiva
ResNet18 Transfer Learning — Dataset real de Kaggle
arafatsahinafridi/multi-class-driver-behavior-image-dataset

Clases reales del dataset:
  safe_driving / turning / texting_phone / talking_phone / other_activities
"""

import os, sys, pickle, shutil
import numpy as np
from pathlib import Path

import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

EPOCHS = 25; BATCH_SIZE = 32; LR = 1e-4; IMG_SIZE = 224; PATIENCE = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEBAPP_DIR = BASE_DIR.parent / "webapp" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WEBAPP_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Descarga ──────────────────────────────────────────────────────────────
def download_dataset() -> Path:
    try:
        import kagglehub
        print("Descargando dataset CNN desde Kaggle...")
        raw = Path(kagglehub.dataset_download(
            "arafatsahinafridi/multi-class-driver-behavior-image-dataset"))
    except Exception as e:
        sys.exit(f"[ERROR] kagglehub falló: {e}\nConfigura ~/.kaggle/kaggle.json")

    print(f"  Raw path: {raw}")

    # Buscar el directorio que tenga subcarpetas con imágenes
    # Acepta CUALQUIER nombre de carpeta (safe_driving, c0, etc.)
    best_dir   = None
    best_count = 0
    for candidate in [raw, *raw.rglob("*")]:
        candidate = Path(candidate)
        if not candidate.is_dir():
            continue
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if len(subdirs) < 2:
            continue
        # Contar imágenes totales en subdirectorios
        total_imgs = 0
        for sd in subdirs:
            total_imgs += len(list(sd.glob("*.jpg")) + list(sd.glob("*.png"))
                              + list(sd.glob("*.jpeg")) + list(sd.glob("*.bmp")))
        if total_imgs > best_count:
            best_count = total_imgs
            best_dir   = candidate

    if best_dir is None or best_count == 0:
        print(f"Contenido de {raw}:")
        for p in sorted(raw.rglob("*")):
            if p.is_dir(): print(f"  DIR: {p}")
        sys.exit("No se encontraron carpetas con imágenes en el dataset.")

    # Listar clases detectadas
    class_dirs = sorted([d for d in best_dir.iterdir() if d.is_dir()])
    print(f"\n  Directorio raíz de clases: {best_dir}")
    print(f"  Clases detectadas ({len(class_dirs)}):")
    for d in class_dirs:
        n = len(list(d.glob("*.jpg")) + list(d.glob("*.png"))
                + list(d.glob("*.jpeg")) + list(d.glob("*.bmp")))
        print(f"    {d.name}: {n} imágenes")
    return best_dir


# ── 2. DataLoaders ───────────────────────────────────────────────────────────
def build_loaders(data_dir: Path):
    MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full_ds = ImageFolder(str(data_dir), transform=val_tf)
    class_names = full_ds.classes
    num_classes = len(class_names)
    print(f"\n  ImageFolder — {num_classes} clases: {class_names}")
    print(f"  Total imágenes: {len(full_ds):,}")

    targets = np.array(full_ds.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_aug = ImageFolder(str(data_dir), transform=train_tf)
    train_loader = DataLoader(Subset(train_aug, train_idx),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(Subset(full_ds, val_idx),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")
    return train_loader, val_loader, class_names, num_classes


# ── 3. Modelo ────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet18(pretrained=True)
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ["layer3", "layer4", "fc"])
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(device)


# ── 4. Entrenamiento ─────────────────────────────────────────────────────────
def train(data_dir: Path):
    train_loader, val_loader, class_names, num_classes = build_loaders(data_dir)
    model     = build_model(num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_acc = 0.0; best_state = None; pat = 0
    history  = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    print(f"\nEntrenando {num_classes} clases | {EPOCHS} épocas máx | device={device}")
    for ep in range(1, EPOCHS + 1):
        # train
        model.train(); tl=0; tc=0; tn=0
        for X,y in train_loader:
            X,y=X.to(device),y.to(device); optimizer.zero_grad()
            out=model(X); loss=criterion(out,y); loss.backward(); optimizer.step()
            tl+=loss.item()*len(y); tc+=(out.argmax(1)==y).sum().item(); tn+=len(y)
        # val
        model.eval(); vl=0; vc=0; vn=0; vp=[]; vt=[]
        with torch.no_grad():
            for X,y in val_loader:
                X,y=X.to(device),y.to(device); out=model(X); loss=criterion(out,y)
                vl+=loss.item()*len(y); vc+=(out.argmax(1)==y).sum().item(); vn+=len(y)
                vp.extend(out.argmax(1).cpu().numpy()); vt.extend(y.cpu().numpy())
        tr_l,tr_a,va_l,va_a = tl/tn,tc/tn,vl/vn,vc/vn
        scheduler.step(va_a)
        history["train_loss"].append(tr_l); history["val_loss"].append(va_l)
        history["train_acc"].append(tr_a);  history["val_acc"].append(va_a)
        print(f"Ep {ep:3d}/{EPOCHS} | TrLoss={tr_l:.4f} TrAcc={tr_a*100:.2f}%"
              f" | VaLoss={va_l:.4f} VaAcc={va_a*100:.2f}%")
        if va_a > best_acc:
            best_acc=va_a; best_state={k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=PATIENCE: print(f"  Early stop ep {ep}"); break

    model.load_state_dict(best_state)
    vp=np.array(vp); vt=np.array(vt)
    acc  = accuracy_score(vt, vp)
    f1m  = f1_score(vt,vp,average="macro",    zero_division=0)
    f1w  = f1_score(vt,vp,average="weighted", zero_division=0)
    prec = precision_score(vt,vp,average="macro", zero_division=0)
    rec  = recall_score(vt,vp,   average="macro", zero_division=0)
    f1pc = f1_score(vt,vp,average=None,zero_division=0).tolist()
    print(f"\nFinal → Acc={acc*100:.2f}% | F1 macro={f1m:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
    print(classification_report(vt,vp,target_names=class_names,zero_division=0))

    metrics = {"accuracy":float(acc),"f1_macro":float(f1m),"f1_weighted":float(f1w),
               "precision_macro":float(prec),"recall_macro":float(rec),
               "best_val_acc":float(best_acc),"num_classes":num_classes,
               "class_names":class_names,"history":history,
               "f1_per_class":{c:float(f) for c,f in zip(class_names,f1pc)}}
    return model, class_names, num_classes, metrics


# ── 5. Guardar ───────────────────────────────────────────────────────────────
def save(model, class_names, num_classes, metrics):
    data = {"state_dict":model.state_dict(),"num_classes":num_classes,"class_names":class_names}
    for out in [MODELS_DIR, WEBAPP_DIR]:
        torch.save(data,         out/"resnet18_driver.pt")
        pickle.dump(class_names, open(out/"class_names.pkl","wb"))
        pickle.dump(metrics,     open(out/"cnn_metrics.pkl","wb"))
        print(f"  Guardado en {out}")


def main():
    print("="*60); print("MÓDULO 2 — CNN (datos reales Kaggle)"); print("="*60)
    data_dir = download_dataset()
    model, class_names, num_classes, metrics = train(data_dir)
    save(model, class_names, num_classes, metrics)
    print(f"\n[OK] CNN entrenado. {num_classes} clases: {class_names}")
    print(f"     Accuracy={metrics['accuracy']*100:.2f}% | F1={metrics['f1_macro']:.4f}")

if __name__ == "__main__":
    main()
