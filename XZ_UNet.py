import os
import glob
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"C:\Users\richa\OneDrive\Documents\College\CreateX\Testing\TrainingDataAndLabel"
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-3
testSplit = 0.2
NUM_CLASSES = 2

CHECKPOINT_BASENAME = "xz_slice_unet"


def atomic_save(obj, out_path: str):
    """
    Safer save: write to a temp file, then atomically move into place.
    Prevents corrupted checkpoints if the process dies mid-write.
    """
    tmp_path = out_path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, out_path)


# ============================================================
# Model
# ============================================================

class DoubleConv(nn.Module):
    """(Conv to BN to ReLU) 2 deep"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StudUNet(nn.Module):
    """
    Lightweight UNet for XZ slices.
    Input:  (B, 2, H, W)
    Output: (B, num_classes, H, W)
    """

    def __init__(self, in_channels=2, num_classes=2, base_ch=32):
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(in_channels, base_ch)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch, base_ch * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 2, base_ch * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_ch * 4, base_ch * 8)
        )

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_ch * 16, base_ch * 8)

        self.up2 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up0  = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.conv0 = DoubleConv(base_ch * 2, base_ch)

        # Final conv
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)         # (B, base, H, W)
        x2 = self.down1(x1)      # (B, 2*base, H/2, W/2)
        x3 = self.down2(x2)      # (B, 4*base, H/4, W/4)
        x4 = self.down3(x3)      # (B, 8*base, H/8, W/8)

        # Bottleneck
        xb = self.bottleneck(x4) # (B, 16*base, H/8, W/8)

        # Decoder with skip connections
        u3 = self.up3(xb)
        # Pad if shapes differ by 1 due to odd sizes
        if u3.shape[-2:] != x4.shape[-2:]:
            u3 = F.interpolate(u3, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        c3 = self.conv3(torch.cat([u3, x4], dim=1))

        u2 = self.up2(c3)
        if u2.shape[-2:] != x3.shape[-2:]:
            u2 = F.interpolate(u2, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        c2 = self.conv2(torch.cat([u2, x3], dim=1))

        u1 = self.up1(c2)
        if u1.shape[-2:] != x2.shape[-2:]:
            u1 = F.interpolate(u1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        c1 = self.conv1(torch.cat([u1, x2], dim=1))

        u0 = self.up0(c1)
        if u0.shape[-2:] != x1.shape[-2:]:
            u0 = F.interpolate(u0, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        c0 = self.conv0(torch.cat([u0, x1], dim=1))

        logits = self.outc(c0)
        return logits

# ============================================================
# Data utilities
# ============================================================

def list_scene_files(data_dir):
    """
    Find all feature-label pairs in DATA_DIR.

    Assumes naming like:
        <scene_id>.rawVox_2ch.npy
        <scene_id>.labels.npy
    """
    feature_files = glob.glob(os.path.join(data_dir, "*.2chVox.npy"))
    scene_pairs = []

    for fpath in feature_files:
        base = os.path.basename(fpath)
        scene_id = base.replace(".2chVox.npy", "")
        label_path = os.path.join(data_dir, f"{scene_id}.labels.npy")
        if os.path.exists(label_path):
            scene_pairs.append((fpath, label_path))
        else:
            print(f"[WARN] Label file missing for {fpath}, skipping.")

    if not scene_pairs:
        raise RuntimeError(f"No scene pairs found in {data_dir}")

    return scene_pairs


def train_val_split(scene_pairs, val_fraction=testSplit, seed=0):
    scene_pairs = list(scene_pairs)
    rng = np.random.RandomState(seed)
    rng.shuffle(scene_pairs)
    n_total = len(scene_pairs)
    n_val = max(1, int(np.round(n_total * testSplit)))
    val_pairs = scene_pairs[:n_val]
    train_pairs = scene_pairs[n_val:]
    if not train_pairs:  # fallback: at least one train
        train_pairs = val_pairs
    return train_pairs, val_pairs


def compute_normalization_stats(scene_pairs):
    """
    Compute per-channel mean and std over all features in scene_pairs.

    Each feature volume has shape (2, Z, Y, X).
    """
    sum_c = np.zeros(2, dtype=np.float64)
    sumsq_c = np.zeros(2, dtype=np.float64)
    count_c = np.zeros(2, dtype=np.float64)

    for fpath, _ in scene_pairs:
        feats = np.load(fpath)  # (2, Z, Y, X)
        if feats.ndim != 4 or feats.shape[0] != 2:
            raise ValueError(f"Unexpected feature shape in {fpath}: {feats.shape}")

        # flatten spatial dimensions
        C, Z, Y, X = feats.shape
        flat = feats.reshape(C, -1)

        sum_c += flat.sum(axis=1)
        sumsq_c += (flat ** 2).sum(axis=1)
        count_c += flat.shape[1]

    mean = sum_c / count_c
    var = sumsq_c / count_c - mean ** 2
    var = np.maximum(var, 1e-8)  # avoid negative variance due to numerical issues
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class XZSliceDataset(Dataset):
    """
    Each item is:
        input:  (2, Z, X) slice at fixed y-index
        target: (Z, X) label slice
    across all scenes and all y indices.
    """

    def __init__(self, scene_pairs, chan_mean, chan_std):
        self.scene_pairs = list(scene_pairs)
        self.chan_mean = chan_mean.reshape(2, 1, 1)  # (2,1,1)
        self.chan_std = chan_std.reshape(2, 1, 1)    # (2,1,1)

        # Build list of (feature_path, label_path, y_idx)
        self.samples = self._build_index()

    def _build_index(self):
        index = []
        for fpath, lpath in self.scene_pairs:
            labels = np.load(lpath)  # (Z, Y, X)
            if labels.ndim != 3:
                raise ValueError(f"Unexpected label shape in {lpath}: {labels.shape}")
            nz, ny, nx = labels.shape
            for y_idx in range(ny):
                index.append((fpath, lpath, y_idx))
        return index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, lpath, y_idx = self.samples[idx]

        feats = np.load(fpath)   # (2, Z, Y, X)
        labels = np.load(lpath)  # (Z, Y, X)

        # sanity check
        if feats.shape[0] != 2:
            raise ValueError(f"Expected 2 channels in {fpath}, got {feats.shape[0]}")
        if feats.shape[1:] != labels.shape:
            raise ValueError(
                f"Feature/label mismatch in {fpath}/{lpath}: "
                f"{feats.shape[1:]} vs {labels.shape}"
            )

        # slice along Y to get XZ plane at y_idx
        # feats_slice: (2, Z, X)
        feats_slice = feats[:, :, y_idx, :]  # C, Z, X
        labels_slice = labels[:, y_idx, :]   # Z, X

        # normalize per channel
        feats_norm = (feats_slice - self.chan_mean) / self.chan_std
        feats_norm = feats_norm.astype(np.float32)

        # labels to int64 (class indices)
        labels_slice = labels_slice.astype(np.int64)

        x = torch.from_numpy(feats_norm)        # (2, Z, X)
        y = torch.from_numpy(labels_slice)      # (Z, X)

        return x, y


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(DEVICE)  # (B, 2, Z, X)
        y = y.to(DEVICE)  # (B, Z, X)

        optimizer.zero_grad()
        logits = model(x)  # (B, C, Z, X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    global_TP = 0
    global_FP = 0
    global_FN = 0
    global_TN = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)               # (B, C, Z, X)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            # simple pixel accuracy
            preds = torch.argmax(logits, dim=1)  # (B, Z, X)
            correct = (preds == y).sum().item()
            total_correct += correct
            total_pixels += y.numel()

            pred_stud = (preds == 1)
            true_stud = (y == 1)

            TP = (pred_stud & true_stud).sum().item()
            FP = (pred_stud & ~true_stud).sum().item()
            FN = (~pred_stud & true_stud).sum().item()
            TN = (~pred_stud & ~true_stud).sum().item()
            global_TP += TP
            global_FP += FP
            global_FN += FN
            global_TN += TN

    avg_loss = total_loss / len(loader.dataset)
    acc = total_correct / max(1, total_pixels)
    iou_stud = global_TP / max(1, (global_TP + global_FP + global_FN))
    dice_stud = (2 * global_TP) / max(1, (2 * global_TP + global_FP + global_FN))
    precision = global_TP / max(1, (global_TP + global_FP))
    recall    = global_TP / max(1, (global_TP + global_FN))

    return avg_loss, acc, iou_stud, dice_stud, precision, recall


# ============================================================
# Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    # Ensure directory exists and is writable
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.access(DATA_DIR, os.W_OK):
        raise RuntimeError(f"DATA_DIR not writable: {DATA_DIR}")

    # 1) discover scenes
    scene_pairs = list_scene_files(DATA_DIR)
    print(f"Found {len(scene_pairs)} scenes.")

    # 2) split train/val
    train_pairs, val_pairs = train_val_split(scene_pairs, val_fraction=testSplit)
    print(f"Train scenes: {len(train_pairs)}, Val scenes: {len(val_pairs)}")

    # 3) compute normalization stats from TRAIN only
    chan_mean, chan_std = compute_normalization_stats(train_pairs)
    print(f"Channel means: {chan_mean}")
    print(f"Channel stds:  {chan_std}")

    # 4) datasets + loaders
    train_ds = XZSliceDataset(train_pairs, chan_mean, chan_std)
    val_ds   = XZSliceDataset(val_pairs,   chan_mean, chan_std)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"Train slices: {len(train_ds)}, Val slices: {len(val_ds)}")

    # 5) model, loss, optimizer

    def dice_loss(logits, targets, epsilon=1e-6):
        """
        Dice loss for stud class (class 1) in a 2-class problem.
        logits: (B, C, H, W)
        targets: (B, H, W) with values {0,1}
        """
        probs = torch.softmax(logits, dim=1)
        p1 = probs[:, 1, :, :]          # stud prob
        t1 = (targets == 1).float()     # stud mask

        intersection = (p1 * t1).sum()
        union = p1.sum() + t1.sum()

        dice = (2 * intersection + epsilon) / (union + epsilon)
        return 1.0 - dice

    model = StudUNet(in_channels=2, num_classes=NUM_CLASSES, base_ch=32).to(DEVICE)
    class_weights = torch.tensor([1.0, 5.0], device=DEVICE)  # [bg, stud]
    ce = nn.CrossEntropyLoss(weight=class_weights)

    def combined_loss(logits, targets):
        return ce(logits, targets) + dice_loss(logits, targets)

    criterion = combined_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_dice = -float("inf")
    last_ckpt = None  # will hold the latest full checkpoint

    try:
        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_iou, val_dice, val_prec, val_rec = evaluate(
                model, val_loader, criterion
            )

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
                f"IoU={val_iou:.4f} | Dice={val_dice:.4f} | "
                f"Precision={val_prec:.4f} | Recall={val_rec:.4f}"
            )

            # Build full checkpoint dict
            last_ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "chan_mean": chan_mean,
                "chan_std": chan_std,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "val_precision": val_prec,
                "val_recall": val_rec,
            }

            # 1) Save "last" checkpoint every epoch
            last_path = os.path.join(DATA_DIR, f"{CHECKPOINT_BASENAME}_last.pt")
            atomic_save(last_ckpt, last_path)
            print(f"Saved LAST checkpoint to {last_path}")

            # 2) Save "best" checkpoint based on validation Dice
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_path = os.path.join(DATA_DIR, f"{CHECKPOINT_BASENAME}_best.pt")
                atomic_save(last_ckpt, best_path)
                print(
                    f"New best Dice={best_val_dice:.4f}; saved BEST checkpoint to {best_path}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (KeyboardInterrupt).")
        if last_ckpt is not None:
            interrupt_path = os.path.join(
                DATA_DIR, f"{CHECKPOINT_BASENAME}_interrupt.pt"
            )
            atomic_save(last_ckpt, interrupt_path)
            print(f"Saved INTERRUPT checkpoint to {interrupt_path}")
        return

    except Exception as e:
        print(f"\n[ERROR] Exception during training: {e}")
        if last_ckpt is not None:
            error_path = os.path.join(DATA_DIR, f"{CHECKPOINT_BASENAME}_error.pt")
            atomic_save(last_ckpt, error_path)
            print(f"Saved ERROR checkpoint to {error_path}")
        raise  # re-raise so you still see the traceback

    # 3) Save a final checkpoint at the end (could be same as last epoch)
    if last_ckpt is None:
        # Fallback: at least save model weights
        final_path = os.path.join(DATA_DIR, f"{CHECKPOINT_BASENAME}_final.pt")
        atomic_save({"model_state_dict": model.state_dict()}, final_path)
    else:
        final_path = os.path.join(DATA_DIR, f"{CHECKPOINT_BASENAME}_final.pt")
        atomic_save(last_ckpt, final_path)

    print(f"Saved FINAL checkpoint to {final_path}")



if __name__ == "__main__":
    main()
