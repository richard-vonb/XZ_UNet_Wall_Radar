import os
import glob
import csv
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import onnx

print("PyTorch version:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

##############################  MODELLL ########################
class Small3DUNet(nn.Module):
    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_ch, 3, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(),
            nn.Conv3d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, 3, padding=1),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(),
            nn.Conv3d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch, 3, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(),
            nn.Conv3d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(),
        )

        self.out_conv = nn.Conv3d(base_ch, 1, 1)  # logits

    def forward(self, x):   # x: (B,1,D,H,W)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        u1 = self.up1(e2)
        d2, h2, w2 = u1.shape[-3:]
        e1_divis = e1[..., :d2, :h2, :w2]
        cat1 = torch.cat([u1, e1_divis], dim=1)
        d1 = self.dec1(cat1)
        logits = self.out_conv(d1)
        return logits  # (B,1,D,H,W)

DATA_DIR = "TrainingDataAndLabel"
TRAIN_SPLIT = 0.75

BATCH_SIZE = 1
NUM_EPOCHS = 10 #20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
THRESH = 0.5

LOG_CSV = "training_log.csv"
CONFUSION_TXT = "confusion_matrix.txt"
MODEL_ONNX_PATH = "small3dunet.onnx"
MODEL_PTH_PATH = "small3dunet_final.pth"

RANDOM_SEED = 42

###################### MODEL  ##################################################


#  Get dataset 
class VolumeDataset(Dataset):

    def __init__(self, file_pairs):
        super().__init__()
        self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        scan_path, label_path = self.file_pairs[idx]
        scan = np.load(scan_path)          # shape (D,H,W) or (nz,ny,nx)
        labels = np.load(label_path)       # shape (D,H,W)

        # Ensure types
        scan = scan.astype(np.float32)
        labels = labels.astype(np.uint8)

        # Add channel dimension: (1,D,H,W)
        scan_t = torch.from_numpy(scan)[None, ...]   # (1,D,H,W)
        labels_t = torch.from_numpy(labels)          # (D,H,W)

        return scan_t, labels_t



def find_file_pairs(data_dir):
    scan_files = sorted(glob.glob(os.path.join(data_dir, "*.sdf.npy")))
    file_pairs = []

    for scan_path in scan_files:
        base = os.path.basename(scan_path)
        scene_id = base.replace(".sdf.npy", "")  

        label_path = os.path.join(data_dir, f"{scene_id}.labels.npy")

        if os.path.exists(label_path):
            file_pairs.append((scan_path, label_path))
        else:
            print(f"WARNING: No labels found for {scan_path}, skipping.")

    return file_pairs


def split_train_test(file_pairs, train_split=0.75, seed=42):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(file_pairs))
    rng.shuffle(indices)
    split_idx = int(len(indices) * train_split)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    train_pairs = [file_pairs[i] for i in train_idx]
    test_pairs = [file_pairs[i] for i in test_idx]
    return train_pairs, test_pairs


def compute_confusion(pred_probs, labels, thresh=0.5):
    """
    pred_probs: torch.Tensor, (B,1,D,H,W) after sigmoid
    labels:     torch.Tensor, (B,D,H,W) with 0/1
    """
    with torch.no_grad():
        preds = (pred_probs >= thresh).long()
        labels = labels.long()

        preds = preds.view(-1)
        labels = labels.view(-1)

        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

    return tp, fp, tn, fn

def estimate_pos_weight(loader, device):
    total_pos = 0
    total_neg = 0
    with torch.no_grad():
        for _, labels in loader:
            labels = labels.to(device)
            total_pos += (labels == 1).sum().item()
            total_neg += (labels == 0).sum().item()
    pos_weight_value = total_neg / max(total_pos, 1)
    print(f"Estimated pos_weight: {pos_weight_value:.4f} (neg:pos = {total_neg}:{total_pos})")
    return torch.tensor([pos_weight_value], device=device)




#  TRAINING 

def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # criterion = nn.BCEWithLogitsLoss()

    for scans, labels in loader:
        scans = scans.to(device)           # (B,1,D,H,W)
        labels = labels.to(device).float() # (B,D,H,W)

        optimizer.zero_grad()
        logits = model(scans)              # (B,1,D,H,W)
        logits = logits.squeeze(1)         # (B,D,H,W)

        _, Dp, Hp, Wp = logits.shape
        labels_c = labels[..., :Dp, :Hp, :Wp]

        loss = criterion(logits, labels_c)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def eval_model(model, loader, device, criterion, thresh=0.5):
    model.eval()
    # criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    num_batches = 0

    total_tp = total_fp = total_tn = total_fn = 0

    with torch.no_grad():
        for scans, labels in loader:
            scans = scans.to(device)
            labels = labels.to(device).float()

            logits = model(scans)          # (B,1,D,H,W)
            logits_squeezed = logits.squeeze(1)  # (B,D,H,W)

            _, Dp, Hp, Wp = logits_squeezed.shape
            labels_c = labels[..., :Dp, :Hp, :Wp]

            loss = criterion(logits_squeezed, labels_c)
            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits)  # (B,1,D,H,W)
            tp, fp, tn, fn = compute_confusion(probs, labels_c, thresh=thresh)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, (total_tp, total_fp, total_tn, total_fn)


def save_confusion_matrix(tp, fp, tn, fn, path):
    eps = 1e-9
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, eps)

    with open(path, "w") as f:
        f.write("Confusion Matrix (binary stud segmentation)\n")
        f.write(f"TP: {tp}\n")
        f.write(f"FP: {fp}\n")
        f.write(f"TN: {tn}\n")
        f.write(f"FN: {fn}\n\n")
        f.write(f"Accuracy:  {accuracy:.6f}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall:    {recall:.6f}\n")
        f.write(f"F1-score:  {f1:.6f}\n")


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    file_pairs = find_file_pairs(DATA_DIR)
    if not file_pairs:
        raise RuntimeError(f"No *_scan.npy / *_labels.npy pairs found in {DATA_DIR}")

    train_pairs, test_pairs = split_train_test(file_pairs, TRAIN_SPLIT, RANDOM_SEED)
    print(f"Train scenes: {len(train_pairs)}, Test scenes: {len(test_pairs)}")

    train_dataset = VolumeDataset(train_pairs)
    test_dataset = VolumeDataset(test_pairs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    pos_weight = estimate_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 2) Model, optimizer
    model = Small3DUNet(in_channels=1, base_ch=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 3) CSV logging
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "tp", "fp", "tn", "fn"])

    # 4) Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        test_loss, (tp, fp, tn, fn) = eval_model(model, test_loader, device, criterion, thresh=THRESH)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Test loss: {test_loss:.4f} | "
            f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}"
        )

        # Append to CSV
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, test_loss, tp, fp, tn, fn])

    # 5) Final evaluation & confusion matrix
    final_loss, (tp, fp, tn, fn) = eval_model(model, test_loader, device, criterion, thresh=THRESH)
    print(f"Final test loss: {final_loss:.4f}")
    save_confusion_matrix(tp, fp, tn, fn, CONFUSION_TXT)

    # 6) Save final PyTorch model
    torch.save(model.state_dict(), MODEL_PTH_PATH)
    print(f"Saved PyTorch model to {MODEL_PTH_PATH}")

    # 7) Export to ONNX
    model.eval()
    # Take one example to get shape
    example_scan, _ = test_dataset[0]   # (1,D,H,W) tensor
    example_scan = example_scan.unsqueeze(0).to(device)  # (B=1,1,D,H,W)

    torch.onnx.export(
        model,
        example_scan,
        MODEL_ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    print(f"Saved ONNX model to {MODEL_ONNX_PATH}")


if __name__ == "__main__":
    main()
