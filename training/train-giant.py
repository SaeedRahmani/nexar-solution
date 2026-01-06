"""
VideoMAEv2-Giant Binary Classification Training Script (Reproducible Version)
--------------------------------------------------------------------------
Key improvements over the original:
1. Global CONFIG dict centralises *all* tunable hyper‑parameters.
2. Deterministic execution via `set_seed()` + `worker_init_fn()`; the *same* seed governs:
   • Dataset splitting / shuffling  • NumPy / Python / Torch RNGs  • CUDA RNGs.
3. Training/validation flow, model architecture, and logging remain **bit‑for‑bit identical**
   to the user‑provided script except for the reproducibility hooks.

Important:  any change to CONFIG is automatically propagated throughout the pipeline –
no manual edits elsewhere are required.
"""
import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------ CONFIG ----------------------------------
CONFIG = dict(
    # Reproducibility
    seed=42,

    # Data
    img_size=224,
    frame_count=16,
    train_root=os.path.join(PROJECT_ROOT, "balanced_dataset_2s/train"),
    val_root=os.path.join(PROJECT_ROOT, "balanced_dataset_2s/val"),
    train_batch_size=3,
    val_batch_size=3,
    num_workers_train=6,
    num_workers_val=4,

    # Optimisation
    learning_rate=1e-5,
    weight_decay=1e-4,
    epochs=3,
    accumulation_steps=8,
    scheduler_T_max=2,
    scheduler_eta_min=1e-6,
    scheduler_step_size=600,   # LR step every N global batches

    # Temperature scaling
    use_temperature_scaling=True,
    temperature=2.0,

    # Model‑specific
    num_classes=2,
    pretrained_backbone=True,
    drop_path_rate=0.1,
)
# ------------------------------------------------------------------------

import os, cv2, random, time, datetime, logging, psutil
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------------- Reproducibility -----------------------------

def set_seed(seed: int) -> None:
    """Set seed for Python / NumPy / PyTorch (+ CUDA)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])

# Ensure DataLoader workers inherit different but deterministic seeds
_def_worker_seed = CONFIG["seed"]

def worker_init_fn(worker_id: int):
    ws = _def_worker_seed + worker_id
    random.seed(ws)
    np.random.seed(ws)
    torch.manual_seed(ws)

# ------------------------------- Logger ----------------------------------

def setup_logger() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger()

# ----------------------------- Device ------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ----------------------------- Dataset -----------------------------------

class VideoDataset(Dataset):
    """Binary video dataset (positive / negative) for VideoMAEv2."""
    def __init__(self, root_dir: str, transform=None, frame_count: int = 16):
        self.classes = ["negative", "positive"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        self.transform = transform
        self.frame_count = frame_count

        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset path {root_dir} not found.")

        for cls in self.classes:
            cdir = os.path.join(root_dir, cls)
            if not os.path.exists(cdir):
                raise ValueError(f"Class path {cdir} not found.")
            for vid in os.listdir(cdir):
                if vid.lower().endswith((".mp4", ".avi", ".mov")):
                    self.samples.append((os.path.join(cdir, vid), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vpath, label = self.samples[idx]
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {vpath}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._get_frame_indices(total)

        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = np.zeros((CONFIG["img_size"], CONFIG["img_size"], 3), np.uint8)
            frames.append(frame)
        cap.release()

        # Pad (rare)
        while len(frames) < self.frame_count:
            frames.append(frames[-1])

        # Deterministic transform per sample (seed = global_seed + idx)
        transformed = []
        if self.transform:
            rseed = CONFIG["seed"] + idx
            random.seed(rseed); np.random.seed(rseed)
            for f in frames:
                transformed.append(self.transform(image=f)["image"])
        else:
            transformed = [ToTensorV2()(image=f)["image"] for f in frames]

        vid_tensor = torch.stack(transformed)  # (T, C, H, W)
        return vid_tensor, torch.tensor(label)

    def _get_frame_indices(self, total):
        if total >= self.frame_count:
            step = total / self.frame_count
            return [int(i * step) for i in range(self.frame_count)]
        else:
            return list(range(total)) + [total-1]*(self.frame_count-total)

# ------------------------- Transforms ------------------------------------

IMG_SIZE = CONFIG["img_size"]
FRAME_COUNT = CONFIG["frame_count"]

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

# ---------------------------- Model --------------------------------------

class VideoMAEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-giant", trust_remote_code=True)

        # Apply drop_path_rate if available
        if hasattr(cfg, "model_config") and isinstance(cfg.model_config, dict):
            cfg.model_config["drop_path_rate"] = CONFIG["drop_path_rate"]
        elif hasattr(cfg, "drop_path_rate"):
            cfg.drop_path_rate = CONFIG["drop_path_rate"]

        if CONFIG["pretrained_backbone"]:
            self.backbone = AutoModel.from_pretrained(
                "OpenGVLab/VideoMAEv2-giant", config=cfg, trust_remote_code=True
            ).to(device)
        else:
            self.backbone = AutoModel.from_config(cfg, trust_remote_code=True).to(device)

        with torch.no_grad():
            dummy = torch.rand(1, 3, FRAME_COUNT, IMG_SIZE, IMG_SIZE).to(device)
            feat_dim = self.backbone(pixel_values=dummy).shape[-1]
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(feat_dim, CONFIG["num_classes"]).to(device)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).float().to(device)  # (B,C,T,H,W)
        feats = self.backbone(pixel_values=x)
        feats = self.dropout(feats)
        return self.classifier(feats)

# ------------------------- Helper functions ------------------------------

def smooth_labels(labels, smoothing=0.1):
    n = CONFIG["num_classes"]
    oh = nn.functional.one_hot(labels, n).float()
    return oh*(1-smoothing) + smoothing/n

def apply_temperature_scaling(logits):
    if CONFIG["use_temperature_scaling"]:
        return logits / CONFIG["temperature"]
    return logits

def get_memory_usage():
    proc = psutil.Process(os.getpid())
    ram = proc.memory_info().rss/1024**2
    if torch.cuda.is_available():
        gpu = torch.cuda.memory_allocated()/1024**2
        return f"RAM: {ram:.1f} MB | GPU: {gpu:.1f} MB"
    return f"RAM: {ram:.1f} MB"

# ------------------------------ Validate ---------------------------------

def validate(model, loader, criterion):
    model.eval(); val_loss = correct = total = 0
    with torch.no_grad():
        for vid, lab in loader:
            vid, lab = vid.to(device), lab.to(device)
            with autocast():
                out = model(vid)
                loss = criterion(apply_temperature_scaling(out), lab)
            val_loss += loss.item()*vid.size(0)
            pred = out.argmax(1)
            correct += pred.eq(lab).sum().item()
            total += lab.size(0)
    return val_loss/len(loader.dataset), (correct/total)*100

# ------------------------------- Train -----------------------------------

def train():
    logger.info("="*60)
    logger.info("Begin training VideoMAEv2‑Giant (deterministic mode)")
    logger.info("="*60)

    model = VideoMAEClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    scheduler = CosineAnnealingLR(optimiser, T_max=CONFIG["scheduler_T_max"],
                                  eta_min=CONFIG["scheduler_eta_min"])

    scaler = GradScaler()

    # Data
    train_ds = VideoDataset(CONFIG["train_root"], train_transform, FRAME_COUNT)
    val_ds   = VideoDataset(CONFIG["val_root"],   val_transform,   FRAME_COUNT)

    train_dl = DataLoader(train_ds, batch_size=CONFIG["train_batch_size"], shuffle=True,
                          num_workers=CONFIG["num_workers_train"], pin_memory=True,
                          worker_init_fn=worker_init_fn)
    val_dl   = DataLoader(val_ds,   batch_size=CONFIG["val_batch_size"],   shuffle=False,
                          num_workers=CONFIG["num_workers_val"], pin_memory=True,
                          worker_init_fn=worker_init_fn)

    best_acc, global_batch = 0.0, 0
    logger.info(f"Train/Val samples: {len(train_ds)}/{len(val_ds)}")

    for epoch in range(CONFIG["epochs"]):
        model.train(); train_loss = 0; epoch_start = time.time()
        optimiser.zero_grad()

        logger.info(f"Epoch {epoch+1:02d}/{CONFIG['epochs']}")
        for b, (vid, lab) in enumerate(train_dl):
            vid, lab = vid.to(device), lab.to(device)
            batch_start = time.time()
            with autocast():
                out = model(vid)
                loss = criterion(apply_temperature_scaling(out), lab) / CONFIG["accumulation_steps"]
            scaler.scale(loss).backward()
            train_loss += loss.item()*CONFIG["accumulation_steps"]*vid.size(0)

            if (b+1) % CONFIG["accumulation_steps"] == 0:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimiser); scaler.update(); optimiser.zero_grad()

            global_batch += 1
            if global_batch % CONFIG["scheduler_step_size"] == 0:
                scheduler.step()
                logger.info(f"LR stepped @ batch {global_batch}: {scheduler.get_last_lr()[0]:.2e}")

            if b % 10 == 0:
                elapsed = time.time() - batch_start
                curr_lr = optimiser.param_groups[0]['lr']
                avg_loss = train_loss/((b+1)*vid.size(0))
                logger.info(f"B{b:04d}/{len(train_dl)} | Loss {loss.item()*CONFIG['accumulation_steps']:.4f} "
                            f"Avg {avg_loss:.4f} | LR {curr_lr:.2e} | {elapsed:.2f}s | {get_memory_usage()}")

            if global_batch % CONFIG["scheduler_step_size"] == 0:
                scheduler.step()

        # Epoch end validation
        vloss, vacc = validate(model, val_dl, criterion)
        tloss = train_loss/len(train_ds)
        logger.info(f"Epoch {epoch+1} done in {time.time()-epoch_start:.1f}s | TrainLoss {tloss:.4f} | "
                    f"ValLoss {vloss:.4f} | ValAcc {vacc:.2f}% | Best {best_acc:.2f}% | LR {scheduler.get_last_lr()[0]:.2e}")

        if vacc > best_acc:
            best_acc = vacc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimiser': optimiser.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_acc': vacc,
                'config': CONFIG,
            }, "best_videomaev2_giant_model.pth")
            logger.info(f"Saved new best model (acc={vacc:.2f}%)")
        logger.info("-"*60)

    logger.info("Training complete. Best Val Acc: %.2f%%" % best_acc)

# -------------------------------------------------------------------------
if __name__ == "__main__":
    train()
