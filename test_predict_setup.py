"""
Quick test to verify predict_all_dataset.py can initialize properly
"""
import os
import sys
import torch

print("="*80, flush=True)
print("TESTING SETUP FOR FULL DATASET PREDICTION", flush=True)
print("="*80, flush=True)

# Test 1: Check files exist
print("\n1. Checking required files...", flush=True)
files_to_check = [
    'logs/archive/best_videomae_large_model.pth',
    'balanced_dataset_2s/train',
    'balanced_dataset_2s/val',
    'balanced_dataset_2s/metadata.csv',
    'dataset/train.csv'
]

all_exist = True
for f in files_to_check:
    exists = os.path.exists(f)
    status = "✅" if exists else "❌"
    print(f"  {status} {f}", flush=True)
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ Some required files are missing!", flush=True)
    sys.exit(1)

print("\n✅ All required files found!", flush=True)

# Test 2: Check GPU
print("\n2. Checking GPU availability...", flush=True)
if torch.cuda.is_available():
    print(f"  ✅ GPU available: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
else:
    print("  ⚠️  No GPU available, will use CPU (slower)", flush=True)

# Test 3: Import train module
print("\n3. Testing module imports...", flush=True)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_module", "train-large.py")
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["train_module"] = train_module
    spec.loader.exec_module(train_module)
    print("  ✅ train-large.py imported successfully", flush=True)
except Exception as e:
    print(f"  ❌ Failed to import train-large.py: {e}", flush=True)
    sys.exit(1)

# Test 4: Count segments
print("\n4. Counting video segments...", flush=True)
train_positive = len([f for f in os.listdir('balanced_dataset_2s/train/positive') if f.endswith(('.mp4', '.avi', '.mov'))])
train_negative = len([f for f in os.listdir('balanced_dataset_2s/train/negative') if f.endswith(('.mp4', '.avi', '.mov'))])
val_positive = len([f for f in os.listdir('balanced_dataset_2s/val/positive') if f.endswith(('.mp4', '.avi', '.mov'))])
val_negative = len([f for f in os.listdir('balanced_dataset_2s/val/negative') if f.endswith(('.mp4', '.avi', '.mov'))])

train_total = train_positive + train_negative
val_total = val_positive + val_negative
total = train_total + val_total

print(f"  Train: {train_total:,} segments ({train_positive:,} positive, {train_negative:,} negative)", flush=True)
print(f"  Val:   {val_total:,} segments ({val_positive:,} positive, {val_negative:,} negative)", flush=True)
print(f"  Total: {total:,} segments", flush=True)

# Estimate time
batch_size = 8
batches = (total + batch_size - 1) // batch_size
seconds_per_batch = 0.5  # rough estimate
estimated_minutes = (batches * seconds_per_batch) / 60
print(f"\n  Estimated time: ~{estimated_minutes:.1f} minutes (with GPU @ batch_size=8)", flush=True)

print("\n" + "="*80, flush=True)
print("✅ SETUP TEST COMPLETE - Ready to run predict_all_dataset.py", flush=True)
print("="*80, flush=True)
print("\nTo run full prediction:", flush=True)
print("  python predict_all_dataset.py", flush=True)
