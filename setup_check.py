"""
Quick script to check dataset structure
"""
import os

base_dir = "C:\\Users\\Janani Nethma\\Desktop\\NeuroScan AI"

print("Checking directory structure...")
print(f"Base directory: {base_dir}\n")

# Check Training folder
train_dir = os.path.join(base_dir, "Training")
if os.path.exists(train_dir):
    print("[OK] Training folder exists")
    for class_name in ["glioma", "meningioma", "pituitary", "notumor"]:
        class_path = os.path.join(train_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"  [OK] {class_name}: {count} images")
        else:
            print(f"  [MISSING] {class_name}: folder missing")
else:
    print("[MISSING] Training folder missing")

print()

# Check Testing folder
test_dir = os.path.join(base_dir, "Testing")
if os.path.exists(test_dir):
    print("[OK] Testing folder exists")
    for class_name in ["glioma", "meningioma", "pituitary", "notumor"]:
        class_path = os.path.join(test_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"  [OK] {class_name}: {count} images")
        else:
            print(f"  [MISSING] {class_name}: folder missing")
else:
    print("[MISSING] Testing folder missing")

print("\nDataset structure check complete!")
