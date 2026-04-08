from ultralytics import YOLO

# Train model
model = YOLO("runs/pose/train/weights/last.pt")

results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640, device="mps", resume=True, save_period=10, patience=5, degrees=180, fliplr=0.5, flipud=0.25,)

model.export(format="onnx", dynamic=True, simplify=True)

# Model to onnx conversion
# model = YOLO("runs/pose/train/weights/epoch60.pt")
# model.export(format="onnx", dynamic=True, simplify=True)

# Checkpoint loading
# import torch

# try:
#     ckpt = torch.load("runs/pose/train/weights/last.pt", map_location="cpu", weights_only=False)
#     print(ckpt.get("epoch"))
# except Exception as e:
#     print(f"Error loading checkpoint: {e}")

# try:
#     ckpt = torch.load("runs/pose/train2/weights/last.pt", map_location="cpu", weights_only=False)
#     print(ckpt.get("epoch"))
# except Exception as e:
#     print(f"Error loading checkpoint: {e}")

# try:
#     ckpt = torch.load("runs/pose/train3/weights/last.pt", map_location="cpu", weights_only=False)
#     print(ckpt.get("epoch"))
# except Exception as e:
#     print(f"Error loading checkpoint: {e}")

# try:
#     ckpt = torch.load("runs/pose/train4/weights/last.pt", map_location="cpu", weights_only=False)
#     print(ckpt.get("epoch"))
# except Exception as e:
#     print(f"Error loading checkpoint: {e}")

# try:
#     ckpt = torch.load("runs/pose/train5/weights/last.pt", map_location="cpu", weights_only=False)
#     print(ckpt.get("epoch"))
# except Exception as e:
#     print(f"Error loading checkpoint: {e}")

# try:
#     ckpt = torch.load("runs/pose/train6/weights/last.pt", map_location="cpu", weights_only=False)
#     print(ckpt.get("epoch"))
# except Exception as e:
#     print(f"Error loading checkpoint: {e}")
