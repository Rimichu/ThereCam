import pandas as pd
import matplotlib.pyplot as plt

# 1. Read the CSV file
# Replace 'results.csv' with your actual file path
df = pd.read_csv('runs/pose/train/results.csv')

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# 2. Extract Data
epochs = df['epoch']
train_loss = df['train/pose_loss']
val_loss = df['val/pose_loss']

# 3. Create the plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='red', linestyle='--', linewidth=2)

# 4. Customize the plot
plt.title('Training and Validation Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 5. Show or Save the plot
plt.show()
# plt.savefig('loss_curve.png') # Uncomment to save the plot