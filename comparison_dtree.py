import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import torch_explain as te
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.setrecursionlimit(20000)  # Set recursion limit to avoid potential stack overflow

# Number of features to use
num_features = 25  # Adjust the number of features as needed  (25, 50, 75, 100 and 200 were used)

# Load training and testing datasets
train_X = pd.read_csv(f'dtree_data/{num_features}/80p_ember2018-1-5-svec_v2_wv_cls_s42_weka_ig_sel_{num_features}atr.csv', encoding='utf8')
test_X = pd.read_csv(f'dtree_data/{num_features}/20p_ember2018-1-5-svec_v2_wv_cls_s42_weka_ig_sel_{num_features}atr.csv', encoding='utf8')

# Prepare the training and testing data
X_train_temp = train_X.drop(columns=['malware_benign'])
y_train_temp = train_X['malware_benign']
train_X.columns = [col.replace('-', '_') for col in train_X.columns]

X_test_temp = test_X.drop(columns=['malware_benign'])
y_test_temp = test_X['malware_benign']
test_X.columns = [col.replace('-', '_') for col in test_X.columns]

# Encode the labels (benign/malware) to numerical values (0/1)
le = LabelEncoder()
y_train_temp = le.fit_transform(y_train_temp)
y_test_temp = le.fit_transform(y_test_temp)

# Convert NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train_temp.values, dtype=torch.float32)
y_train = torch.tensor(y_train_temp, dtype=torch.int64)
X_test = torch.tensor(X_test_temp.values, dtype=torch.float32)
y_test = torch.tensor(y_test_temp, dtype=torch.int64)

# Convert labels to one-hot encoded tensors
num_classes = 2
y_train_1h = torch.nn.functional.one_hot(y_train, num_classes=num_classes).to(torch.float32)
y_test_1h = torch.nn.functional.one_hot(y_test, num_classes=num_classes).to(torch.float32)

# Create a TensorDataset from X_train (features) and y_train_1h (labels)
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train_1h))

# Create a DataLoader to batch and shuffle the data during training
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the model architecture
layers = [
    te.nn.EntropyLinear(X_train.shape[1], 256, n_classes=y_train_1h.shape[1]),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, 128),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(64, 1),
]
model = torch.nn.Sequential(*layers)

# Compile the model
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
loss_form = torch.nn.BCEWithLogitsLoss()

# Train the model
model.train()
for epoch in range(150):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_X).squeeze(-1)
        loss = loss_form(y_pred, batch_y) + 0.00001 * te.nn.functional.entropy_logic_loss(model)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test).squeeze(-1)
    predicted_labels = torch.round(torch.sigmoid(y_pred_test)).detach().numpy()

# Set a custom threshold for predictions
threshold = 0.5

# Convert one-hot encoded tensors back to binary labels for evaluation
y_test_np = y_test_1h[:, 1].numpy()

# Convert predicted probabilities to binary labels using the custom threshold
predicted_labels_binary_custom = (torch.sigmoid(y_pred_test).numpy() >= threshold).astype(int)

# Calculate evaluation metrics with the custom threshold
accuracy_custom = accuracy_score(y_test_np, predicted_labels_binary_custom)
precision_custom = precision_score(y_test_np, predicted_labels_binary_custom)
recall_custom = recall_score(y_test_np, predicted_labels_binary_custom)
f1_custom = f1_score(y_test_np, predicted_labels_binary_custom)
conf_matrix_custom = confusion_matrix(y_test_np, predicted_labels_binary_custom)

# Calculate True Positive Rate (TPR) and False Positive Rate (FPR) with the custom threshold
tn_custom, fp_custom, fn_custom, tp_custom = conf_matrix_custom.ravel()
tpr_custom = tp_custom / (tp_custom + fn_custom)
fpr_custom = fp_custom / (fp_custom + tn_custom)

# Print model's performance
print("LEN's performance")
print("Accuracy:", accuracy_custom)
print("Precision:", precision_custom)
print("Recall:", recall_custom)
print("F1 Score:", f1_custom)
print("True Positive Rate (TPR):", tpr_custom)
print("False Positive Rate (FPR):", fpr_custom)
