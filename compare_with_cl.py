import sys
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import torch_explain as te
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.setrecursionlimit(20000)  # Set recursion limit to avoid potential stack overflow

# Load dataset
data = pd.read_csv('malware_dataset.csv', encoding='utf8', nrows=5000)
X = data.drop(columns=['malware_benign'])
X.columns = [col.replace('-', '_') for col in X.columns]
y = data['malware_benign']

# Encode the labels (benign/malware) to numerical values (0/1)
le = LabelEncoder()
y = le.fit_transform(y)

# Convert NumPy arrays to PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

# Convert labels to one-hot encoded tensors
y_1h = torch.nn.functional.one_hot(y, num_classes=2).to(torch.float32)

# Initialize KFold with 5 folds
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store metrics for each fold
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
fpr_list = []

# Iterate over folds
for train_index, test_index in kf.split(X, y):
    # Split the data into training and test sets for the current fold
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Convert labels to one-hot encoded tensors
    y_train_fold_1h = torch.nn.functional.one_hot(y_train_fold, num_classes=2).to(torch.float32)
    y_test_fold_1h = torch.nn.functional.one_hot(y_test_fold, num_classes=2).to(torch.float32)

    # Create a TensorDataset and DataLoader for training data
    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold_1h)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=64, shuffle=True)

    # Define the model architecture
    layers = [
        te.nn.EntropyLinear(X.shape[1], 256, n_classes=y_1h.shape[1]),
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
    for epoch in range(160):
        for batch_X, batch_y in train_loader_fold:
            optimizer.zero_grad()
            y_pred = model(batch_X).squeeze(-1)
            loss = loss_form(y_pred, batch_y) + 0.00001 * te.nn.functional.entropy_logic_loss(model)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set for the current fold
    model.eval()
    with torch.no_grad():
        y_pred_test_fold = model(X_test_fold).squeeze(-1)
        predicted_labels_fold = (torch.sigmoid(y_pred_test_fold) >= 0.5).numpy().astype(int)

    # Convert one-hot encoded tensors back to binary labels
    y_test_fold_binary = torch.argmax(y_test_fold_1h, dim=1).numpy()

    # Calculate evaluation metrics for the current fold
    accuracy_fold = accuracy_score(y_test_fold_binary, predicted_labels_fold)
    precision_fold = precision_score(y_test_fold_binary, predicted_labels_fold)
    recall_fold = recall_score(y_test_fold_binary, predicted_labels_fold)
    f1_fold = f1_score(y_test_fold_binary, predicted_labels_fold)
    conf_matrix_fold = confusion_matrix(y_test_fold_binary, predicted_labels_fold)

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for the current fold
    tn_fold, fp_fold, fn_fold, tp_fold = conf_matrix_fold.ravel()
    tpr_fold = tp_fold / (tp_fold + fn_fold)
    fpr_fold = fp_fold / (fp_fold + tn_fold)

    # Append metrics to lists
    accuracy_list.append(accuracy_fold)
    precision_list.append(precision_fold)
    recall_list.append(recall_fold)
    f1_list.append(f1_fold)
    fpr_list.append(fpr_fold)

# Calculate mean and standard deviation of metrics across folds
final_accuracy_mean = np.mean(accuracy_list)
final_precision_mean = np.mean(precision_list)
final_recall_mean = np.mean(recall_list)
final_f1_mean = np.mean(f1_list)
final_fpr_mean = np.mean(fpr_list)

final_accuracy_std = np.std(accuracy_list)
final_precision_std = np.std(precision_list)
final_recall_std = np.std(recall_list)
final_f1_std = np.std(f1_list)
final_fpr_std = np.std(fpr_list)

# Print metrics with standard deviation
print("\nAverage Metrics across Folds:")
print(f"Mean Accuracy: {final_accuracy_mean:.3f} ± {final_accuracy_std:.3f}")
print(f"Mean Precision: {final_precision_mean:.3f} ± {final_precision_std:.3f}")
print(f"Mean Recall: {final_recall_mean:.3f} ± {final_recall_std:.3f}")
print(f"Mean F1 Score: {final_f1_mean:.3f} ± {final_f1_std:.3f}")
print(f"Mean FPR: {final_fpr_mean:.3f} ± {final_fpr_std:.3f}")
