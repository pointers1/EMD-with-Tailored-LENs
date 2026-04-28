import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch_explain as te
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch_explain.logic.metrics import test_explanation, complexity
from tailor_len import get_explanation, evaluate_explanation, check_complexity, get_prediction, predict_with_explanation

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sys.setrecursionlimit(20000)  # Set recursion limit to avoid potential stack overflow

import pandas as pd

# Function to read the dictionary from the text file
def load_features_dict(input_file):
    with open(input_file, 'r') as file:
        features_dict = eval(file.read())
    return features_dict

features_dict = load_features_dict('features.txt')
    
# Create the key for the dictionary
num_features= 10  #put the appropriate number of features you want to use (5,10, 15,20, and 25 were used)
    
# Get the selected columns
selected_columns = features_dict[f"{num_features}_features"]
    
# Load the data with only the selected columns
data = pd.read_csv('malware_dataset.csv', encoding='utf8', nrows=25000, usecols=selected_columns)
#data.head()

# Split the data into features and labels
X = data.drop(columns=['malware_benign'])
X.columns = [col.replace('-', '_') for col in X.columns]
y = data['malware_benign']

# Encode the labels (benign/malware) to numerical values (0/1)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train_temp.values, dtype=torch.float32)
y_train = torch.tensor(y_train_temp, dtype=torch.int64)
X_test = torch.tensor(X_test_temp.values, dtype=torch.float32)
y_test = torch.tensor(y_test_temp, dtype=torch.int64)

# Convert labels to one-hot encoded tensors
num_classes = 2
y_train_1h = torch.nn.functional.one_hot(y_train, num_classes=num_classes).to(torch.float32)
y_test_1h = torch.nn.functional.one_hot(y_test, num_classes=num_classes).to(torch.float32)

# Create DataLoader for training data
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train_1h))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Adjust batch_size as needed

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
y_test_np = y_test_1h[:, 1].numpy()
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

# Evaluate explanation performance
print('RAW-LEN explanation performance')
raw_exp = entropy1.explain_classes(model, X_train, y_train, c_threshold=0.5, y_threshold=0., class_names=['benign', 'malware'], concept_names=X.columns, train_mask=None, val_mask=None)[0]['1']['explanation']
evaluate_explanation(raw_exp, X_test_temp, y_test_temp)
y_len = get_prediction(model, X_test)
y_exp = predict_with_explanation(raw_exp, X_test_temp)
fidelity = accuracy_score(y_len, y_exp)
complexity_count = complexity(raw_exp)
print(f'Fidelity score: {round(fidelity, 3)}')
print(f'Complexity: {complexity_count}')

print('Standard-LEN explanation performance')
len_explanations = entropy2.explain_classes(model, X_train, y_train, c_threshold=0.5, y_threshold=0, simplify=True, class_names=['benign', 'malware'], concept_names=X.columns, train_mask=None, val_mask=None)[0]['1']['explanation']
evaluate_explanation(len_explanations, X_test_temp, y_test_temp)
y_len = get_prediction(model, X_test)
y_exp = predict_with_explanation(len_explanations, X_test_temp)
fidelity = accuracy_score(y_len, y_exp)
complexity_count = complexity(len_explanations)
print(f'Fidelity score: {round(fidelity, 3)}')
print(f'Complexity: {complexity_count}')

print('Tailor-LEN explanation performance')
tailor_exp = get_explanation(model, X_train_temp, y_train_temp, X.columns)
evaluate_explanation(tailor_exp, X_test_temp, y_test_temp)
y_len = get_prediction(model, X_test)
y_exp = predict_with_explanation(tailor_exp, X_test_temp)
fidelity = accuracy_score(y_len, y_exp)
complexity_count = complexity(tailor_exp)
print(f'Fidelity score: {round(fidelity, 3)}')
print(f'Complexity: {complexity_count}')
