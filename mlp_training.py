import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import shuffle

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =========================
# DATA LOADING
# =========================
DATA_FOLDER = "extracted_beats"

all_data = []
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_FOLDER, file))
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print("Total samples:", len(data))

X = data.iloc[:, 1:242].values
y = data.iloc[:, -1].values
y = np.where(y == "N", 0, 1)

print("Feature dimension:", X.shape[1])

X, y = shuffle(X, y, random_state=42)

# =========================
# NORMALIZATION
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

print("Scaler parameters saved.")

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL TRAINING
# =========================
model = MLPClassifier(
    hidden_layer_sizes=(16,),
    activation='relu',
    solver='adam',
    alpha=0.055,
    max_iter=1,
    warm_start=True,
    random_state=42
)

epochs = 150
train_loss = []
test_loss = []

for epoch in range(epochs):

    X_train, y_train = shuffle(X_train, y_train)
    model.fit(X_train, y_train)

    train_loss.append(model.loss_)
    y_test_prob = model.predict_proba(X_test)
    test_loss.append(log_loss(y_test, y_test_prob))

# =========================
# FLOATING EVALUATION
# =========================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

probs = model.predict_proba(X_test)[:, 1]

print("\n=== FLOATING POINT RESULTS ===")
print("\nTraining Accuracy:", accuracy_score(y_train, y_train_pred))
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

# =========================
# ⭐ CPU LATENCY MEASUREMENT
# =========================
import time
import numpy as np

print("\n=== CPU LATENCY MEASUREMENT ===")

# warm-up run (important for stable timing)
_ = model.predict(X_test[:10])

num_trials = 2000   # higher = more stable timing

sample = X_test[0].reshape(1, -1)

start = time.perf_counter()

for _ in range(num_trials):
    model.predict(sample)

end = time.perf_counter()

avg_latency_sec = (end - start) / num_trials
avg_latency_us = avg_latency_sec * 1e6

print("Average CPU inference latency per beat:")
print(f"{avg_latency_sec:.8f} seconds")
print(f"{avg_latency_us:.2f} microseconds")

# =========================
# ⭐ BEST THRESHOLD SEARCH (F1-based)
# =========================
best_f1 = 0
best_thresh = 0

for t in np.linspace(0.01, 0.9, 300):

    y_pred_t = (probs > t).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_t, average=None
    )

    f1_abnormal = f1[1]

    if f1_abnormal > best_f1:
        best_f1 = f1_abnormal
        best_thresh = t

print("\nBest probability threshold =", best_thresh)
print("Best abnormal F1 =", best_f1)

# =========================
# FIXED THRESHOLD GENERATION
# =========================
logit_thresh = math.log(best_thresh / (1 - best_thresh))
print("Logit threshold =", logit_thresh)

# =========================
# FIXED POINT WEIGHT EXPORT
# =========================
W1 = model.coefs_[0]
W2 = model.coefs_[1]
b1 = model.intercepts_[0]
b2 = model.intercepts_[1]

all_params = np.concatenate([
    W1.flatten(),
    W2.flatten(),
    b1.flatten(),
    b2.flatten(),
    X.flatten()
])

max_abs = np.max(np.abs(all_params))
print("\nMaximum absolute value:", max_abs)

integer_bits = int(np.ceil(np.log2(max_abs + 1)))
total_bits = 16
fractional_bits = total_bits - 1 - integer_bits

print(f"Using Q{integer_bits}.{fractional_bits}")

scale_factor = 2 ** fractional_bits

fixed_thresh = int(logit_thresh * scale_factor)
print("Fixed threshold =", fixed_thresh)

W1_fixed = np.round(W1 * scale_factor).astype(np.int16)
W2_fixed = np.round(W2 * scale_factor).astype(np.int16)
b1_fixed = np.round(b1 * scale_factor).astype(np.int16)
b2_fixed = np.round(b2 * scale_factor).astype(np.int16)

np.savetxt("W1_fixed.csv", W1_fixed, fmt="%d", delimiter=",")
np.savetxt("W2_fixed.csv", W2_fixed, fmt="%d", delimiter=",")
np.savetxt("b1_fixed.csv", b1_fixed, fmt="%d", delimiter=",")
np.savetxt("b2_fixed.csv", b2_fixed, fmt="%d", delimiter=",")

np.save("fractional_bits.npy", fractional_bits)

print("Fixed-point weights saved.")

# =========================
# ⭐ SAVE ONLY TEST DATA (DEPLOYMENT VALIDATION)
# =========================
os.makedirs("quantized_test_data", exist_ok=True)
os.makedirs("quantized_test_labels", exist_ok=True)

X_test_fixed = np.clip(
    np.round(X_test * scale_factor),
    -32768,
    32767
).astype(np.int16)

np.savetxt("quantized_test_data/X_test.txt",
           X_test_fixed,
           fmt="%d")

np.savetxt("quantized_test_labels/y_test.txt",
           y_test,
           fmt="%d")

print("Quantized TEST dataset saved.")

# =========================
# ⭐ ACCUMULATOR SAFETY CHECK
# =========================
W1_abs_sum = np.sum(np.abs(W1_fixed), axis=0)
max_input = np.max(np.abs(X_test_fixed))

worst_case_hidden = (max_input * W1_abs_sum) >> fractional_bits

print("Worst hidden accumulator estimate:",
      np.max(worst_case_hidden))