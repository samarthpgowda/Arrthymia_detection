import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================
# LOAD QUANTIZED TEST DATA
# ============================
X_fixed = np.loadtxt("quantized_test_data/X_test.txt").astype(np.int16)
y = np.loadtxt("quantized_test_labels/y_test.txt").astype(int)

print("Total test samples:", len(y))

# ============================
# LOAD FIXED WEIGHTS
# ============================
W1_fixed = np.loadtxt("W1_fixed.csv", delimiter=",").astype(np.int16)
W2_fixed = np.loadtxt("W2_fixed.csv", delimiter=",").astype(np.int16)
b1_fixed = np.loadtxt("b1_fixed.csv", delimiter=",").astype(np.int16)
b2_fixed = np.loadtxt("b2_fixed.csv", delimiter=",").astype(np.int16)

fractional_bits = int(np.load("fractional_bits.npy"))

input_size = W1_fixed.shape[0]
hidden_size = W1_fixed.shape[1]

# ⭐ use new calibrated threshold
OUTPUT_THRESHOLD = -1235

# ============================
# FIXED INFERENCE
# ============================
def fixed_point_inference(x):

    hidden = np.zeros(hidden_size, dtype=np.int16)

    for j in range(hidden_size):

        acc = np.int64(0)

        for i in range(input_size):
            acc += (np.int64(x[i]) * np.int64(W1_fixed[i][j])) >> fractional_bits

        acc += np.int64(b1_fixed[j])

        if acc < 0:
            acc = 0

        hidden[j] = np.int16(acc)

    acc_out = np.int64(0)

    for j in range(hidden_size):
        acc_out += (np.int64(hidden[j]) * np.int64(W2_fixed[j])) >> fractional_bits

    acc_out += np.int64(b2_fixed)

    return 1 if acc_out > OUTPUT_THRESHOLD else 0

# ============================
# RUN TEST
# ============================
preds = [fixed_point_inference(x) for x in X_fixed]

print("\n===== FIXED POINT TEST RESULTS =====")

print("Accuracy:", accuracy_score(y, preds))
print("\nConfusion Matrix:\n", confusion_matrix(y, preds))
print("\nClassification Report:\n", classification_report(y, preds))