from pynq import Overlay, allocate
import numpy as np
import time

# Load overlay
overlay = Overlay("final_mlp.bit")
mlp = overlay.mlp_0

# Load preprocessing parameters
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")

FRAC_BITS = int(np.load("fractional_bits.npy"))
scale_factor = 1 << FRAC_BITS

# Input file
file_path = "200_beats_data.txt"

normal = 0
abnormal = 0

print(f"\nProcessing file: {file_path}")

# Allocate buffer once (IMPORTANT)
input_buffer = allocate(shape=(241,), dtype=np.int32)

with open(file_path, "r") as f:
    for line_num, line in enumerate(f):

        # Read raw data
        raw = np.array(list(map(int, line.strip().split())))

        if len(raw) != 241:
            continue

        # -----------------------------
        # Preprocessing (MATCH TRAINING)
        # -----------------------------
        x_scaled = (raw - mean) / scale
        x_fixed = np.round(x_scaled * scale_factor).astype(np.int32)

        # Load into buffer
        input_buffer[:] = x_fixed

        # Flush cache (CRITICAL)
        input_buffer.flush()

        # -----------------------------
        # Send to FPGA
        # -----------------------------
        mlp.write(0x10, input_buffer.physical_address)

        # Start IP
        mlp.write(0x00, 1)

        # Wait for DONE
        while (mlp.read(0x00) & 0x2) == 0:
            pass

        # Small delay (stability)
        time.sleep(0.0001)

        # Read result
        result = mlp.read(0x18)

        # Debug output
        print(f"Beat {line_num} → Output: {result}")

        # Classification
        if result != 0:
            abnormal += 1
        else:
            normal += 1


# -----------------------------
# Final Result
# -----------------------------
total = normal + abnormal

if total == 0:
    print("No valid data found")
else:
    abnormal_percent = (abnormal / total) * 100

    print("\n--- FINAL RESULT ---")
    print(f"Total Beats: {total}")
    print(f"Normal: {normal}")
    print(f"Abnormal: {abnormal}")
    print(f"Abnormal %: {abnormal_percent:.2f}%")

    if abnormal_percent > 5:
        print("⚠️ ARRHYTHMIA DETECTED")
    else:
        print("✅ NORMAL ECG")