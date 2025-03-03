import os
import json
import time
import matplotlib.pyplot as plt

patch_sizes = [2, 4, 8, 16]
config_path = "model_configs/mlpmixer.json"

experiment_results = {}

for patch_size in patch_sizes:
    logdir = f"results/MLPMixer_patch_{patch_size}"

    print(f"\n Training MLPMixer with patch size {patch_size}...\n")

    with open(config_path, "r") as f:
        model_config = json.load(f)

    model_config["patch_size"] = patch_size

    temp_config_path = f"model_configs/temp_mlpmixer.json"
    with open(temp_config_path, "w") as f:
        json.dump(model_config, f, indent=4)

    start_time = time.time()

    os.system(f"python main.py --model mlpmixer --model_config {temp_config_path} --logdir {logdir} --epochs 50")

    total_time = time.time() - start_time

    results_file = os.path.join(logdir, "results.json")
    wait_time = 0
    while not os.path.exists(results_file) and wait_time < 300:  # Wait up to 5 minutes
        time.sleep(5)
        wait_time += 5

    with open(results_file, "r") as f:
        results_data = json.load(f)

    model_param_count = os.popen(f"python -c \"from mlpmixer import MLPMixer; import json; config=json.load(open('{temp_config_path}')); model=MLPMixer(**config); print(sum(p.numel() for p in model.parameters()))\"").read().strip()

    results_data["patch_size"] = patch_size
    results_data["total_parameters"] = int(model_param_count)
    results_data["training_time_seconds"] = total_time

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=4)

    experiment_results[patch_size] = {
        "Total Parameters": int(model_param_count),
        "Training Time (s)": total_time
    }

# Generate Patch Size vs Training Time Plot
patch_sizes_sorted = sorted(experiment_results.keys())
training_times = [experiment_results[p]["Training Time (s)"] for p in patch_sizes_sorted]

plt.figure(figsize=(8, 6))
plt.plot(patch_sizes_sorted, training_times, marker='o', linestyle='--', color='b', label="Training Time (seconds)")
plt.xlabel("Patch Size")
plt.ylabel("Training Time (seconds)")
plt.title("Effect of Patch Size on Training Time")
plt.grid(True)
plt.xticks(patch_sizes_sorted)
plt.legend()
plt.savefig("results/patch_size_vs_training_time.png")
plt.show()

# Generate Patch Size vs Model Parameters Plot
param_counts = [experiment_results[p]["Total Parameters"] for p in patch_sizes_sorted]

plt.figure(figsize=(8, 6))
plt.plot(patch_sizes_sorted, param_counts, marker='o', linestyle='--', color='r', label="Total Model Parameters")
plt.xlabel("Patch Size")
plt.ylabel("Number of Parameters")
plt.title("Effect of Patch Size on Model Size")
plt.grid(True)
plt.xticks(patch_sizes_sorted)
plt.legend()
plt.savefig("results/patch_size_vs_parameters.png")
plt.show()

# Print Summary
print("\nSummary:")
for patch, stats in experiment_results.items():
    print(
        f"Patch Size: {patch}x{patch} → Parameters: {stats['Total Parameters']}, Training Time: {stats['Training Time (s)']:.2f} sec")
