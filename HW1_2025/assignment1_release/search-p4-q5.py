import os
import json
import random
import numpy as np

# Define hyperparameter search space
search_space = {
    "lr": [10 ** np.random.uniform(-5, -2) for _ in range(100)],
    "optimizer": ["momentum", "adam", "adamw"],
    "batch_size": [32, 64, 128, 256],
    "weight_decay": [0, 1e-4, 1e-3, 1e-2]
}

num_runs = 40
best_run = None
best_validation_acc = 0.0
experiment_results = {}

# Load previous results if available (resume training)
results_file_path = "results/resnet18_random_search_results.json"
if os.path.exists(results_file_path):
    with open(results_file_path, "r") as f:
        experiment_results = json.load(f)
        print("✅ Loaded previous results.")

# Perform Random Search
for trial in range(num_runs):
    # Randomly sample hyperparameters with smart sampling
    lr = random.choice(search_space["lr"])
    optimizer = random.choice(search_space["optimizer"])
    batch_size = random.choice(search_space["batch_size"])
    weight_decay = random.choice(search_space["weight_decay"])

    logdir = f"results/random_search_trial_{trial}_lr{lr:.5f}_opt{optimizer}_bs{batch_size}_wd{weight_decay}"

    print(f"\nRunning Random Search Trial {trial + 1}/{num_runs}...")
    print(
        f"Hyperparameters: LR={lr:.5f}, Optimizer={optimizer}, Batch Size={batch_size}, Weight Decay={weight_decay}")

    # Run training with early stopping
    os.system(
        f"python main.py --model resnet18 --model_config configs/resnet18.json --logdir {logdir} --epochs 100 --optimizer {optimizer} --lr {lr} --batch_size {batch_size} --weight_decay {weight_decay}")
    # Wait for `results.json` before proceeding
    results_file = os.path.join(logdir, "results.json")

    # Read results.json
    with open(results_file, "r") as f:
        results_data = json.load(f)

    # Extract validation accuracy
    validation_acc = max(results_data["valid_accs"])

    # Store results
    experiment_results[f"Trial {trial}"] = {
        "lr": lr,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "Train Accuracy": max(results_data["train_accs"]),
        "Validation Accuracy": validation_acc,
        "Epochs Trained": len(results_data["valid_accs"])
    }

    # Dynamically update the best model if this run is the best so far
    if validation_acc > best_validation_acc:
        best_validation_acc = validation_acc
        best_run = experiment_results[f"Trial {trial}"]
        print(f"\n🏆 New Best Model Found! Validation Accuracy: {validation_acc:.3f}")

        # Save the best model configuration dynamically
        with open("results/best_model.json", "w") as f:
            json.dump(best_run, f, indent=4)

    # Save intermediate results after each run
    with open(results_file_path, "w") as f:
        json.dump(experiment_results, f, indent=4)

# Print Summary
print("\nSummary:")
for trial, stats in experiment_results.items():
    print(
        f"{trial}: LR={stats['lr']:.5f}, Optimizer={stats['optimizer']}, Batch Size={stats['batch_size']}, "
        f"WD={stats['weight_decay']} → Train Acc: {stats['Train Accuracy']:.3f}, "
        f"Valid Acc: {stats['Validation Accuracy']:.3f}, Time: {stats['Training Time (s)']:.2f}s, "
        f"Epochs Trained: {stats['Epochs Trained']}")

# Print Best Model Details
print("\nBest Model Found:")
print(json.dumps(best_run, indent=4))

print("\nDone")
