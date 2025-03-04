import os
import json
import time
import matplotlib.pyplot as plt
import random

# Define hyperparameter ranges and configurations
lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
batch_sizes = [16, 32, 64, 128, 256]
optimizers = ["adam", "adamw", "momentum"]
weight_decays = [0, 1e-5, 1e-4, 1e-3]
num_runs = 10
max_epochs = 100
config_path = "model_configs/resnet18.json"

experiment_results = {}

for run in range(num_runs):
    lr = random.choice(lrs)
    batch_size = random.choice(batch_sizes)
    optimizer = random.choice(optimizers)
    weight_decay = random.choice(weight_decays)

    logdir = f"results/p4-q5/resnet18_{run}"
    os.makedirs(logdir, exist_ok=True)

    print(f"\nRun {run+1}/{num_runs}: Training resnet18 with parameters:")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Weight Decay: {weight_decay}")

    with open(config_path, "r") as f:
        model_config = json.load(f)

    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=4)

    command = (
        f"python main.py "
        f"--model resnet18 "
        f"--model_config ./model_configs/resnet18.json "
        f"--batch_size {batch_size} "
        f"--lr {lr} "
        f"--optimizer {optimizer} "
        f"--weight_decay {weight_decay} "
        f"--logdir {logdir} "
        f"--epochs {max_epochs}"
    )

    os.system(command)

    results_file = os.path.join(logdir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results_data = json.load(f)
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=4)
        experiment_results[run] = {
            "config": {
                "lr": lr,
                "batch_size": batch_size,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
            },
            "results": results_data,
        }
    else:
        print(f"Results file {results_file} not found for run {run}")
        experiment_results[run] = {
            "config": {
                "lr": lr,
                "batch_size": batch_size,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
            },
            "results": None,
        }

overall_results_path = "results/p4-q5/experiment_results.json"
os.makedirs(os.path.dirname(overall_results_path), exist_ok=True)
with open(overall_results_path, "w") as f:
    json.dump(experiment_results, f, indent=4)

best_run = None
best_accuracy = 0

for run, data in experiment_results.items():
    results = data["results"]
    if results is not None and "accuracy" in results:
        acc = results["accuracy"]
        if acc > best_accuracy:
            best_accuracy = acc
            best_run = run

if best_run is not None:
    print("\nBest Hyperparameters Found:")
    best_config = experiment_results[best_run]["config"]
    print(f"Run {best_run} with configuration:")
    print(f"  Learning Rate: {best_config['lr']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Optimizer: {best_config['optimizer']}")
    print(f"  Weight Decay: {best_config['weight_decay']}")
    print(f"Achieved Accuracy: {best_accuracy}")
else:
    print("No valid results found to determine the best hyperparameters.")
