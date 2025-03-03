import os
from HW1_2025.assignment1_release.utils import generate_plots

learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

for lr in learning_rates:
    print(f"\nTraining ResNet18 with learning rate {lr}...\n")
    os.system(f"python main.py --model resnet18 --model_config model_configs/resnet18.json --logdir results/lr_{lr} --epochs 50 --optimizer adam --lr {lr}")

generate_plots(
    ['results/lr_0.1', 'results/lr_0.01', 'results/lr_0.001', 'results/lr_0.0001', 'results/lr_1e-05'],
    ['LR=0.1', 'LR=0.01', 'LR=0.001', 'LR=0.0001', 'LR=0.00001'],
    'results'
)

print("\nDone")
