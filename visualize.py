import matplotlib.pyplot as plt
from tabulate import tabulate


def visualize_and_compare(prompt_names, zero_shot_accuracies, linear_probe_accuracy, ensemble_accuracy):
    methods = prompt_names + ['Ensemble', 'Linear Probe']
    accuracies = zero_shot_accuracies + [ensemble_accuracy, linear_probe_accuracy]
    colors = ['skyblue'] * len(prompt_names) + ['lightgreen', 'salmon']
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color=colors)
    plt.xlabel('Method / Prompt')
    plt.ylabel('Accuracy (%)')
    plt.title('Zero-Shot vs. Linear Probe Performance on CIFAR-10 (Open CLIP ViT-B-32)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.2f}%', ha='center')
    plt.tight_layout()
    plt.savefig('cifar10_comparison.png')
    plt.show()
    
    results = []
    for name, acc in zip(prompt_names, zero_shot_accuracies):
        results.append(['Zero-Shot', name, acc])
    results.append(['Zero-Shot', 'Ensemble', ensemble_accuracy])
    results.append(['Linear Probe', 'N/A', linear_probe_accuracy])
    headers = ['Method', 'Prompt', 'Accuracy (%)']
    print("\nResults Table:")
    print(tabulate(results, headers=headers, tablefmt='grid', floatfmt='.2f'))


