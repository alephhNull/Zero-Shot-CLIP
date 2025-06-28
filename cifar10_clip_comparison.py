import argparse
from loaders import load_pretrained_clip, load_cifar10_data
from classifiers import zero_shot_classification, ensemble_classification, linear_probe_classification  
from visualize import visualize_and_compare



def main(args):
    prompt_names = [prompt.replace('{}', '').strip().capitalize() for prompt in args['prompts']]

    model, _ = load_pretrained_clip(args['encoder'], args['checkpoint'])
    train_loader, test_loader, class_names = load_cifar10_data(args['data_dir'], args['batch_size'])

    print("\nComputing zero-shot accuracies...")
    zero_shot_accuracies = []
    for prompt_template, name in zip(args['prompts'], prompt_names):
        acc = zero_shot_classification(model, test_loader, class_names, prompt_template)
        zero_shot_accuracies.append(acc)
        print(f"Zero-shot accuracy with prompt '{prompt_template}': {acc:.2f}%")

    ensemble_accuracy = ensemble_classification(model, test_loader, class_names, args['prompts'])
    print(f"Ensemble accuracy: {ensemble_accuracy:.2f}%")

    print("\nComputing linear probe accuracy...")
    linear_probe_accuracy = linear_probe_classification(model, train_loader, test_loader)
    print(f"Linear probe accuracy: {linear_probe_accuracy:.2f}%")

    visualize_and_compare(prompt_names, zero_shot_accuracies, linear_probe_accuracy, ensemble_accuracy)



if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compare zero-shot and linear probe performance on CIFAR-10 with Open CLIP')
    parser.add_argument('--prompts', nargs='+', default=['a photo of a {}', 'a picture of a {}', 'an image of a {}'],
                        help='List of prompt templates (e.g., "a photo of a {}")')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loading')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for CIFAR-10 data')
    parser.add_argument('--encoder', type=str, default='ViT-B-32', help='CLIP encoder type')
    parser.add_argument('--checkpoint', type=str, default='laion2b_s34b_b79k', help='Pretrained checkpoint')
    args = parser.parse_args()
    args = vars(args)
    main(args)