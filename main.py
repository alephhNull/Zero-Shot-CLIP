import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_clip(encoder_type, checkpoint_path):
  model, _, preprocess = open_clip.create_model_and_transforms(encoder_type, pretrained=checkpoint_path)
  model = model.to(device)
  return model, preprocess


def load_cifar10_data(root_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match CLIP input size
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    train_loader = train_loader.to(device)
    test_loader = test_loader.to(device)

    return train_loader, test_loader


