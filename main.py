import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import open_clip
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

    class_names = train_dataset.classes

    train_loader = train_loader.to(device)
    test_loader = test_loader.to(device)

    return train_loader, test_loader, class_names


def zero_shot_classification(model, test_loader, class_names, prompt_template):
    model.eval()
    text_inputs = open_clip.tokenize([prompt_template.format(class_name) for class_name in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits_per_image = image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1)    
            _, predicted = torch.max(probs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    return 100 * correct / total


def linear_probe_classification(model, train_loader, test_loader):
    X_train = []
    y_train = []

    X_test= []
    y_test = []

    model.eval()
    with torch.no_grad():
        for images,labels in tqdm(train_loader, desc='Extracting Train Features...'):
            images = images.to(device)
            img_features = model.encode_image(images)
            X_train.append(img_features.cpu().numpy())
            y_train.append(labels.cpu().numpy())
            
        for images,labels in tqdm(test_loader, desc='Extracting Test Features...'):
            images = images.to(device)
            img_features = model.encode_image(images)
            X_test.append(img_features.cpu().numpy())
            y_test.append(labels.cpu().numpy())
        
            
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    acc = 100 * accuracy_score(y_pred, y_test)
    return acc


def main():
   encoder = 'ViT-B-32'
   checkpoint = 'laion2b_s34b_b79k'
   model = load_pretrained_clip(encoder, checkpoint)

   train_loader, test_loader, class_names = load_cifar10_data('./data', 128)
   zero_shot_classification(model, test_loader, class_names)


if __name__ == '__main__':
   main()