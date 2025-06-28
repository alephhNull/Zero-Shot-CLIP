import torch
import open_clip
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    acc = 100 * correct / total
    return acc


def ensemble_classification(model, test_loader, class_names, prompt_templates):
    model.eval()
    text_features_list = []
    for prompt_template in prompt_templates:
        text_inputs = open_clip.tokenize([prompt_template.format(class_name) for class_name in class_names]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_list.append(text_features)

    ensemble_text_features = torch.stack(text_features_list, dim=0).mean(dim=0)
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits_per_image = image_features @ ensemble_text_features.T
            probs = logits_per_image.softmax(dim=-1)
            _, predicted = torch.max(probs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    acc = 100 * correct / total
    return acc

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