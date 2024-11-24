from os.path import isfile, join
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import wavemix.classification as wmix

__root_folder = Path(__file__).parents[1].resolve().as_posix()
datasets_location = join(__root_folder, "datasets/")

num_epochs = 200
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.005
momentum = 0.5
log_interval = 500
weights_file = join(__root_folder, "models/checkpoint-wavemix.pt")

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
    ]
)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST(
        root=datasets_location,
        split="balanced",
        train=True,
        transform=transform,
    ),
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=8,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST(
        datasets_location,
        split="balanced",
        train=False,
        transform=transform,
    ),
    batch_size=batch_size_test,
    shuffle=True,
    num_workers=8,
)


def get_model() -> wmix.WaveMix:
    # WaveMix-Lite-128/7
    if isfile(weights_file):
        return torch.load(weights_file)
    else:
        return train(train_loader, test_loader)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(
    train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader
) -> wmix.WaveMix:
    device = "cuda"

    model = wmix.WaveMix(
        num_classes=47,
        depth=7,
        final_dim=128,
        ff_channel=256,
        level=1,
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)

    best_accuracy = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            print(images.shape)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 499:
                print(
                    "Spinal Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            if best_accuracy >= correct / total:
                curr_lr = learning_rate * pow(np.random.rand(1), 3).item()
                update_lr(optimizer, curr_lr)
                print(
                    "Test Accuracy of WaveMix: {} % Best: {} %".format(
                        100 * correct / total, 100 * best_accuracy
                    )
                )
            else:
                best_accuracy = correct / total
                best_model = model
                torch.save(best_model, weights_file)
                print(
                    "Test Accuracy of WaveMix: {} % (improvement)".format(
                        100 * correct / total
                    )
                )

            model.train()

    return best_model.eval()
