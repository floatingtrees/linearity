import torch
import torchvision
from torchvision import transforms

class linear_cnn(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1, 32, 5, padding = 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding = 2)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(50176, 64)
        self.output = torch.nn.Linear(64, 10)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.output(x)
        return x



if __name__ == "__main__":
    model = linear_cnn()
    transform = transforms.Compose([
            transforms.ToTensor(),
])
    training_set = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients

        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0
    torch.save(model, "linear_model.pt")
