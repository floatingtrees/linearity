import torch
import torchvision
from torchvision import transforms

class nonlinear_cnn(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1, 32, 5, padding = 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding = 2)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(50176, 64)
        self.output = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.square(x)* x.sign()
        x = self.conv2(x)
        x = torch.square(x) * x.sign()
        x = self.flatten(x)
        x = self.linear1(x)
        x = torch.square(x)* x.sign()
        x = self.output(x)
        return x

if __name__ == "__main__":
    model = nonlinear_cnn()
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
    torch.save(model, "nonlinear_model.pt")
    with torch.no_grad():
        count = 0
        for i, validation_data in enumerate(validation_loader):
            inputs, labels = validation_data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            count += 1
        print("Validation Loss: ", running_loss / count)
'''
scalar_label = torch.argmax(labels, dim = 1)
vector_label = torch.argmax(outputs, dim = 1)
outputs = scalar_label - vector_label
correct = torch.bincount(outputs)'''
