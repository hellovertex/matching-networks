from src import omniglot
from src import utils
from src.embeddings import OmniglotEmbeddingF
import torch.optim as optim
import torch
import os
PATH = './models/omniglot/baseline_net.pth'
N = 5
k = 1
batch_size = N * k

""" Supervised training of baseline classifier on Omniglot"""
training_data = omniglot.training_data
augmented_dataset = omniglot.add_rotated_classes(training_data)

# dataloader to generate batches
train_loader = omniglot.get_data_loader(augmented_dataset, N=N, k=k)
# images, targets = next(iter(train_loader))

# load model
model = OmniglotEmbeddingF(num_classes=1200)
if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH))

# move data and model to GPU
train_loader = utils.DeviceDataLoader(train_loader, utils.get_default_device())
model = utils.to_device(model, utils.get_default_device())

# optimizer + lr + loss
optimizer = optim.Adam(model.parameters(), lr=0.005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 100000
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        images, targets = data
        # print(images.size())  # torch.Size([5, 1, 28, 28])

        # reset optimizer gradients
        optimizer.zero_grad()

        # forward + loss
        outputs = model.forward(images)
        loss = criterion(outputs, targets)

        # backward + update
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    torch.save(model.state_dict(), PATH)
