import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import Face
from utils import Trainer
from network import resnet34, resnet101

# Hyper-params
data_root = r"D:\jhu-2021\sparse_sampling\project\Dictionary-learning-vs-Deep-learning-main\whole_data"
list_root = r"D:\jhu-2021\sparse_sampling\project\Dictionary-learning-vs-Deep-learning-main"
model_path = './network/'
batch_size = 60  # batch_size per GPU, if use GPU mode; resnet34: batch_size=120
num_workers = 1

init_lr = 0.01
lr_decay = 0.8
momentum = 0.9
weight_decay = 0.000
nesterov = True
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet101(pretrained=False, modelpath=model_path, num_classes=39).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=init_lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay,
                                   nesterov=nesterov)

# load data
print("Loading dataset...")
train_data = Face(list_root,train=True)
val_data = Face(list_root,train=False)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('train dataset len: {}'.format(len(train_dataloader.dataset)))

val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('val dataset len: {}'.format(len(val_dataloader.dataset)))

# models
model = resnet101(pretrained=False, modelpath=model_path, num_classes=39)  # batch_size=60, 1GPU Memory > 9000M

#Train
for epoch in range(num_epochs):
    print("Epoch: {}/{}".format(epoch + 1, num_epochs))

    model.train()

    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 因为这里梯度是累加的，所以每次记得清零
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))

        acc = torch.mean(correct_counts.type(torch.FloatTensor))

        train_acc += acc.item() * inputs.size(0)


