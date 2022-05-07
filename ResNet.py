import torch
from torch import nn
from torch.utils.data import DataLoader
from data import Face
from network import resnet34, resnet101
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
def main():
    # Hyper-params
    data_root = r"D:\jhu-2021\sparse_sampling\project\Dictionary-learning-vs-Deep-learning-main\whole_data"
    list_root = r"D:\jhu-2021\sparse_sampling\project\Dictionary-learning-vs-Deep-learning-main"
    model_path = './network/'
    batch_size = 40  # batch_size per GPU, if use GPU mode; resnet34: batch_size=120
    num_workers = 1
    num_classes = 39

    init_lr = 0.01
    lr_decay = 0.8
    momentum = 0.9
    weight_decay = 0.000
    nesterov = True
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet101(pretrained=False, modelpath=model_path, num_classes=num_classes).to(device)
    # model = vgg16(pretrained=False).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr,
                                       momentum=momentum,
                                       weight_decay=weight_decay,
                                       nesterov=nesterov)

    # load data
    print("Loading dataset...")
    train_data = Face(data_root,list_root,train=True)
    val_data = Face(data_root,list_root,train=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('train dataset len: {}'.format(len(train_dataloader.dataset)))

    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('val dataset len: {}'.format(len(val_dataloader.dataset)))


    #Train
    train_loss_list = []
    train_acc_list = []
    for epoch in range(num_epochs):
        print("Epoch: {}/{}".format(epoch + 1, num_epochs))

        model.train()
        train_loss = 0.0
        total_train = 0
        correct_train = 0
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
            total_train += labels.size(0)  # 测试了多少个数据
            correct_train += (predictions == labels).sum().item()
        train_acc_epoch = correct_train/total_train
        train_loss_list.append(train_loss/total_train)
        train_acc_list.append(train_acc_epoch)
    plt.figure(1)
    plt.plot(train_acc_list)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.figure(2)
    plt.plot(train_loss_list)
    plt.title('Training Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    path = '.\models\pretrained_resnet101.pth'
    torch.save(model,path)
    #Test
    model = torch.load(path)
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_val = 0
        val_loss = 0
        correct_val = 0
        val_predict = []
        val_label = []

        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            val_predict.extend(predictions.detach().cpu().numpy())
            val_label.extend(labels.detach().cpu().numpy())
            total_val += labels.size(0)  # 测试了多少个数据
            correct_val += (predictions == labels).sum().item()
        val_accuracy = correct_val / total_val

        sklearn_precision = precision_score(val_label, val_predict, average='micro')
        sklearn_recall = recall_score(val_label, val_predict, average='micro')
        sklearn_f1 = f1_score(val_label, val_predict, average='micro')

    print("Test accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(val_accuracy,
                                                                                              sklearn_precision,
                                                                                              sklearn_recall,
                                                                                        sklearn_f1))
    print('Test loss:', val_loss/total_val)
if __name__ == '__main__':
    main()