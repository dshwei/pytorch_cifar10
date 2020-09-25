#coding:utf-8
import torch
from torch.nn import modules
from torch import nn
from torch import optim
from  torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import os

class VGG16(modules.Module):

    def __init__(self,num_class=10):
        super(VGG16,self).__init__()
        self.feature = modules.Sequential(
            # #1,
            modules.Conv2d(3,64,kernel_size=3,padding=1),
            modules.BatchNorm2d(64),
            modules.ReLU(True),
            #2
            modules.Conv2d(64,64,kernel_size=3,padding=1),
            modules.BatchNorm2d(64),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            #3
            modules.Conv2d(64,128,kernel_size=3,padding=1),
            modules.BatchNorm2d(128),
            modules.ReLU(True),
            # modules.MaxPool2d(kernel_size=2,stride=2),
            #4
            modules.Conv2d(128,128,kernel_size=3,padding=1),
            modules.BatchNorm2d(128),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            #5
            modules.Conv2d(128,256,kernel_size=3,padding=1),
            modules.BatchNorm2d(256),
            modules.ReLU(True),
            #6
            modules.Conv2d(256,256,kernel_size=3,padding=1),
            modules.BatchNorm2d(256),
            modules.ReLU(True),
            #7
            modules.Conv2d(256,256,kernel_size=3,padding=1),
            modules.BatchNorm2d(256),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            #8
            modules.Conv2d(256,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            #9
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            #10
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            #11
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            #12
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            #13
            modules.Conv2d(512,512,kernel_size=3,padding=1),
            modules.BatchNorm2d(512),
            modules.ReLU(True),
            modules.MaxPool2d(kernel_size=2,stride=2),
            modules.AvgPool2d(kernel_size=1,stride=1),

            )
        # 全连接层
        self.classifier = modules.Sequential(
            # #14
            modules.Linear(512,4096),
            modules.ReLU(True),
            modules.Dropout(),
            #15
            modules.Linear(4096,4096),
            modules.ReLU(True),
            modules.Dropout(),
            #16
            modules.Linear(4096,num_class),

        )
    def forward(self,x):
        out = self.feature(x)
        # print(out.shape),batch_size/heigth,width,
        out = out.view(out.size(0),-1)

        out = self.classifier(out)
        return out

def load_data():
    '''下载训练数据集'''
    # train_dataset = datasets.CIFAR10("I:\datasets",train=True,transform=transforms.ToTensor(),download=True)
    train_dataset = datasets.CIFAR10("I:\datasets", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # test_dataset = datasets.CIFAR10("I:\datasets",train=False,transform=transforms.ToTensor(),download=True)
    test_dataset = datasets.CIFAR10("I:\datasets", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataset,train_loader,test_dataset,test_loader

'''超参数'''
batch_size = 256
learning_rate = 1e-2
num_epoches = 10
# 创建模型
model = VGG16()
'''定义loss 和optimizer'''
criterion = modules.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train_validate(model,train_loader,epoch):
    '''超参数'''
    batch_size = 256
    learning_rate = 1e-2
    num_epoches = 10
    '''加载数据集'''
    train_dataset,train_loader,test_dataset,test_loader =load_data()
    # 创建模型
    model = VGG16()
    model.train()
    '''定义loss 和optimizer'''
    criterion = modules.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 查看是否有GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use gpu")
        model = model.cuda()

    '''训练测试模型'''
    # for epoch in range(num_epoches):
    print('*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)  # .format为输出格式，formet括号里的即为左边花括号的输出
    running_loss = 0.0
    running_acc = 0.0
    # for i ,data in tqdm(enumerate(train_loader,1)):
    '''训练模型'''
    for i, data in tqdm(enumerate(train_loader, 1)):
        img, label = data
        print(img.shape)
        # cuda
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)  # 预测最大值所在的位置标签
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))

    '''保存训练好的模型  pt pth pkl rar'''
    torch.save(model.state_dict(), "./cnn.pth")


def test(model,test_loader,test_data_len):
    # 模型评估
    model.eval()
    eval_loss = 0
    eval_acc = 0
    # 查看是否有GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use gpu")
        model = model.cuda()
    with torch.no_grad():
        '''测试模型'''
        for i, data in tqdm(enumerate(test_loader, 1)):
            img, label = data
            if use_gpu:
                print("use gpu")
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
                out = model(img)
                loss = criterion(out, label)
                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label).sum()
                eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_data_len, eval_acc / test_data_len))

if __name__== '__main__':
    # train_validate()
    test_flag=True
    model = VGG16()
    '''使用加载训练好的模型'''
    if os.path.exists("./cnn.pth"):
        checkpoint = torch.load("./cnn.pth")
        model_state_dict = model.state_dict()
        print(checkpoint.keys())
        print(model_state_dict.keys())
        print(checkpoint)
        print(model_state_dict)
        # 将保存的模型中key和当前模型中参数key相同的过滤出出来
        checkpoint_state_dict = {k:v for k,v in checkpoint.items() if k in model_state_dict}
        # 过滤保存的参数和当前模型中的参数键值对是否相同
        model_state_dict.update(checkpoint_state_dict)

        model.load_state_dict(checkpoint)
    '''加载数据集'''
    train_dataset, train_loader, test_dataset, test_loader = load_data()

    print(len(test_loader))
    print(len(test_dataset))
    '''测试模型'''
    if test_flag:
        test(model, test_loader,len(test_dataset))

    else:
        for epoch in range(0,num_epoches):
            train_validate(model,train_loader,epoch)
            test(model, test_loader,len(test_dataset))