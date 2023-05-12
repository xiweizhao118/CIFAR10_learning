
# get traning dataset
from torchvision import datasets, transforms
from net import MyModel
import torch
from torch.utils.data import DataLoader

# data pre-precessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))# average and standard variance adjusting
])

train_data_set = datasets.CIFAR10('./dataset', train=True, transform = transform, download=True)

# get test dataset
test_data_set = datasets.CIFAR10('./dataset', train=False, transform = transform, download=True)

# load the dataset
train_data_loader = DataLoader(train_data_set, batch_size=64, shuffle = True)
test_data_loader = DataLoader(test_data_set, batch_size=64, shuffle = True)
train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

# define networks
myModel = MyModel()

# verify if using the gpu
vse_gpu = torch.cuda.is_available()
if(vse_gpu):
    print("gpu is available")
    myModel = myModel.cuda()

# number of rolling for training
epochs = 100

# loss function
lossFn = torch.nn.CrossEntropyLoss()

# optimization filter
optimizer = torch.optim.SGD(myModel.parameters(), lr=0.01)

for epoch in range(epochs):
    print("training No.{}/{}".format(epoch+1, epochs))

    # loss variable
    train_total_loss = 0.0
    test_total_loss = 0.0

    # accurate rate
    train_total_acc = 0.0
    test_total_acc = 0.0

    # training starts
    for data in train_data_loader:
        inputs, labels = data
        
        if vse_gpu:
            inputs = inputs.cuda()# cuda: 将模型部署到gpu上去
            labels = labels.cuda()

        # the whole optimization steps
        optimizer.zero_grad() # clear the old gradiance

        outputs = myModel(inputs)

        # calculate the loss
        loss = lossFn(outputs, labels)

        loss.backward() #propogate backward?

        # update params
        optimizer.step()

        # calculate the accurate
        _, index = torch.max(outputs, 1) # get the maximum prediction and index
        acc = torch.sum(index == labels).item()

        train_total_loss += loss.item()
        train_total_acc += acc

    # testing code
    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            if vse_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = myModel(inputs)

            # calculate the loss
            loss = lossFn(outputs, labels)

            # calculate the accurate
            _, index = torch.max(outputs, 1) # get the maximum prediction and index
            acc = torch.sum(index == labels).item()

            test_total_loss += loss.item()
            test_total_acc += acc

    # print the results after every training
    print("train loss:{}, train acc:{}, test loss:{}, test acc:{}".format(train_total_loss,train_total_acc/train_data_size,test_total_loss,test_total_acc/test_data_size))


torch.save(myModel,"model/model.pth")
