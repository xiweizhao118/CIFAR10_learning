import torch, os, cv2
from torchvision import transforms

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

use_gpu = torch.cuda.is_available()
model = torch.load("./model/model.pth",map_location=torch.device('cuda' if use_gpu else 'cpu'))

# data pre-processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))# average and standard variance adjusting
])

# 指定文件夹
folder_path = "./testimages"

files = os.listdir(folder_path)

# 得到每一个文件的地址
images_files = [os.path.join(folder_path, f) for f in files]

for img in images_files:
    image = cv2.imread(img)
    # cv2.imshow('image', image)
    image = cv2.resize(image,(32,32))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    image = transform(image)
    image = torch.reshape(image,(1,3,32,32))
    image = image.to('cuda' if use_gpu else 'cpu')
    output = model(image)

    value,index = torch.max(output,1)
    pre_val = classes[index]

    print("prediction probability:{}, prediction index:{}, prediction result:{}".format(value,index,pre_val))

    # 等待用户按键
    # cv2.waitKey(0)
