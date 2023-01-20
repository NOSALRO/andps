import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pyLasaDataset as lasa
from torchvision import transforms
from PIL import Image
from utils import andps_images as net
from utils import CustomDataset as CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

convert_tensor = transforms.ToTensor()

# Lasa dataset demos
data = [lasa.DataSet.JShape, lasa.DataSet.Angle, lasa.DataSet.Khamesh]
names = ["JShape", "Angle", "Khamesh"]
# Images corresponding to each demo
# [convert_tensor(Image.open("data/train/Sine_demo_1.png")), convert_tensor(Image.open("data/train/Spoon_demo_1.png")), convert_tensor(Image.open("data/train/JShape_demo_1.png"))]
images = []


#  Create Dataset with image indexing
for d in range(len(data)):
    for i in range(7):
        images.append(convert_tensor(Image.open(
            "data/images/train/" + names[d] + "_demo_" + str(i+1) + ".png")))
        cur_demo = data[d].demos[i]
        cur_pos = cur_demo.pos
        cur_vel = cur_demo.vel
        dt_global = cur_demo.dt
        cur_np_X = cur_pos.transpose()
        temp = torch.Tensor([d*7 + i]).repeat(cur_np_X.shape[0], 1)
        cur_np_X = torch.cat((torch.Tensor(cur_np_X), temp), 1)
        cur_np_Y = torch.Tensor(cur_vel.transpose())
        if d == 0 and i == 0:
            np_X = cur_np_X
            np_Y = cur_np_Y
        else:
            np_X = torch.cat((np_X, cur_np_X), 0)
            np_Y = torch.cat((np_Y, cur_np_Y), 0)


# Dataset now is in the form X = [[posX, posY, image_id],...]  Y = [[velX, velY]] (tensors)
# ds Matrix dimension
dim = np_Y.shape[1]
# print(np_X)

# all trajectories have the same target (0, 0)
target = np_X[-1, :2]

# Number of dynamical systems
num_DS = 9
net = net(dim, num_DS, target, len(names), images, device)
net.to(device)

# Hyperparameters
batch_size = 16
epochs = 20
le_r = 1e-3

# Load dataset
dataset = CustomDataset(np_X, np_Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

batches_per_epoch = np_X.shape[0]//batch_size
optimizer = torch.optim.AdamW(net.parameters(), lr=le_r, weight_decay=0.1)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

net.apply(weights_init)

for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, datas in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, output]
        inputs, outputs = datas
        # print(inputs)
        batch_loss = 0.0
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = net(inputs.to(device).view(-1, 3))
        loss = torch.nn.functional.mse_loss(output, outputs.to(device).view(-1, dim), reduction='mean')
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_mean_loss = running_loss/batches_per_epoch
    print("Epoch: ", epoch+1, "Loss: ", train_mean_loss)
    if ((epoch + 1) % 10 == 0):
        torch.save(net.state_dict(),'models/lasa_image_' + str(epoch + 1) + '.pt')
print('Finished Training')
torch.save(net.state_dict(), 'models/lasa_image_' + str(epoch + 1) + '.pt')
