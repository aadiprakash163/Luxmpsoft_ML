import os.path as ops
import numpy as np
import torch
import torch.nn as nn
import time
import sys
from data_generator import APPLE
from model import Net
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


batch_size = 10
learning_rate = 5e-4
num_epochs = 15

training_set = APPLE()
data_loader_train = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("Length of the dataset is: ", len(data_loader_train.dataset))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net()
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0002)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

criterion = nn.CrossEntropyLoss()

loss_all = []
print("Starting Training!!!!!!")
for epoch in range(num_epochs):
    model.train()
    ts = time.time()
    for iter, batch in enumerate(data_loader_train):
        input_image = Variable(batch[0]).to(device)
        label = Variable(batch[1]).to(device)
        # instance_labels = Variable(batch[2]).to(device)
        
        op = model(input_image)        
        loss = criterion(label, op)
        optimizer.zero_grad()
        loss_all.append(loss.item())        
        loss.backward()
        optimizer.step()
        
        if iter % 20 == 0:
            print("epoch[{}] iter[{}] loss: [{}] ".format(epoch, iter, loss.item()))
    lr_scheduler.step()
    print("Finish epoch[{}], time elapsed[{}]".format(epoch, time.time() - ts))
    # torch.save(model.state_dict(), f"storage/Trained_models/LanenetV0.1/lanenet_epoch_{epoch}_batch_{batch_size}.model")
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'binary_segmentation_loss': binary_segmentation_loss
    #     'instance_segmentation_loss': instance_segmentation_loss
    #     }, f"storage/Trained_models/LanenetV0.1/lanenet_epoch_{epoch}_batch_{batch_size}.model")





