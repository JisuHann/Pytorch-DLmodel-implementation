from utils import *
from modules import *
from dataset import *
from plain_unet import *
from tmp import *
import torch
import torchvision
import torchsummary
# version of (N, 9, 256, 256)
# Prepare whole dataset ######################################################################
dataset_path = "./log"
files = os.listdir("./log")
train_loader, valid_loader = dataset(of = False, augmentation=False, split=0.9, train_batch=32, test_batch=10)
print(len(train_loader.dataset))
print(len(valid_loader.dataset))

# train ############################################################################################
learning_rate = 0.000001
epochs = 20

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
model = UNetWithResnet50Encoder_2_deep().to(device)
#torchsummary.summary(model, (9, 256, 256))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

result_save_dir ='./g'
createFolder(result_save_dir)
predict_save_dir = result_save_dir + "/predicted"
savepath1 ='./g/history/'
createFolder(savepath1)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
print("Training")

for epoch in range(epochs):

    train_model(train_loader, model, criterion, optimizer, scheduler, device)
    train_acc, train_loss = get_loss_train(model, train_loader, criterion, device)
    print("epoch", epoch + 1, "train loss : ", train_loss, "train acc : ", train_acc)

    predict_save_folder = predict_save_dir + '/epoch' + str(epoch) + '/'
    createFolder(predict_save_folder)
    val_acc, val_loss = val_model(model, valid_loader, criterion, device, predict_save_folder)
    print("epoch", epoch + 1, "val loss : ", val_loss, "val acc : ", val_acc)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if epoch % 4 == 0:
        savepath2 = savepath1 + str(epoch) + ".pth"
        torch.save(model.state_dict(), savepath2)

print('Finish Training')

# visualization ####################################################################################################
import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot(range(epoch + 1), history['train_loss'], label='Loss', color='red')
plt.plot(range(epoch + 1), history['val_loss'], label='Loss', color='blue')

plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.subplot(2, 1, 2)
plt.plot(range(epoch + 1), history['train_acc'], label='Accuracy', color='red')
plt.plot(range(epoch + 1), history['val_acc'], label='Accuracy', color='blue')

plt.title('Accuracy history')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(result_save_dir + 'result')

print("Fin")
