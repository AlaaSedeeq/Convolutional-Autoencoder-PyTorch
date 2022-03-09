import numpy as np
from collections import namedtuple
from tqdm import tqdm 
import torch 
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision 
from torchvision.datasets import MNIST
from torchvision.transforms import transforms as T 
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#################################
#   Gathering the prepare data  #
#################################
def prepare_data():
    train_data = MNIST(
        root=r'data',
        download=True,
        train=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(mean = (0.1307,),
                        std = (0.3081,))
        ]))

    test_data = MNIST(
        root=r'data',
        download=True,
        train=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(mean = (0.1307,),
                        std = (0.3081,))
        ]))
    # For Rnormalization
    invTrans = T.Compose([
        T.Normalize(mean = (0),
                    std = (1/0.3081,))
    ])

    # Split the train data loader for train/validation
    train_size = train_data.data.shape[0]
    val_size, train_size = int(0.20 * train_size), int(0.80 * train_size) # 80 / 20 train-val split
    test_size = test_data.data.shape[0]
    batch_size = 128

    # Add dataset to dataloader that handles batching
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(np.arange(val_size, val_size+train_size))
    )

    val_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        sampler=SubsetRandomSampler(np.arange(0, val_size))
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader, invTrans


#####################
#   Label decoding  #
#####################
def decode_label(l):
    label_to_class = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
                      5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
    return label_to_class[l]


# show examples
def show_examples(data_loader, r, c):
    fig, axes = plt.subplots(r, c, figsize=(12,8))
    for i in range(12):
        idx = np.random.randint(train_data.data.shape[0], size=1)[0]
        r, c = i // 4, i % 4
        img = train_data.data[idx]
        label = decode_label(train_data.targets[idx]).split('-')[1].upper()
        axes[r][c].set_title(label)
        axes[r][c].axis('off')
        axes[r][c].imshow(img.numpy(), interpolation='nearest')

    plt.draw()
    

#################################
#   Swish Activation Function   #
#################################
class Swish(nn.Module):
    def forward(self, input):
        return (input * torch.sigmoid(input))
    
    def __repr__(self):
        return self.__class__.__name__ + 'Swish'

    
#########################
#   Weight initializer  #
#########################   
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
        
        
################################################
#   Compare model's results with ground-truth  #
################################################
def compare_results(model, data_loader, samples):

    fig, axes = plt.subplots(samples, 2, figsize=(10,10))
    plt.tight_layout()
    data = iter(data_loader)
    model.eval()
    with torch.no_grad():
        for i in range(samples):
            # Make a prediction
            idx = np.random.randint(0, data_loader.batch_size, 1)[0]
            images, labels = next(data)
            image, label = images[idx].unsqueeze(1), decode_label(labels[idx])
            image_pred = model(image).squeeze(1)
            # Compaer the results with the Ground-Truth
            # Real
            axes[i][0].set_title(f'Ture {label.upper()}')
            axes[i][0].imshow(image.squeeze(1).numpy().reshape(28, 28, 1))
            axes[i][0].set_xticks([], minor=False)
            axes[i][0].set_yticks([], minor=False)
            # Generated
            axes[i][1].set_title(f'Generated {label.upper()}')
            axes[i][1].imshow(image_pred.squeeze(1).numpy().reshape(28, 28, 1))
            axes[i][1].set_xticks([], minor=False)
            axes[i][1].set_yticks([], minor=False)
            
    plt.draw()