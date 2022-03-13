import numpy as np
from tqdm import tqdm 
from collections import namedtuple
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import prepare_data, decode_label, show_examples, Swish, weight_init, compare_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###########################
#   Creating model class  #
###########################
class MnistAutoencoder (nn.Module):
    def __init__(
        self, 
        input_shape, 
        encoder_f, 
        decoder_f, 
        lat_space_size
    ):
        super(MnistAutoencoder, self).__init__()
        # Input Shape
        self.ch, self.h, self.w = input_shape
        self.dropout = nn.Dropout(0.25)
        # Encoder
        self.Encoder = nn.Sequential(
            # Conv1
            nn.Conv2d(
                in_channels=self.ch, 
                out_channels=encoder_f[0], 
                kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(encoder_f[0]),
            Swish(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.3),
            # Conv2
            nn.Conv2d(
                in_channels=encoder_f[0], 
                out_channels=encoder_f[1], 
                 kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(encoder_f[1]),
            Swish(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.3),
            # Conv3
            nn.Conv2d(
                in_channels=encoder_f[1], 
                out_channels=encoder_f[2], 
                kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(encoder_f[2]),
            Swish(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.3),
            # Conv4
            nn.Conv2d(
                in_channels=encoder_f[2], 
                out_channels=encoder_f[3], 
                kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(encoder_f[3]),
            Swish(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.3)
        )
        
        # Get the shape of Encoder output
        self.CNN, self.CNN_flatten = self._get_conv_output((self.ch, self.h, self.w), self.Encoder)
        
        # Latent Space
        self.latent_space1 = nn.Linear(self.CNN_flatten, lat_space_size)
        self.latent_space2 = nn.Linear(lat_space_size, lat_space_size)
        self.latent_space3 = nn.Linear(lat_space_size, self.CNN_flatten)
        
        #Decoder (Upsample the input from decoder's output's shape to Mnist shape)
        self.Decoder = nn.Sequential(
            # Transposed conv1
            nn.ConvTranspose2d(in_channels=encoder_f[3],
                               out_channels=decoder_f[0], 
                               kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(decoder_f[0]),
            Swish(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.3),
            # Transposed conv2
            nn.ConvTranspose2d(in_channels=decoder_f[0], 
                               out_channels=self.ch, 
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(self.ch),
            Swish(),
            nn.Dropout(0.3)
        )
        
    def _get_conv_output(self, shape, layers):
        bs = 1
        dummy_x = torch.empty(bs, *shape)
        x = layers(dummy_x)
        CNN = x.size()
        CNN_flatten = x.flatten(1).size(1)
        return CNN, CNN_flatten

    def encoder(self, x):
        encoded = self.Encoder(x)
        return encoded
    
    def decoder(self, x):
        decoded = self.Decoder(x)        
        return decoded

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        encoded = encoded.view(-1, self.CNN_flatten)
        # Latent Space
        latent_space1 = self.latent_space1(encoded)
        latent_space2 = self.latent_space2(latent_space1)
        latent_space3 = self.latent_space3(latent_space2)
        # Decoding
        decoded = latent_space3.view(-1, self.CNN[1], self.CNN[2], self.CNN[3])
        decoded = self.decoder(decoded) 
        return decoded
        
        
#########################################
#            Trainer Class              #
#########################################
class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, load_path=None):
        self.__class__.__name__ = "PyTorch Trainer"
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        ## Setup Metric class
        self.metrics = namedtuple('Metric', ['loss', 'train_error', 'val_error'])
        
        # if model exist
        if load_path:
            self.model = torch.load(load_path)
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def run(self, train_loader, val_loader, n_epochs):
        min_valid_loss = np.inf
        ## Setup Metric class
        Metric = namedtuple('Metric', ['loss', 'agv_train_error', 'avg_val_error'])
        self.metrics = []
        self.model.train() 
        min_valid_loss = np.inf
#         np.mean([self.criterion(self.model(g),l).item() for g, l in val_loader])
        for epoch in range(n_epochs):
            train_loss = 0.0
#             lr = self.optimizer.param_groups[0]['lr']
            lr = self.scheduler.get_last_lr()[0]
            data_iter = iter(train_loader)
            prog_bar = tqdm(range(len(train_loader)))
            for step in prog_bar: # iter over batches

                ######################
                # Get the data ready #
                ######################
                # get the input images and their corresponding labels
                images, _ = data_iter.next() # no need for labels
                
                # wrap them in a torch Variable and move tnsors to the configured device
                images = Variable(images).to(device)                                  
                
                ################
                # Forward Pass #
                ################
                # Feed input images
                out_images = self.model(images)
                # Find the Loss
                loss = self.criterion(images, out_images)

                #################
                # Backward Pass #
                #################
                # Calculate gradients
                loss.backward()
                # Update Weights
                self.optimizer.step()
                # clear the gradient
                self.optimizer.zero_grad()
                
                #################
                # Training Logs #
                #################
                # Calculate total Loss
                train_loss += loss.item()
                # Calculate total samples
                
                prog_bar.set_description('Epoch {}/{}, Loss: {:.4f}, lr={:.7f}'.format(epoch+1, n_epochs, loss.item(),lr))
#                 torch.cuda.empty_cache()
                del images
                del out_images
                del loss
                
            valid_loss = 0.0
            self.model.eval() # Optional when not using Model Specific layer
            with torch.no_grad():
                for images, _ in (val_prog_bar := tqdm(val_loader)):
                    # Forward Pass
                    out_imagse = self.model(images)
                    # Find the Loss
                    loss = self.criterion(images, out_imagse)
                    # Calculate Loss
                    valid_loss += loss.item()
                    
                    val_prog_bar.set_description('Validation, Loss: {:.4f}'\
                                                 .format(epoch+1, n_epochs, loss.item()))

            #Check point
            if min_valid_loss > valid_loss:
                print('Validation Loss Decreased ({:.6f} ===> {:.6f}) \nSaving The Model'.format(min_valid_loss/len(val_loader), 
                                                                                                 valid_loss/len(val_loader)))

                min_valid_loss = valid_loss/len(val_loader)

            self.metrics.append(Metric(loss=train_loss, 
                                       agv_train_error=train_loss/len(train_loader.dataset),
                                       avg_val_error=valid_loss/len(train_loader.dataset)))
            
            # Decrease the lr
            scheduler.step()
            
            
# gathering the data
train_loader, val_loader, test_loader, invTrans = prepare_data()

# Defining Parameters
model = MnistAutoencoder(input_shape=(1, 28, 28), 
                         encoder_f=[32,64,64,64], 
                         decoder_f=[64,64,64,32], 
                         lat_space_size=225)
# initialize the model's parametrs
model.apply(weight_init)
# define the learninig criterion
criterion = nn.MSELoss()
# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=P.lr, weight_decay=5e-4)
# define the scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Refresh tqdm bar
tqdm.refresh

# Define model trainer and start training 
model_trainer = Trainer(model, optimizer, criterion, scheduler, '')
model_trainer.run(train_loader, val_loader, 5)

# save the model
torch.save(model_trainer.model, 'Model.pt')

# compare_results
compare_results(model_trainer.model, test_loader, 5)
