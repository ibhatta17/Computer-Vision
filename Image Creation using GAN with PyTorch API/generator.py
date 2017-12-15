import torch.nn as nn

class G(nn.Module): 
    def __init__(self): # init function to define the architecture of the generator
        super(G, self).__init__() # Inheriting from the nn.Module tools
        self.main = nn.Sequential( 
             '''
             This a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, 
             etc)
             
             Need to start with the inversed conv. The CNN takes image as an input and outputs a vector. The role of the 
             generator is to generate fake images from a vector of random noise. Hence the inverse convolution is required.
             
             '''
            nn.ConvTranspose2d(100, # random vector of size 100
                               512, # feature maps
                               4, # kernel size
                               1, # stride of 1
                               0, # padding
                               bias = False),             
            nn.BatchNorm2d(512), # normalizing all the features along the dimension of the batch
            nn.ReLU(True), # ReLU rectification to break the linearity
            
            # 2nd Inverse Convolution Layer
            nn.ConvTranspose2d(512, # here the input is the output of the previous layer
                               256, # new feature maps
                               4, 
                               2, 
                               1, # padding of 1
                               bias = False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            
            # 3rd Inverse Convolution Layer
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            
            # 4th Inverse Convolution Layer
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64), 
            nn.ReLU(True), 
            
            # Final Inverse Convolution Layer
            nn.ConvTranspose2d(64, # number of input = output from previous layer
                               3, # 3 color-channel output(3-D for RGB)
                               4, 2, 1, bias = False),
            nn.Tanh() # Hyperbolic-tangent rectification at the output convolution layer.
            # Hence the output of the generator lies between -1 and 1 centered around 0.
        )

    def forward(self, input):
        '''
        This will forward propagate the input signal(random noise of size 100) thru layers of neural network  of the generator. 
        The function returns the output containing the generated images(3 color channel values).
        
        '''
        output = self.main(input) 
        return output