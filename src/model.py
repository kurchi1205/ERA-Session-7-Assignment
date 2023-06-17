import torch.nn as nn
import torch.nn.functional as F


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), # rin=1 rout=3
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), # rin=3 rout=5
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),  # rin=5 rout=7
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) 

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11  # rin=7 rout=8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # rin=8 rout=8
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),  # rin=8 rout=12
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),  # rin=12 rout=16
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) 

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # rin=16 rout=16
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) 

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
    
    
    
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False), # rin = 1 rout = 3
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),  # rin = 3 rout = 5
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),  # rin = 5 rout = 7
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # rin = 7 rout = 8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # rin = 8 rout = 8
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),  # rin = 8 rout = 12
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=30, kernel_size=(3, 3), padding=0, bias=False),  # rin = 12 rout = 16
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=20, kernel_size=(1, 1), padding=0, bias=False),  # rin = 16 rout = 16
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1
        self.out = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)  # rin = 16 rout = 16

        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    

class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        # Input Block 28  >>> 64
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), # rin = 1 rout = 3
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.02)
        ) # output_size = 26 >>> 62

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=1, bias=False), # rin = 3 rout = 5
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.02)
        )

        # TRANSITION BLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), # rin = 6 rout = 6
            nn.BatchNorm2d(10),
            nn.ReLU(),
        ) # output_size = 11 >>> 29
        self.pool1 = nn.MaxPool2d(2, 2)# rin = 5 rout = 6
        

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False), # rin = 6 rout = 10
            nn.BatchNorm2d(20),
            nn.Dropout(0.02)
        )

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), # rin = 12 rout = 12
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2) # rin = 10 rout = 12
        

        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False), # rin = 12 rout = 20
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.02)
        )
        self.out = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=0, bias=False) # rin = 20 rout = 28
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        ) # output_size = 1


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
       
        x = self.pool1(x)
        x = self.convblock4(x)
        
        x = self.convblock5(x)
        
        x = self.pool2(x)
        x = self.convblock6(x)
        
        x = self.convblock7(x)
        x = self.out(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)