import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/paul-krug/pytorch-tcn
from pytorch_tcn import TCN
from f1_dataset import CustomF1Dataloader
from earth_movers_distance import torch_wasserstein_loss
from torch.utils.data import random_split, DataLoader

INPUT_DIM = 2
TCN_OUTPUT_SIZE = 512
TCN_CHANNELS = [32, 64, 128]
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.2

DATASET = CustomF1Dataloader(4, "TyreLife,Compound", "../Data Gathering")
EPOCHS = 2000
OPTIM = torch.optim.Adam
MAE_LOSS = nn.L1Loss()
BATCH_SIZE = 50
LR = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnifiedModelCNNRNN(nn.Module):
    def __init__(self, input_dim, tcn_channels, tcn_kernel_size, tcn_dropout):
        super(UnifiedModelCNNRNN, self).__init__()
        
        self.tcn = TCN(
            num_inputs=input_dim,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout
        )
        
        # Fully connected layers for outputs
        self.fc_pit = nn.Sequential(
            nn.Linear(tcn_channels[-1] * 87, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.fc_time = nn.Sequential(
            nn.Linear(tcn_channels[-1] * 87, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.tcn(x)
        print(features.shape)

        pit_output = self.fc_pit(features)
        time_output = self.fc_time(features)
        
        return pit_output, time_output
    
def train():
    model = UnifiedModelCNNRNN(INPUT_DIM, TCN_CHANNELS, TCN_KERNEL_SIZE, TCN_DROPOUT)
    model.to(device)

    optim = OPTIM(model.parameters(), lr=LR)

    training_dataset, testing_dataset, validation_dataset = random_split(DATASET, [0.8, 0.1, 0.1])

    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    data_iteration = iter(training_dataloader) 
    n = 0

    for epoch in range(EPOCHS):
        model.train()

        for inputs, pit_label, time_label in training_dataloader:
            #print(f"Inputs shape: {inputs.shape}")
            #print(f"Pit label shape: {pit_label.shape}")
            #print(f"Time label shape: {time_label.shape}")

            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("Inputs contain NaN or Inf")
                break

            optim.zero_grad()

            inputs = inputs.permute(0, 2, 1) 
            pit_output, time_output = model(inputs)

            pit_loss = torch_wasserstein_loss(pit_output.squeeze(0), pit_label.squeeze(0))
            time_loss = MAE_LOSS(time_output, time_label)
            total_loss = pit_loss + time_loss

            total_loss.backward()

            optim.step()

            print(total_loss)    

train()