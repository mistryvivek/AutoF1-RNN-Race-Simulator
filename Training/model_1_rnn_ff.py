import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from f1_dataset import CustomF1Dataloader
from earth_movers_distance import torch_wasserstein_loss

HIDDEN_SIZE = 5
LR = 0.0001
EPOCHS = 2000
INPUT_SIZE = 10
OPTIM = torch.optim.Adam
BATCH_SIZE = 50
MAE_LOSS = nn.MSELoss()
CE_LOSS = nn.CrossEntropyLoss()
# There is an unknown tire in our dataset.
PIT_CHOICES_NUM = 7

DATASET = CustomF1Dataloader(4, "TyreLife,Compound,SpeedI1,SpeedI2,SpeedFL,SpeedST,DRS,DriverNumber,Team,TrackStatus", "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnifiedModelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pit_choices_num):
        super(UnifiedModelRNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        self.fc_pit = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, PIT_CHOICES_NUM)
        )
        
        self.fc_time = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        h_seq, _ = self.rnn(x)
        
        pit_output = self.fc_pit(h_seq)
        time_output = self.fc_time(h_seq)
        
        return pit_output, time_output
    
def train():
    model = UnifiedModelRNN(INPUT_SIZE, HIDDEN_SIZE, PIT_CHOICES_NUM)
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

            pit_output, time_output = model(inputs)

            pit_loss = CE_LOSS(pit_output.view(-1, 7), pit_label.view(-1))
            time_loss = MAE_LOSS(time_output, time_label)
            total_loss = pit_loss + time_loss

            total_loss.backward()

            optim.step()

            print(time_loss, pit_loss)

train()
