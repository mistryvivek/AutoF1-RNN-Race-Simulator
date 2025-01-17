import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from f1_dataset import CustomF1Dataloader
from earth_movers_distance import torch_wasserstein_loss

HIDDEN_SIZE = 5
LR = 0.000001
EPOCHS = 50
INPUT_SIZE = 2
OPTIM = torch.optim.Adagrad
BATCH_SIZE = 50
MSE_LOSS = nn.MSELoss()

DATASET = CustomF1Dataloader(4, "TyreLife,Compound", "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnifiedModelRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UnifiedModelRNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        self.fc_pit = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
    model = UnifiedModelRNN(INPUT_SIZE, HIDDEN_SIZE)
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

            # Check for NaN or Inf in pit_output and time_output
            if torch.isnan(pit_output).any() or torch.isinf(pit_output).any():
                print("Pit output contains NaN or Inf")
                break  # Exit or handle the issue

            if torch.isnan(time_output).any() or torch.isinf(time_output).any():
                print("Time output contains NaN or Inf")
                break  # Exit or handle the issue

            pit_loss = torch_wasserstein_loss(pit_output.squeeze(0), pit_label.squeeze(0))
            time_loss = MSE_LOSS(time_output, time_label)
            total_loss = pit_loss + time_loss

            total_loss.backward()

            optim.step()

            print(pit_loss, time_loss)    

train()
