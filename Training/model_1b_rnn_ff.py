# https://www.youtube.com/watch?v=iKZzXisK1-Q
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from f1_dataset import CustomF1Dataloader
from earth_movers_distance import torch_wasserstein_loss

HIDDEN_SIZE = 20
LR = 0.0001
EPOCHS = 2000
INPUT_SIZE = 2
OPTIM = torch.optim.Adam
BATCH_SIZE = 50
MAE_LOSS = nn.L1Loss()
CROSS_E_LOSS = nn.CrossEntropyLoss()

DATASET = CustomF1Dataloader(4, "TyreLife,Compound", "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnifiedModelRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UnifiedModelRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        self.fc_pit = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.fc_time = nn.Sequential(
            # Need to include pit prediction here.
            nn.Linear(hidden_size + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.fc_autoregress = nn.Sequential(
            nn.Linear(input_size + 2, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )
    
    def forward(self, x):
        # [Batch Size, Sequence Length, Feature Length]
        batch_size, seq_len, _ = x.shape

        # Initialize hidden and cell states for LSTM
        h_s = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c_s = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        pit_predictions = []
        time_predictions = []
        autoregressive_output = []

        x_t = x[:, 0, :].unsqueeze(1)  # Initial input for first timestep
        autoregressive_output.append(x_t)

        for timestep in range(seq_len):
            # Step 1: Pass the input through the LSTM.
            h_t, (h_s, c_s) = self.rnn(x_t, (h_s, c_s))  # LSTM processes the input
            h_t = h_t.squeeze(1)  # Remove the sequence dimension (it will be 1)

            # Step 2: Make pit prediction
            pit_output = self.fc_pit(h_t)
            pit_predictions.append(pit_output)

            # Step 3: Make time prediction
            time_input = torch.cat((h_t, pit_output), dim=1)
            time_output = self.fc_time(time_input)
            time_predictions.append(time_output)

            # Step 4: Autoregress to get next input based upon pit decision.
            if timestep < seq_len - 1:
                autoregress_input = torch.cat((pit_output, time_output, x[:, timestep, :]), dim=-1)
                next_input = self.fc_autoregress(autoregress_input)
                autoregressive_output.append(next_input.unsqueeze(1))
                x_t = next_input.unsqueeze(1)

        # Convert lists to tensors for final output
        pit_predictions = torch.cat(pit_predictions, dim=1)
        time_predictions = torch.cat(time_predictions, dim=1)
        autoregressive_output = torch.cat(autoregressive_output, dim=1)

        return pit_predictions, time_predictions, autoregressive_output
    
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

            pit_output, time_output, autoreg_output = model(inputs)

            pit_loss = torch_wasserstein_loss(pit_output, pit_label.squeeze(-1))
            time_loss = MAE_LOSS(time_output, time_label.squeeze(-1))
            autoreg_loss = CROSS_E_LOSS(autoreg_output, inputs)
            total_loss = pit_loss + time_loss + autoreg_loss

            total_loss.backward()

            optim.step()

            print(pit_loss)
            print(time_loss)
            print(autoreg_loss)
            print("+++=")  
            print(total_loss)
            print("\n")


train()
