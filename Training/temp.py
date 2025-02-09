import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, DataLoader
from FocalLoss import FocalLoss
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, mean_absolute_percentage_error, precision_score, recall_score, f1_score
from torchviz import make_dot

from final_f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(1, "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 1
HIDDEN_SIZE = 5
EPOCHS = 2000
LR = 0.0001

OPTIM = torch.optim.Adam
PIT_DECISION_LOSS_FN = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]), reduction="none")
LAP_TIME_LOSS_FN = nn.MSELoss()
COMPOUND_PREDICTION_LOSS_FN = nn.CrossEntropyLoss()

NUM_COMPOUNDS = 5

class AutoF1LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoF1LSTM, self).__init__()
        self.compound_encoding = nn.Embedding(NUM_COMPOUNDS, 5)
        
        # Input size vector - any embeddings + what embeddings return
        self.lstm = nn.LSTM(input_size, hidden_size) #- 1 + 5

        self.pit_decision = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.lap_time = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.compound_prediction = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_COMPOUNDS)
        )

    def forward(self, lap, h_s, c_s):
        h_t, (h_s, c_s) = self.lstm(lap, (h_s, c_s))        

        pit_decision = self.pit_decision(h_t)

        return h_s, c_s, pit_decision
   

def training_loop(model, laps):
    h_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    model_pit_decisions = []

    laps_in_race = laps.shape[1]
    
    for lap in range(laps_in_race - 1):
        h_s, c_s, pit_decision = model(laps[:,lap].unsqueeze(0), h_s, c_s)
        model_pit_decisions.append(pit_decision.view(-1))

    pit_decision_loss = PIT_DECISION_LOSS_FN(torch.stack(model_pit_decisions).squeeze(1), laps[:, 1:, 0].squeeze(0))

    print(pit_decision_loss)
    print("==========")
    return pit_decision_loss.mean()

def train():
    model = AutoF1LSTM(INPUT_SIZE, HIDDEN_SIZE)
    model.to(device)

    optim = OPTIM(model.parameters(), lr=LR)

    training_dataset, _, _ = random_split(DATASET, [0.8, 0.1, 0.1])
    training_dataloader = DataLoader(training_dataset, shuffle=True)

    for epoch in range(EPOCHS):
        for race_data in training_dataloader:
            optim.zero_grad()
            
            loss = training_loop(model, race_data)

            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Warning: {name} has no gradient!")
                else:
                    print(f"{name} gradient: {param.grad.norm()}")


            optim.step()

train()