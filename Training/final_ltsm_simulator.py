import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, DataLoader
from earth_movers_distance import torch_wasserstein_loss
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from final_f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(1, "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 2
HIDDEN_SIZE = 5
EPOCHS = 2000
LR = 0.001

OPTIM = torch.optim.Adam
PIT_DECISION_LOSS_FN = torch_wasserstein_loss
LAP_TIME_LOSS_FN = torch.nn.MSELoss()

class AutoF1LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoF1LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.ltsm = nn.LSTM(input_size, hidden_size)

        self.pit_decision = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )

        self.lap_time = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, lap, h_s, c_s):
        h_t, (h_s, c_s) = self.ltsm(lap, (h_s, c_s)) 
        pit_decision = self.pit_decision(h_t)
        lap_time_prediction = self.lap_time(h_t)
        return h_s, c_s, pit_decision, lap_time_prediction

def testing_loop(model, laps):
    h_s = torch.zeros(1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(1, HIDDEN_SIZE).to(device)
    model_pit_decisions = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, c_s, pit_decision, _ = model(laps[:,lap], h_s, c_s)
        model_pit_decisions.append(pit_decision)
    
    return model_pit_decisions, laps[:,1:,0].squeeze(0)
    

def stats(testing_dataset, model):
    model.eval()

    pit_true_labels = []
    pit_predicted_labels = []

    # Run through testing data.
    training_dataloader = DataLoader(testing_dataset)
    for race_data in training_dataloader:
        model_pit_decisions, real_pit_decisions = testing_loop(model, race_data)
        pit_true_labels.append(real_pit_decisions.numpy())
        pit_predicted_labels.append(torch.stack(model_pit_decisions).detach().numpy().flatten())

    pit_true_labels = np.concatenate(pit_true_labels)
    pit_predicted_labels = np.concatenate(pit_predicted_labels).flatten()
    # Skilearn needs everything in same format.
    pit_predicted_labels = np.array([1.0 if prediction >= 0.75 else 0.0 for prediction in pit_predicted_labels])
    
    # PIT DECISIONS
    # Confusion Matrix
    conf_matrix = confusion_matrix(pit_true_labels, pit_predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Accuracy
    accuracy = accuracy_score(pit_true_labels, pit_predicted_labels)
    print(f"Accuracy: {accuracy:.4f}%")

    # Precision
    precision = precision_score(pit_true_labels, pit_predicted_labels, zero_division=1)
    print(f"Precision: {precision:.4f}%")

    # Recall
    recall = recall_score(pit_true_labels, pit_predicted_labels)
    print(f"Recall: {recall:.4f}%")

    # F1 Score
    f1 = f1_score(pit_true_labels, pit_predicted_labels)
    print(f"F1 Score: {f1:.4f}%")

    # LAP TIME

def training_loop(model, laps):
    h_s = torch.zeros(1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(1, HIDDEN_SIZE).to(device)
    model_pit_decisions = []
    model_lap_time_predictions = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, c_s, pit_decision, lap_time_prediction = model(laps[:,lap], h_s, c_s)
        model_pit_decisions.append(pit_decision)
        model_lap_time_predictions.append(lap_time_prediction)

    h_s = h_s.detach()

    pit_decision_loss = PIT_DECISION_LOSS_FN(torch.tensor(model_pit_decisions, requires_grad=True), laps[:,1:,0].squeeze(0))
    #lap_time_loss = LAP_TIME_LOSS_FN(torch.tensor(model_lap_time_predictions, requires_grad=True), laps[:,1:,1].squeeze(0))

    return pit_decision_loss #+ lap_time_loss

def train():
    model = AutoF1LSTM(INPUT_SIZE, HIDDEN_SIZE)
    model.to(device)

    optim = OPTIM(model.parameters(), lr=LR)

    training_dataset, _, testing_dataset = random_split(DATASET, [0.8, 0.1, 0.1])
    training_dataloader = DataLoader(training_dataset, shuffle=True)

    for epoch in range(EPOCHS):
        for race_data in training_dataloader:
            model.train()

            optim.zero_grad()
            
            loss = training_loop(model, race_data)
            print(loss)

            loss.backward()

            optim.step()
        
        stats(testing_dataset, model)

train()