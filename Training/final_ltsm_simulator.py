import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, DataLoader
from earth_movers_distance import torch_wasserstein_loss
from FocalLoss import FocalLoss
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, mean_absolute_percentage_error, precision_score, recall_score, f1_score

from final_f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(1, "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 4
HIDDEN_SIZE = 5
EPOCHS = 2000
LR = 0.0001

OPTIM = torch.optim.Adam
PIT_DECISION_LOSS_FN = FocalLoss(alpha=0.5)
LAP_TIME_LOSS_FN = torch.nn.MSELoss()
COMPOUND_PREDICTION_LOSS_FN = torch.nn.CrossEntropyLoss()

NUM_COMPOUNDS = 5

class AutoF1LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoF1LSTM, self).__init__()
        self.compound_encoding = nn.Embedding(NUM_COMPOUNDS, 5)
        
        # Input size vector - any embeddings + what embeddings return
        self.ltsm = nn.LSTM(input_size - 1 + 5, hidden_size)

        self.pit_decision = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.lap_time = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        self.compound_prediction = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_COMPOUNDS)
        )

    def forward(self, lap, h_s, c_s):
        compound_tensor = lap[0, 0, 2].long()
        compound_embedding = self.compound_encoding(compound_tensor).unsqueeze(0).unsqueeze(0) # Embed compounds first.
        lap = torch.cat((lap[:, :, :2], lap[:, :, 3:]), dim=2) # Remove value without embedding.
        lap = torch.cat((lap, compound_embedding), dim=2) # Add embedded values.
        
        h_t, (h_s, c_s) = self.ltsm(lap, (h_s, c_s))

        pit_decision = self.pit_decision(h_t)
        lap_time_prediction = self.lap_time(h_t)
        compound_prediction = self.compound_prediction(h_t)

        return h_s, c_s, pit_decision, lap_time_prediction, compound_prediction

def testing_loop(model, laps):
    h_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    model_pit_decisions = []
    model_time_predictions = []
    model_compound_decisions = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, c_s, pit_decision, time_prediction, compound_decision = model(laps[:,lap].unsqueeze(0), h_s, c_s)
        model_pit_decisions.append(torch.sigmoid(pit_decision))
        model_time_predictions.append(time_prediction)
        model_compound_decisions.append(torch.argmax(compound_decision))
    
    return model_pit_decisions, laps[:,1:,0].squeeze(0), \
           model_time_predictions, laps[:,1:,1].squeeze(0), \
           model_compound_decisions, laps[:,1:,2].squeeze(0)

def labeling_stats(true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    # Precision
    precision = precision_score(true_labels, predicted_labels, zero_division=1)
    print(f"Precision: {precision:.4f}")

    # Recall
    recall = recall_score(true_labels, predicted_labels)
    print(f"Recall: {recall:.4f}")

    # F1 Score
    f1 = f1_score(true_labels, predicted_labels)
    print(f"F1 Score: {f1:.4f}")

def continous_stats(true_values, predicted_values):
    mape = mean_absolute_percentage_error(true_values, predicted_values) * 100
    print(f"Mean Absolute Percentage Error: {mape:.4f}")

    msd = np.mean(predicted_values - true_values)
    print(f"Mean Signed Deviation: {msd:.4f}")

    # R-Squared (R² Score)
    r2 = r2_score(true_values, predicted_values)
    print(f"R² Score: {r2:.4f}")

def stats(testing_dataset, model):
    model.eval()

    pit_true_labels = []
    pit_predicted_labels = []
    time_true_values = []
    time_predicted_values = []
    compound_true_values = []
    compound_predicted_values = []

    # Run through testing data.
    training_dataloader = DataLoader(testing_dataset)
    for race_data in training_dataloader:
        model_pit_decisions, real_pit_decisions, model_time_outputs, real_time_outputs, model_compound_outputs, real_compound_outputs = testing_loop(model, race_data)
        pit_true_labels.append(real_pit_decisions.numpy())
        pit_predicted_labels.append(torch.stack(model_pit_decisions).detach().numpy().flatten())
        time_true_values.append(real_time_outputs.numpy())
        time_predicted_values.append(torch.stack(model_time_outputs).detach().numpy().flatten())
        compound_true_values.append(real_compound_outputs.numpy())
        compound_predicted_values.append(torch.stack(model_compound_outputs).detach().numpy().flatten())

    # PIT DECISIONS
    pit_true_labels = np.concatenate(pit_true_labels)
    pit_predicted_labels = np.concatenate(pit_predicted_labels).flatten()
    # Skilearn needs everything in same format.
    pit_predicted_labels = np.array([1.0 if prediction >= 0.3 else 0.0 for prediction in pit_predicted_labels])
    
    print("PIT DECISION METRICS:")
    labeling_stats(pit_true_labels, pit_predicted_labels)

    """# LAP TIME
    print("LAP TIME METRICS")
    time_true_values = np.concatenate(time_true_values)
    time_predicted_values = np.concatenate(time_predicted_values).flatten()
    continous_stats(time_true_values, time_predicted_values)

    # COMPOUND DECISIONS
    print("COMPOUND METRICS")
    time_true_values = np.concatenate(compound_true_values).astype(int)
    time_predicted_values = np.concatenate(compound_predicted_values).flatten().astype(int)
    # TODO: ADD STATS FOR COMPOUND DETECTION"""
   

def training_loop(model, laps):
    h_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    model_pit_decisions = []
    model_lap_time_predictions = []
    model_compound_predictions = []

    laps_in_race = laps.shape[1]
    
    for lap in range(laps_in_race - 1):
        h_s, c_s, pit_decision, lap_time_prediction, compound_prediction = model(laps[:,lap].unsqueeze(0), h_s, c_s)
        model_pit_decisions.append(pit_decision)
        model_lap_time_predictions.append(lap_time_prediction)
        model_compound_predictions.append(compound_prediction)

    h_s = h_s.detach()

    pit_decision_loss = PIT_DECISION_LOSS_FN(torch.tensor(model_pit_decisions, requires_grad=True), laps[:,1:,0].squeeze(0))
    lap_time_loss = LAP_TIME_LOSS_FN(torch.tensor(model_lap_time_predictions, requires_grad=True), laps[:,1:,1].squeeze(0))
    compound_prediction_loss = COMPOUND_PREDICTION_LOSS_FN(torch.cat(model_compound_predictions, dim=0).squeeze(1), laps[:,1:,2].squeeze(0).to(torch.long))

    print(pit_decision_loss)
    print("==========")
    return pit_decision_loss

def train():
    model = AutoF1LSTM(INPUT_SIZE, HIDDEN_SIZE)
    model.to(device)

    optim = OPTIM(model.parameters(), lr=LR)

    training_dataset, _, testing_dataset = random_split(DATASET, [0.8, 0.1, 0.1])
    training_dataloader = DataLoader(training_dataset, shuffle=True)

    for epoch in range(EPOCHS):
        for race_data in training_dataloader:
            optim.zero_grad()
            
            loss = training_loop(model, race_data)

            loss.backward()

            optim.step()
        
            stats(testing_dataset, model)

train()