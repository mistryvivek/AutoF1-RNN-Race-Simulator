import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_percentage_error, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from final_f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(4, "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 22
HIDDEN_SIZE = 128
EPOCHS = 20
LR = 0.001
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 0.005
BATCH_SIZE = 25

OPTIM = torch.optim.Adam
COMPOUND_LOSS_FN = nn.CrossEntropyLoss()
LAP_TIME_LOSS_FN = nn.HuberLoss()
POSITION_LOSS_FN = nn.CrossEntropyLoss()
SPEED_LOSS_FN = nn.HuberLoss()

NUM_COMPOUNDS = 5
NUM_TRACK_STATUS = 48
NUM_TEAMS = 10 # Since 2018
NUM_DRIVERS = 46
NUM_POSITIONS = 20 + 1 #(Due to indexing)
EMBEDDING_DIMS = 8

LAP_TIME_SCALE = 1
COMPOUND_SCALE = 3.0
POS_SCALE = 0.6
SPEED_SCALE = 0.05

class AutoF1LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoF1LSTM, self).__init__()
        self.team_embedding = nn.Embedding(NUM_TEAMS, EMBEDDING_DIMS)
        self.track_status_embedding = nn.Embedding(NUM_TRACK_STATUS, EMBEDDING_DIMS)
        self.driver_embedding = nn.Embedding(NUM_DRIVERS, EMBEDDING_DIMS)
        self.compound_embedding = nn.Embedding(NUM_COMPOUNDS, EMBEDDING_DIMS)
        
        # Input size vector - any embeddings + what embeddings return
        self.lstm = nn.LSTM((input_size - 4) + (EMBEDDING_DIMS * 4), hidden_size, num_layers=NUM_LAYERS, dropout=DROPOUT)
        self.layer_norm = nn.LayerNorm(hidden_size) 

        self.compound_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, NUM_COMPOUNDS) # Multiple outputs for multi-class classification
        )

        self.lap_time_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1) # Multiple outputs for multi-class classification
        )

        self.position_predicted = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, NUM_POSITIONS) # Multiple outputs for multi-class classification
        )

        self.speed_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 4)
        )

    def forward(self, lap, h_s, c_s):
        team_encoded = lap[:, :, -1:].long()
        track_status_encoded = lap[:, :, -2:-1].long()
        driver_encoded = lap[:, :, -3:-2].long()
        compound_encoded = lap[:, :, -4:-3].long()
        team_embedded = self.team_embedding(team_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status_embedded = self.track_status_embedding(track_status_encoded).view(1, 1, EMBEDDING_DIMS)
        driver_embedded = self.driver_embedding(driver_encoded).view(1, 1, EMBEDDING_DIMS)
        compound_embedded = self.compound_embedding(compound_encoded).view(1, 1, EMBEDDING_DIMS)
        h_t, (h_s, c_s) = self.lstm(torch.cat((lap[:, :, :-4], team_embedded, track_status_embedded, driver_embedded, compound_embedded), dim=-1), (h_s, c_s))      
        h_t = self.layer_norm(h_t)

        compound_decision = self.compound_prediction(h_t)
        lap_time = self.lap_time_prediction(h_t)
        position = self.position_predicted(h_t)
        speed = self.speed_prediction(h_t)

        return h_s, c_s, compound_decision, lap_time, position, speed

def testing_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    model_compound_decisions = []
    lap_times_simulated = []
    position_simulated = []
    speed_simulated = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, c_s, compound_decisions, lap_times, position, speed = model(laps[:,lap].unsqueeze(0), h_s, c_s)
        # Convert logits to actual class predictions
        model_compound_decisions.append(torch.argmax(F.sigmoid(compound_decisions), dim=-1))
        lap_times_simulated.append(lap_times)
        position_simulated.append(torch.argmax(F.sigmoid(position), dim=-1))
        speed_simulated.append(speed)

    return model_compound_decisions, laps[:,1:,-4].squeeze(0), lap_times_simulated, laps[:,1:,0].squeeze(0), \
        position_simulated, laps[:,1:,-5].squeeze(0), speed_simulated, laps[:,1:,4:8].squeeze(0)

def labeling_stats(true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    # Precision
    precision = precision_score(true_labels, predicted_labels, zero_division=1, average='weighted')
    print(f"Precision: {precision:.4f}")

    # Recall
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    print(f"Recall: {recall:.4f}")

    # F1 Score
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    print(f"F1 Score: {f1:.4f}")

def continous_stats(true_values, predicted_values):
    mape = mean_absolute_percentage_error(true_values, predicted_values) * 100
    print(f"Mean Absolute Percentage Error: {mape:.4f}")

    msd = np.mean(predicted_values - true_values)
    print(f"Mean Signed Deviation: {msd:.4f}")

    mse = np.mean((predicted_values - true_values) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")

def stats(testing_dataloader, model):
    model.eval()

    real_compound_labels = []
    predicted_compound_labels = []
    mandatory_stops_made = []

    real_lap_labels = []
    simualted_lap_labels = []

    real_position_labels = []
    simulated_position_labels = []

    real_speed_labels = []
    simulated_speed_labels = []

    # Run through testing data.
    for race_data in testing_dataloader:
        predicted_compound_decisions, real_compound_decisions, predicted_lap_times, real_lap_times, predicted_position, real_position, predicted_speed, real_speed = testing_loop(model, race_data)

        # ================== Compound Decision Prediction ==================
        real_compound_labels.append(real_compound_decisions.numpy())
        # Convert logits to actual class predictions
        predicted_compound_labels.append(torch.cat(predicted_compound_decisions).cpu().numpy().flatten())
        # Checking for DSQs.
        first_value = predicted_compound_decisions[0].item()  # Get the value from the first tensor
        mandatory_stop_made = any(tensor.item() != first_value for tensor in predicted_compound_decisions)
        mandatory_stops_made.append(mandatory_stop_made)

        # ================== Lap Time Prediction ==================
        real_lap_labels.append(real_lap_times.numpy())
        simualted_lap_labels.append(torch.cat(predicted_lap_times).detach().cpu().numpy().flatten())

        # ================== Position Prediction ==================
        real_position_labels.append(real_position.numpy())
        simulated_position_labels.append(torch.cat(predicted_position).detach().cpu().numpy().flatten())

        # ================== Speed Prediction ==================
        real_speed_labels.append(real_speed.numpy().flatten())
        simulated_speed_labels.append(torch.cat(predicted_speed).detach().cpu().numpy().flatten())
    
    real_compound_labels = np.concatenate(real_compound_labels)
    predicted_compound_labels = np.concatenate(predicted_compound_labels).flatten()
    real_lap_labels = np.concatenate(real_lap_labels)
    simualted_lap_labels = np.concatenate(simualted_lap_labels).flatten()
    real_position_labels = np.concatenate(real_position_labels)
    simulated_position_labels = np.concatenate(simulated_position_labels).flatten()
    real_speed_labels = np.concatenate(real_speed_labels)
    simulated_speed_labels = np.concatenate(simulated_speed_labels).flatten()

    print("COMPOUND DECISION METRICS:")
    labeling_stats(real_compound_labels, predicted_compound_labels)
    print("MANDATORY STOPS MADE:")
    print(np.mean(mandatory_stops_made))

    print("LAP TIME METRICS:")
    continous_stats(real_lap_labels, simualted_lap_labels)

    print("POSITION METRICS:")
    labeling_stats(real_position_labels, simulated_position_labels)

    print("SPEED METRICS:")
    continous_stats(real_speed_labels, simulated_speed_labels)

def training_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    model_compound_decisions = []
    lap_time_simulations = []
    position_simulations = []
    speed_simulations = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, c_s, compound_decisions, lap_times, position, speed = model(laps[:,lap].unsqueeze(0), h_s, c_s)
        model_compound_decisions.append(compound_decisions.view(-1, NUM_COMPOUNDS))
        lap_time_simulations.append(lap_times)
        position_simulations.append(position.view(-1, NUM_POSITIONS))
        speed_simulations.append(speed.view(-1, 4))

    compound_decision_loss = COMPOUND_LOSS_FN(torch.cat(model_compound_decisions), laps[:, 1:, -4].squeeze(0).long())
    lap_time_loss = LAP_TIME_LOSS_FN(torch.cat(lap_time_simulations).view(-1), laps[:, 1:, 0].squeeze(0))
    position_loss = POSITION_LOSS_FN(torch.cat(position_simulations), laps[:, 1:, -5].squeeze(0).long())
    speed_loss = SPEED_LOSS_FN(torch.cat(speed_simulations), laps[:, 1:, 4:8].squeeze(0))
    return lap_time_loss * LAP_TIME_SCALE + compound_decision_loss * COMPOUND_SCALE + position_loss * POS_SCALE + speed_loss * SPEED_SCALE

def train():
    model = AutoF1LSTM(INPUT_SIZE, HIDDEN_SIZE)
    model.to(device)

    optim = OPTIM(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    training_dataset, _, testing_dataset = random_split(DATASET, [0.8, 0.1, 0.1])

    training_dataloader = DataLoader(training_dataset, shuffle=True)
    testing_dataloader = DataLoader(testing_dataset)

    model.train()
    total_loss = torch.tensor([0.0])

    for epoch in range(EPOCHS):
        for idx, race_data in enumerate(training_dataloader):
            total_loss = total_loss + training_loop(model, race_data)

            if idx % BATCH_SIZE == 0 and idx > 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optim.step()
                optim.zero_grad()
                print(total_loss)
                total_loss = torch.tensor([0.0])
        
        stats(testing_dataloader, model)

for LAP_TIME_SCALE in [1, 1.5]:
    for COMPOUND_SCALE in [3.0, 4.0]:
        for POS_SCALE in [0.8, 0.6]:
            for SPEED_SCALE in [0.1, 0.05]:
                print(f"Training with scales: {LAP_TIME_SCALE}, {COMPOUND_SCALE}, {POS_SCALE}, {SPEED_SCALE}")
                train()