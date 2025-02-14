import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, mean_absolute_percentage_error, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from final_f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(4, "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 12
HIDDEN_SIZE = 128
EPOCHS = 5
LR = 0.005
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 0.001
BATCH_SIZE = 25

OPTIM = torch.optim.Adam
COMPOUND_LOSS_FN = nn.CrossEntropyLoss()

NUM_COMPOUNDS = 5
NUM_TRACK_STATUS = 48
NUM_TEAMS = 10 # Since 2018
NUM_DRIVERS = 46
EMBEDDING_DIMS = 8
NUM_CLASSES = 5  # Multi-class classification

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
            nn.Linear(64, NUM_CLASSES) # Multiple outputs for multi-class classification
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

        return h_s, c_s, compound_decision

def testing_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    model_compound_decisions = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, c_s, compound_decision = model(laps[:,lap].unsqueeze(0), h_s, c_s)
        # Convert logits to actual class predictions
        model_compound_decisions.append(torch.argmax(F.sigmoid(compound_decision), dim=-1))
    
    return model_compound_decisions, laps[:,1:,-4].squeeze(0)

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

    # R-Squared (R² Score)
    r2 = r2_score(true_values, predicted_values)
    print(f"R² Score: {r2:.4f}")

def stats(testing_dataloader, model):
    model.eval()

    real_compound_labels = []
    predicted_compound_labels = []

    # Run through testing data.
    for race_data in testing_dataloader:
        predicted_compound_decisions, real_compound_decisions = testing_loop(model, race_data)
        real_compound_labels.append(real_compound_decisions.numpy())
        # Convert logits to actual class predictions
        predicted_compound_labels.append(torch.cat(predicted_compound_decisions).cpu().numpy().flatten())
    
    real_compound_labels = np.concatenate(real_compound_labels)
    predicted_compound_labels = np.concatenate(predicted_compound_labels).flatten()
    
    print("COMPOUND DECISION METRICS:")
    labeling_stats(real_compound_labels, predicted_compound_labels)

def training_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    model_compound_decisions = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, c_s, compound_decisions = model(laps[:,lap].unsqueeze(0), h_s, c_s)
        model_compound_decisions.append(compound_decisions.view(-1, NUM_CLASSES))

    compound_decision_loss = COMPOUND_LOSS_FN(torch.cat(model_compound_decisions), laps[:, 1:, -4].squeeze(0).long())

    return compound_decision_loss

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

            if idx % BATCH_SIZE == 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optim.step()
                optim.zero_grad()
                print(total_loss)
                total_loss = torch.tensor([0.0])
                stats(testing_dataloader, model)

train()