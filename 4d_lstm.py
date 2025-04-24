"""
Script used to train the AutoF1LSTM model for F1 telemetry data prediction.

If you want to use the preloaded serialized tensors used:
DATASET = torch.load(*Insert filename here*, weights_only=False)

OR 
load from the CustomF1Dataloader class:
DATASET = CustomF1Dataloader(*Insert file path of all csvs*, *dataset subset (1/2/3/4))

Most the functions are in shared_functions.py, and everything should already imported.
"""
import torch
import torch.nn as nn
from final_f1_dataset import CustomF1Dataloader
from shared_functions import (
    device, NUM_COMPOUNDS, NUM_TRACK_STATUS, NUM_TEAMS, NUM_DRIVERS, NUM_GEARS, NUM_STATUS, EMBEDDING_DIMS,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, train
)

DATASET = torch.load("DataloadersSaved/dataset_2.pt", weights_only=False)

INPUT_SIZE = 41

class AutoF1LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoF1LSTM, self).__init__()
        self.team_embedding = nn.Embedding(NUM_TEAMS, EMBEDDING_DIMS)
        self.track_status_embedding = nn.Embedding(NUM_TRACK_STATUS, EMBEDDING_DIMS)
        self.driver_embedding = nn.Embedding(NUM_DRIVERS, EMBEDDING_DIMS)
        self.compound_embedding = nn.Embedding(NUM_COMPOUNDS, EMBEDDING_DIMS)
        self.status_embedding = nn.Embedding(NUM_STATUS, EMBEDDING_DIMS)
        self.gear_embedding = nn.Embedding(NUM_GEARS, EMBEDDING_DIMS)
        
        # Input size vector - any embeddings + what embeddings return
        self.lstm = nn.LSTM((input_size - 10) + (EMBEDDING_DIMS * 10), hidden_size, num_layers=NUM_LAYERS, dropout=DROPOUT)
        self.layer_norm = nn.LayerNorm(hidden_size) 

        self.compound_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, NUM_COMPOUNDS) # Multiple outputs for multi-class classification
        )

        self.lap_time_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1) # Multiple outputs for multi-class classification
        )

        self.speedi1_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

        self.speedi2_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

        self.speedfl_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

        self.speedst_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

        self.speedtelemetry_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

        self.rpm_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

        self.gear_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, NUM_GEARS)
        )

        self.throttle_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

        self.distance_ahead = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

        self.distance_behind = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

        self.pit_now = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lap, h_s, c_s):
       # Extract categorical values from lap tensor
        team_encoded = lap[:, :, -1].long().to(device)
        track_status1_encoded = lap[:, :, -6].long().to(device)
        track_status2_encoded = lap[:, :, -5].long().to(device)
        track_status3_encoded = lap[:, :, -4].long().to(device)
        track_status4_encoded = lap[:, :, -3].long().to(device)
        track_status5_encoded = lap[:, :, -2].long().to(device)
        driver_encoded = lap[:, :, -7].long().to(device)
        compound_encoded = lap[:, :, -8].long().to(device)
        status_encoded = lap[:, :, -9].long().to(device)
        gear_encoded = lap[:, :, -10].long().to(device)

        track_status1_embedded = self.track_status_embedding(track_status1_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status2_embedded = self.track_status_embedding(track_status2_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status3_embedded = self.track_status_embedding(track_status3_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status4_embedded = self.track_status_embedding(track_status4_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status5_embedded = self.track_status_embedding(track_status5_encoded).view(1, 1, EMBEDDING_DIMS)

        team_embedded = self.team_embedding(team_encoded).view(1, 1, EMBEDDING_DIMS)
        driver_embedded = self.driver_embedding(driver_encoded).view(1, 1, EMBEDDING_DIMS)
        compound_embedded = self.compound_embedding(compound_encoded).view(1, 1, EMBEDDING_DIMS)
        status_embedded = self.status_embedding(status_encoded).view(1, 1, EMBEDDING_DIMS)
        gear_embedded = self.gear_embedding(gear_encoded).view(1, 1, EMBEDDING_DIMS)

        # Concatenate embeddings with raw lap data before feeding into LSTM
        h_t, (h_s, c_s) = self.lstm(
            torch.cat((lap[:, :, :-10].to(device),  # Exclude last 9 categorical columns
                    track_status1_embedded, track_status2_embedded, track_status3_embedded,
                    track_status4_embedded, track_status5_embedded, 
                    team_embedded, driver_embedded, compound_embedded, status_embedded, gear_embedded), dim=-1), 
            (h_s, c_s)
        )

        h_t = self.layer_norm(h_t)

        compound_decision = self.compound_prediction(h_t)
        lap_time = self.lap_time_prediction(h_t)
        speedi1 = self.speedi1_prediction(h_t)
        speedi2 = self.speedi2_prediction(h_t)
        speedfl = self.speedfl_prediction(h_t)
        speedst = self.speedst_prediction(h_t)
        speedtelemetry = self.speedtelemetry_prediction(h_t)
        rpm = self.rpm_prediction(h_t)
        gear = self.gear_prediction(h_t)
        throttle = self.throttle_prediction(h_t)
        distance_ahead = self.distance_ahead(h_t)
        distance_behind = self.distance_behind(h_t)
        pit_now = self.pit_now(h_t)

        return h_s, c_s, compound_decision, lap_time, speedi1, speedi2, speedfl, speedst, speedtelemetry, rpm, gear, throttle, distance_ahead, distance_behind, pit_now

train(AutoF1LSTM(INPUT_SIZE, HIDDEN_SIZE), DATASET, "final_run_without_scheduler_lstm", is_lstm=True)
