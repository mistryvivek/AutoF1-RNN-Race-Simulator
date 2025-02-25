import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from final_f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(4, "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 39
HIDDEN_SIZE = 128
EPOCHS = 20
LR = 0.005
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 0.005
BATCH_SIZE = 25

OPTIM = torch.optim.Adam
COMPOUND_LOSS_FN = nn.CrossEntropyLoss()
LAP_TIME_LOSS_FN = nn.HuberLoss()
SPEED_LOSS_FN = nn.HuberLoss()
TELEMETRY_RPM_FN = nn.HuberLoss()
TELEMETRY_GEAR_FN = nn.CrossEntropyLoss()
TELEMETRY_THROTTLE_FN = nn.HuberLoss()
DISTANCE_LOSS_FN = nn.HuberLoss()

NUM_COMPOUNDS = 5
NUM_TRACK_STATUS = 8
NUM_TEAMS = 10 # Since 2018
NUM_DRIVERS = 40
NUM_POSITIONS = 20 + 1 #(Due to indexing)
NUM_GEARS = 9 # (8 forward/1 reverse)
NUM_STATUS = 5
EMBEDDING_DIMS = 8

LAP_TIME_SCALE = 0.1
COMPOUND_SCALE = 1.0
SPEED_SCALE = 0.01
TELEMETRY_RPM_SCALE = 0.01
TELEMETRY_GEAR_SCALE = 0.01
TELEMETRY_THROTTLE_SCALE = 0.1
DISTANCE_SCALE = 0.1

class AutoF1LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoF1LSTM, self).__init__()
        self.team_embedding = nn.Embedding(NUM_TEAMS, EMBEDDING_DIMS)
        self.track_status_embedding = nn.Embedding(NUM_TRACK_STATUS, EMBEDDING_DIMS)
        self.driver_embedding = nn.Embedding(NUM_DRIVERS, EMBEDDING_DIMS)
        self.compound_embedding = nn.Embedding(NUM_COMPOUNDS, EMBEDDING_DIMS)
        self.status_embedding = nn.Embedding(NUM_STATUS, EMBEDDING_DIMS)
        
        # Input size vector - any embeddings + what embeddings return
        self.gru = nn.GRU((input_size - 9) + (EMBEDDING_DIMS * 9), hidden_size, num_layers=NUM_LAYERS, dropout=DROPOUT)
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

        self.speedi1_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1)
        )

        self.speedi2_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1)
        )

        self.speedfl_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1)
        )

        self.speedst_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1)
        )

        self.speedtelemetry_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1)
        )

        self.rpm_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1)
        )

        self.gear_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, NUM_GEARS)
        )

        self.throttle_prediction = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1)
        )

        self.distance_ahead = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1)
        )

        self.distance_behind = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1)
        )

    def forward(self, lap, h_s):
       # Extract categorical values from lap tensor
        team_encoded = lap[:, :, -1:].long().to(device)
        track_status1_encoded = lap[:, :, -5:-4].long().to(device)
        track_status2_encoded = lap[:, :, -4:-3].long().to(device)
        track_status3_encoded = lap[:, :, -3:-2].long().to(device)
        track_status4_encoded = lap[:, :, -2:-1].long().to(device)
        track_status5_encoded = lap[:, :, -6:-5].long().to(device)
        driver_encoded = lap[:, :, -7:-6].long().to(device)
        compound_encoded = lap[:, :, -8:-7].long().to(device)
        status_encoded = lap[:, :, -9:-8].long().to(device)

        track_status1_embedded = self.track_status_embedding(track_status1_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status2_embedded = self.track_status_embedding(track_status2_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status3_embedded = self.track_status_embedding(track_status3_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status4_embedded = self.track_status_embedding(track_status4_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status5_embedded = self.track_status_embedding(track_status5_encoded).view(1, 1, EMBEDDING_DIMS)

        team_embedded = self.team_embedding(team_encoded).view(1, 1, EMBEDDING_DIMS)
        driver_embedded = self.driver_embedding(driver_encoded).view(1, 1, EMBEDDING_DIMS)
        compound_embedded = self.compound_embedding(compound_encoded).view(1, 1, EMBEDDING_DIMS)
        status_embedded = self.status_embedding(status_encoded).view(1, 1, EMBEDDING_DIMS)

        # Concatenate embeddings with raw lap data before feeding into LSTM
        h_t, h_s = self.lstm(
            torch.cat((lap[:, :, :-9].to(device),  # Exclude last 9 categorical columns
                    track_status1_embedded, track_status2_embedded, track_status3_embedded,
                    track_status4_embedded, track_status5_embedded, 
                    team_embedded, driver_embedded, compound_embedded, status_embedded), dim=-1), 
            h_s
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

        return h_s, compound_decision, lap_time, speedi1, speedi2, speedfl, speedst, speedtelemetry, rpm, gear, throttle, distance_ahead, distance_behind

def testing_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    model_compound_decisions = []
    lap_times_simulated = []
    speed_i1_simulated = []
    speed_i2_simulated = []
    speed_fl_simulated = []
    speed_st_simulated = []
    speed_telemetry_simulated = []
    rpm_simulated = []
    gear_simulated = []
    throttle_simulated = []
    distance_ahead_simulated = []
    distance_behind_simulated = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, compound_decisions, lap_times, speedi1, speedi2, speedfl, speedst, speedtelemetry, rpm, gear, throttle, d_ahead, d_behind = model(laps[:,lap].unsqueeze(0), h_s)
        # Convert logits to actual class predictions
        model_compound_decisions.append(torch.argmax(F.log_softmax(compound_decisions, dim=-1), dim=-1))
        lap_times_simulated.append(lap_times)
        speed_i1_simulated.append(speedi1)
        speed_i2_simulated.append(speedi2)
        speed_fl_simulated.append(speedfl)
        speed_st_simulated.append(speedst)
        speed_telemetry_simulated.append(speedtelemetry)
        rpm_simulated.append(rpm)
        gear_simulated.append(torch.argmax(F.log_softmax(gear, dim=-1), dim=-1))
        throttle_simulated.append(throttle)
        distance_ahead_simulated.append(d_ahead)
        distance_behind_simulated.append(d_behind)
    
    return model_compound_decisions, laps[:,1:,-8].squeeze(0).long().to(device), lap_times_simulated, laps[:,1:,0].squeeze(0).to(device), \
        speed_i1_simulated, laps[:,1:,1].squeeze(0).to(device), speed_i2_simulated, laps[:,1:,2].squeeze(0).to(device), \
        speed_fl_simulated, laps[:,1:,3].squeeze(0).to(device), speed_st_simulated, laps[:,1:,4].squeeze(0).to(device), \
        speed_telemetry_simulated, laps[:,1:,5].squeeze(0).to(device), rpm_simulated, laps[:,1:,6].squeeze(0).to(device), \
        gear_simulated, laps[:,1:,7].squeeze(0).long().to(device), throttle_simulated, laps[:,1:,8].squeeze(0).to(device), \
        distance_ahead_simulated, laps[:,1:,12].squeeze(0).to(device), distance_behind_simulated, laps[:,1:,13].squeeze(0).to(device)

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

    return accuracy

def continous_stats(true_values, predicted_values):
    mae_pred = mean_absolute_error(true_values, predicted_values)
    # Compute MAE of naive forecast (using a simple one-step lag)
    mae_naive = np.mean(np.abs(np.diff(true_values)))  # |y_t - y_{t-1}|
    mase = mae_pred / mae_naive if mae_naive != 0 else np.inf
    # Avoid division by zero (if the denominator is zero, MASE is undefined)
    print(f"Mean Absolute Error: {mase}")

    msd = np.mean(predicted_values - true_values)
    print(f"Mean Signed Deviation: {msd:.4f}")

    mse = np.mean((predicted_values - true_values) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")

    return mse

def stats(testing_dataloader, model):
    model.eval()

    real_compound_labels = []
    predicted_compound_labels = []
    mandatory_stops_made = []
    mandatory_stops_counts = []

    real_lap_labels = []
    simualted_lap_labels = []
    average_pit_time_difference = []

    real_speed_i1_labels = []
    simulated_speed_i1_labels = []

    real_speed_i2_labels = []
    simulated_speed_i2_labels = []

    real_speed_fl_labels = []
    simulated_speed_fl_labels = []

    real_speed_st_labels = []
    simulated_speed_st_labels = []

    real_speed_telemetry_labels = []
    simulated_speed_telemetry_labels = []

    real_rpm_telemetry_labels = []
    simulated_rpm_telemetry_labels = []

    real_gear_telemetry_labels = []
    simulated_gear_telemetry_labels = []

    real_throttle_telemetry_labels = []
    simulated_throttle_telemetry_labels = []

    real_distance_ahead_labels = []
    simulated_distance_ahead_labels = []

    real_distance_behind_labels = []
    simulated_distance_behind_labels = []
    
    # Run through testing data.
    for race_data in testing_dataloader:
        predicted_compound_decisions, real_compound_decisions, predicted_lap_times, real_lap_times, predicted_speed_i1, real_speed_i1, predicted_speed_i2, real_speed_i2, predicted_speed_fl, real_speed_fl, predicted_speed_st, real_speed_st, predicted_speed_telemetry, real_speed_telemetry, predicted_rpm_telemetry, real_rpm_telemetry, predicted_gear_telemetry, real_gear_telemetry, predicted_throttle_telemetry, real_throttle_telemetry, predicted_d_ahead, real_d_ahead, predicted_d_behind, real_d_behind = testing_loop(model, race_data)

        # ================== Compound Decision Prediction ==================
        real_compound_labels.append(real_compound_decisions.cpu().numpy())
        # Convert logits to actual class predictions
        predicted_compound_labels.append(torch.cat(predicted_compound_decisions).cpu().numpy().flatten())
        # Checking for DSQs.
        mandatory_stops_temp = []
        for i in range(1, len(predicted_compound_decisions)):
            current_value = predicted_compound_decisions[i].item()  # Extract scalar value from tensor
            previous_value = predicted_compound_decisions[i - 1].item()  # Extract scalar value from the previous tensor

            # Check if the current value differs from the previous value (pit stop or decision change)
            mandatory_stops_temp.append(current_value != previous_value)

        mandatory_stops_made.append(any(mandatory_stops_temp))
        mandatory_stops_counts.append(sum(mandatory_stops_temp))  

        # ================== Lap Time Prediction ==================
        real_lap_labels.append(real_lap_times.cpu().numpy())
        simualted_lap_labels.append(torch.cat(predicted_lap_times).detach().cpu().numpy().flatten())

        # Get the indices where mandatory stops were made (pit-in laps)
        pit_in_indices = [i for i, stop in enumerate(mandatory_stops_temp) if stop]
        lap_time_differences_local = []

        # Loop through each pit-in lap and compute the difference with the previous lap
        for idx in pit_in_indices:
            if idx > 0:
                previous_lap_time = predicted_lap_times[idx - 1]  # Previous lap time
                pit_in_lap_time = predicted_lap_times[idx] 
                
                lap_time_difference = previous_lap_time - pit_in_lap_time
                lap_time_differences_local.append(lap_time_difference.detach().cpu().numpy())

        # Compute the average lap time difference
        average_lap_time_difference_local = np.mean(lap_time_differences_local) if len(lap_time_differences_local) > 0 else 0

        average_pit_time_difference.append(average_lap_time_difference_local)
        
        # ================== Speed Prediction ==================
        real_speed_i1_labels.append(real_speed_i1.cpu().numpy())
        simulated_speed_i1_labels.append(torch.cat(predicted_speed_i1).detach().cpu().numpy().flatten())
        real_speed_i2_labels.append(real_speed_i2.cpu().numpy())
        simulated_speed_i2_labels.append(torch.cat(predicted_speed_i2).detach().cpu().numpy().flatten())
        real_speed_fl_labels.append(real_speed_fl.cpu().numpy())
        simulated_speed_fl_labels.append(torch.cat(predicted_speed_fl).detach().cpu().numpy().flatten())
        real_speed_st_labels.append(real_speed_st.cpu().numpy())
        simulated_speed_st_labels.append(torch.cat(predicted_speed_st).detach().cpu().numpy().flatten())
        real_speed_telemetry_labels.append(real_speed_telemetry.cpu().numpy())
        simulated_speed_telemetry_labels.append(torch.cat(predicted_speed_telemetry).detach().cpu().numpy().flatten())

        # ============== Telemetry Prediction ===================
        real_rpm_telemetry_labels.append(real_rpm_telemetry.cpu().numpy())
        simulated_rpm_telemetry_labels.append(torch.cat(predicted_rpm_telemetry).detach().cpu().numpy().flatten())

        real_gear_telemetry_labels.append(real_gear_telemetry.cpu().numpy())
        simulated_gear_telemetry_labels.append(torch.cat(predicted_gear_telemetry).detach().cpu().numpy().flatten())

        real_throttle_telemetry_labels.append(real_throttle_telemetry.cpu().numpy())
        simulated_throttle_telemetry_labels.append(torch.cat(predicted_throttle_telemetry).detach().cpu().numpy().flatten())

        # =========== Distance Prediction =======================
        real_distance_ahead_labels.append(real_d_ahead.cpu().numpy())
        simulated_distance_ahead_labels.append(torch.cat(predicted_d_ahead).detach().cpu().numpy().flatten())

        real_distance_behind_labels.append(real_d_behind.cpu().numpy())
        simulated_distance_behind_labels.append(torch.cat(predicted_d_behind).detach().cpu().numpy().flatten())

    real_compound_labels = np.concatenate(real_compound_labels)
    predicted_compound_labels = np.concatenate(predicted_compound_labels).flatten()
    real_lap_labels = np.concatenate(real_lap_labels)
    simualted_lap_labels = np.concatenate(simualted_lap_labels).flatten()
    real_speed_i1_labels = np.concatenate(real_speed_i1_labels)
    simulated_speed_i1_labels = np.concatenate(simulated_speed_i1_labels).flatten()
    real_speed_i2_labels = np.concatenate(real_speed_i2_labels)
    simulated_speed_i2_labels = np.concatenate(simulated_speed_i2_labels).flatten()
    real_speed_fl_labels = np.concatenate(real_speed_fl_labels)
    simulated_speed_fl_labels = np.concatenate(simulated_speed_fl_labels).flatten()
    real_speed_st_labels = np.concatenate(real_speed_st_labels)
    simulated_speed_st_labels = np.concatenate(simulated_speed_st_labels).flatten()
    real_speed_telemetry_labels = np.concatenate(real_speed_telemetry_labels)
    simulated_speed_telemetry_labels = np.concatenate(simulated_speed_telemetry_labels).flatten()
    real_rpm_telemetry_labels = np.concatenate(real_rpm_telemetry_labels)
    simulated_rpm_telemetry_labels = np.concatenate(simulated_rpm_telemetry_labels).flatten()
    real_gear_telemetry_labels = np.concatenate(real_gear_telemetry_labels)
    simulated_gear_telemetry_labels = np.concatenate(simulated_gear_telemetry_labels).flatten()
    real_throttle_telemetry_labels = np.concatenate(real_throttle_telemetry_labels)
    simulated_throttle_telemetry_labels = np.concatenate(simulated_throttle_telemetry_labels).flatten()
    real_distance_ahead_labels = np.concatenate(real_distance_ahead_labels)
    simulated_distance_ahead_labels = np.concatenate(simulated_distance_ahead_labels).flatten()
    real_distance_behind_labels = np.concatenate(real_distance_behind_labels)
    simulated_distance_behind_labels = np.concatenate(simulated_distance_behind_labels).flatten()

    print("COMPOUND DECISION METRICS:")
    compound_precision = labeling_stats(real_compound_labels, predicted_compound_labels)
    print("MANDATORY STOPS MADE:")
    print(np.mean(mandatory_stops_made))
    print("AVERAGE STOPS MADE:")
    print(np.mean(mandatory_stops_counts))

    print("LAP TIME METRICS:")
    lap_time_mase = continous_stats(real_lap_labels, simualted_lap_labels)
    print("AVERAGE PIT STOP TIME DIFFERENCE:")
    print(np.mean(average_pit_time_difference))

    print("SPEED METRICS:")
    print("===========I1============")
    speedi1_mase = continous_stats(real_speed_i1_labels, simulated_speed_i1_labels)
    print("===========I2============")
    speedi2_mase = continous_stats(real_speed_i2_labels, simulated_speed_i2_labels)
    print("===========FL============")
    speedfl_mase = continous_stats(real_speed_fl_labels, simulated_speed_fl_labels)
    print("===========ST============")
    speedst_mase = continous_stats(real_speed_st_labels, simulated_speed_st_labels)
    print("TELEMETRY METRICS:")
    print("===========SPEED============")
    speedtelemetry_mase = continous_stats(real_speed_telemetry_labels, simulated_speed_telemetry_labels)
    print("============RPM=============")
    rpm_mase = continous_stats(real_rpm_telemetry_labels, simulated_rpm_telemetry_labels)
    print("============GEAR============")
    gear_precision = labeling_stats(real_gear_telemetry_labels, simulated_gear_telemetry_labels)
    print("==========THROTTLE==========")
    throttle_mase = continous_stats(real_throttle_telemetry_labels, simulated_throttle_telemetry_labels)

    print("DISTANCE TO DRIVER AHEAD METRICS:")
    distance_ahead = continous_stats(real_distance_ahead_labels, simulated_distance_ahead_labels)
    print("DISTANCE TO DRIVER BEHIND METRICS:")
    distance_behind = continous_stats(real_distance_behind_labels, simulated_distance_behind_labels)

    model.train()
    return ([lap_time_mase, speedi1_mase, speedi2_mase, speedfl_mase, speedst_mase, \
        speedtelemetry_mase, rpm_mase, throttle_mase, distance_ahead, distance_behind], \
        [compound_precision, gear_precision])

def training_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    model_compound_decisions = []
    lap_time_simulations = []
    speedi1_simulations = []
    speedi2_simulations = []
    speedfl_simulations = []
    speedst_simulations = []
    speedtelemetry_simulations = []
    rpm_telemetry_simulations = []
    gear_telemetry_simulations = []
    throttle_telemetry_simulations = []
    distance_ahead_simulations = []
    distance_behind_simulations = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, compound_decisions, lap_times, speedi1, speedi2, speedfl, speedst, speed_telemetry, rpm_telemetry, gear_telemetry, throttle_telemetry, d_ahead, d_behind = model(laps[:,lap].unsqueeze(0), h_s)
        model_compound_decisions.append(compound_decisions.view(-1, NUM_COMPOUNDS))
        lap_time_simulations.append(lap_times)
        speedi1_simulations.append(speedi1)
        speedi2_simulations.append(speedi2)
        speedfl_simulations.append(speedfl)
        speedst_simulations.append(speedst)
        speedtelemetry_simulations.append(speed_telemetry)
        rpm_telemetry_simulations.append(rpm_telemetry)
        gear_telemetry_simulations.append(gear_telemetry.view(-1, NUM_GEARS))
        throttle_telemetry_simulations.append(throttle_telemetry)
        distance_ahead_simulations.append(d_ahead)
        distance_behind_simulations.append(d_behind)

    compound_decision_loss = COMPOUND_LOSS_FN(torch.cat(model_compound_decisions).to(device), laps[:, 1:, -8].squeeze(0).long().to(device))
    lap_time_loss = LAP_TIME_LOSS_FN(torch.cat(lap_time_simulations).view(-1).to(device), laps[:, 1:, 0].squeeze(0).to(device))
    speedi1_loss = SPEED_LOSS_FN(torch.cat(speedi1_simulations).view(-1).to(device), laps[:, 1:, 1].squeeze(0).to(device))
    speedi2_loss = SPEED_LOSS_FN(torch.cat(speedi2_simulations).view(-1).to(device), laps[:, 1:, 2].squeeze(0).to(device))
    speedfl_loss = SPEED_LOSS_FN(torch.cat(speedfl_simulations).view(-1).to(device), laps[:, 1:, 3].squeeze(0).to(device))
    speedst_loss = SPEED_LOSS_FN(torch.cat(speedst_simulations).view(-1).to(device), laps[:, 1:, 4].squeeze(0).to(device))
    speedtelemetry_loss = SPEED_LOSS_FN(torch.cat(speedtelemetry_simulations).view(-1).to(device), laps[:, 1:, 5].squeeze(0).to(device))
    rpm_telemetry_loss = TELEMETRY_RPM_FN(torch.cat(rpm_telemetry_simulations).view(-1).to(device), laps[:, 1:, 6].squeeze(0).to(device))
    gear_telemetry_loss = TELEMETRY_GEAR_FN(torch.cat(gear_telemetry_simulations).to(device), laps[:, 1:, 7].squeeze(0).long().to(device))
    throttle_telemetry_loss = TELEMETRY_THROTTLE_FN(torch.cat(throttle_telemetry_simulations).view(-1).to(device), laps[:, 1:, 8].squeeze(0).to(device))
    distance_ahead_loss = DISTANCE_LOSS_FN(torch.cat(distance_ahead_simulations).view(-1), laps[:,1:,12].squeeze(0).to(device))
    distance_behind_loss = DISTANCE_LOSS_FN(torch.cat(distance_behind_simulations).view(-1), laps[:,1:,13].squeeze(0).to(device))
    return lap_time_loss * LAP_TIME_SCALE + compound_decision_loss * COMPOUND_SCALE + speedi1_loss * SPEED_SCALE + speedi2_loss * SPEED_SCALE + speedfl_loss * SPEED_SCALE + speedst_loss * SPEED_SCALE + speedtelemetry_loss * SPEED_SCALE + rpm_telemetry_loss * TELEMETRY_RPM_SCALE + gear_telemetry_loss * TELEMETRY_GEAR_SCALE + throttle_telemetry_loss * TELEMETRY_THROTTLE_SCALE + distance_ahead_loss * DISTANCE_SCALE + distance_behind_loss * DISTANCE_SCALE

def plot_graph(experiment_id, losses, continous_preds, discrete_preds):
    # Convert lists of 1D arrays to 2D arrays
    continuous_values = np.vstack(continous_preds)  # Shape: (num_samples, num_features)
    discrete_values = np.vstack(discrete_preds)    # Shape: (num_samples, num_features)

    CONTINUOUS_LABELS = ["LapTime", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "Speed", 
        "RPM",  "Throttle", "Distance Ahead", "Distance Behind"]  # Fill with labels for continuous variables
    DISCRETE_LABELS = ["Compound", "nGear"]  # Fill with labels for discrete variables
    CONTINUOUS_COLOURS = ['blue', 'red', 'green', 'orange', 'purple', 
        'brown', 'pink', 'magenta', 'yellow',
        'black']  # Fill with colors for continuous variables
    DISCRETE_COLOURS = ['gray', 'lime']  # Fill with colors for discrete variables

    plt.figure(figsize=(16, 6))

    # Ensure correct shape for losses
    losses = np.array(losses).squeeze()

    # Create subplots: 1 row, 3 columns (for continuous, discrete, and loss vs time)
    fig, (ax_cont, ax_disc, ax_loss_time) = plt.subplots(1, 3, figsize=(16, 6))

    # Plot continuous variables
    for i, var in enumerate(CONTINUOUS_LABELS):
        ax_cont.plot(losses, continuous_values[:, i], marker='o', linestyle='-', 
                     color=CONTINUOUS_COLOURS[i], label=var)

    ax_cont.set_title("Loss vs. MSE for Continuous Variables")
    ax_cont.set_xlabel("Loss")
    ax_cont.set_ylabel("MSE")
    ax_cont.legend()

    # Plot discrete variables
    for i, var in enumerate(DISCRETE_LABELS):
        ax_disc.plot(losses, discrete_values[:, i], marker='o', linestyle='-', 
                     color=DISCRETE_COLOURS[i], label=var)

    ax_disc.set_title("Loss vs. Accuracy for Discrete Variables")
    ax_disc.set_xlabel("Loss")
    ax_disc.set_ylabel("Accuracy")
    ax_disc.legend()

    # Plot loss vs index (time step is implicit as loss is in ascending order)
    ax_loss_time.plot(np.arange(len(losses)), losses, marker='o', linestyle='-', color='black', label="Loss vs Time")
    ax_loss_time.set_title("Loss vs Time")
    ax_loss_time.set_xlabel("Iteration (Index)")
    ax_loss_time.set_ylabel("Loss")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(experiment_id)
    plt.show()

def train(experiment_id):
    model = AutoF1LSTM(INPUT_SIZE, HIDDEN_SIZE)
    model.to(device)

    loss_values = []
    continous_preds, discrete_preds = [], []

    optim = OPTIM(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    training_dataset, validation_dataloader, _ = random_split(DATASET, [0.8, 0.1, 0.1])
    iter_counter = 1

    training_dataloader = DataLoader(training_dataset, shuffle=True)
    validation_dataloader = DataLoader(validation_dataloader)

    model.train()
    total_loss = torch.tensor([0.0], requires_grad=True).to(device)

    for epoch in range(EPOCHS):
        for _, race_data in enumerate(training_dataloader):
            total_loss = total_loss + training_loop(model, race_data)

            if iter_counter % BATCH_SIZE == 0 or (iter_counter == len(training_dataset) * EPOCHS):
                total_loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"No gradient found for {name}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optim.step()
                optim.zero_grad()
                loss_values.append(total_loss.cpu().detach().numpy().copy())
                print(f"\n LOSS: {total_loss}") 
                total_loss = torch.tensor([0.0], requires_grad=True).to(device)
                (continous, discrete) = stats(validation_dataloader, model)
                continous_preds.append(continous)
                discrete_preds.append(discrete)
                
            iter_counter += 1

    plot_graph(experiment_id, loss_values, continous_preds, discrete_preds)

print(f"HIDDEN_SIZE {HIDDEN_SIZE}, EPOCHS {EPOCHS}, LR {LR}, NUM_LAYERS {NUM_LAYERS}, DROPOUT {DROPOUT}, WEIGHT_DECAY {WEIGHT_DECAY}, BATCH_SIZE {BATCH_SIZE}, EMBEDDING_DIMS {EMBEDDING_DIMS}, LAP_TIME_SCALE {LAP_TIME_SCALE}, COMPOUND_SCALE {COMPOUND_SCALE}, SPEED_SCALE {SPEED_SCALE}, TELEMETRY_RPM_SCALE {TELEMETRY_RPM_SCALE}, TELEMETRY_GEAR_SCALE {TELEMETRY_GEAR_SCALE}, TELEMETRY_THROTTLE_SCALE {TELEMETRY_THROTTLE_SCALE}, DISTANCE_SCALE {DISTANCE_SCALE}, OPTIM {OPTIM}, COMPOUND_LOSS_FN {COMPOUND_LOSS_FN}, LAP_TIME_LOSS_FN {LAP_TIME_LOSS_FN}, SPEED_LOSS_FN {SPEED_LOSS_FN}, TELEMETRY_RPM_FN {TELEMETRY_RPM_FN}, TELEMETRY_GEAR_FN {TELEMETRY_GEAR_FN}, TELEMETRY_THROTTLE_FN {TELEMETRY_THROTTLE_FN}, DISTANCE_LOSS_FN {DISTANCE_LOSS_FN} \n")

LOSS_FN_OPTIONS = {
    "COMPOUND_LOSS_FN": [nn.CrossEntropyLoss(), nn.MultiMarginLoss()],
    "LAP_TIME_LOSS_FN": [nn.HuberLoss(), nn.MSELoss(), nn.L1Loss()],
    "SPEED_LOSS_FN": [nn.HuberLoss(), nn.MSELoss(), nn.L1Loss()],
    "TELEMETRY_RPM_FN": [nn.HuberLoss(), nn.MSELoss(), nn.L1Loss()],
    "TELEMETRY_GEAR_FN": [nn.CrossEntropyLoss(), nn.MultiMarginLoss()],  # Gear is categorical
    "TELEMETRY_THROTTLE_FN": [nn.HuberLoss(), nn.MSELoss(), nn.L1Loss()],
    "DISTANCE_LOSS_FN": [nn.HuberLoss(), nn.MSELoss(), nn.L1Loss()],
}


for key in LOSS_FN_OPTIONS.keys():
    for loss_fn in LOSS_FN_OPTIONS[key]:
        globals()[key]  = loss_fn
        print(f"LOSS FN: {loss_fn} \n")
        train(f"{key}_{loss_fn}")

print("Dataset 4")
train("Dataset 4 GRU")
DATASET = CustomF1Dataloader(3, "../Data Gathering")
print("Dataset 3")
train("Dataset 3 GRU")
DATASET = CustomF1Dataloader(2, "../Data Gathering")
print("Dataset 2")
train("Dataset 2 GRU")
DATASET = CustomF1Dataloader(1, "../Data Gathering")
print("Dataset 1")
train("Dataset 1 GRU")