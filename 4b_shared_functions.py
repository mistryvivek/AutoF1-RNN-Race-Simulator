"""This module contains shared functions and constants used across the both the GRU and LSTM architecture."""

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Shared constants
NUM_COMPOUNDS = 5
NUM_TRACK_STATUS = 8
NUM_TEAMS = 10
NUM_DRIVERS = 40
NUM_GEARS = 9
NUM_STATUS = 5
EMBEDDING_DIMS = 8

INPUT_SIZE = 40
HIDDEN_SIZE = 256
EPOCHS = 10
LR = 0.001
WEIGHT_DECAY = 0.0001
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
OPTIM = torch.optim.AdamW

PIT_IN_SCALE = 0.07
GEAR_SCALE = 0.70
COMPOUND_SCALE = 0.07
CONTINOUS_SCALE = 0.53

# Shared loss functions
COMPOUND_LOSS_FN = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.0104, 0.0069, 0.0069, 0.0590, 0.9167]).to(device))
LAP_TIME_LOSS_FN = torch.nn.HuberLoss()
SPEED_LOSS_FN = torch.nn.HuberLoss()
TELEMETRY_RPM_FN = torch.nn.HuberLoss()
TELEMETRY_GEAR_FN = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.636, 0.103, 0.049, 0.147, 0.041, 0.011, 0.005, 0.003, 0.005]).to(device))
TELEMETRY_THROTTLE_FN = torch.nn.HuberLoss()
DISTANCE_LOSS_FN = torch.nn.HuberLoss()
PIT_NOW_LOSS_FN = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([32.0]).to(device))

# Used to generate stats for any discrete classes.
def labeling_stats(true_labels, predicted_labels, pit_now=False):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(true_labels, predicted_labels, zero_division=1, average='weighted')
    print(f"Precision: {precision:.4f}")

    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    print(f"Recall: {recall:.4f}")

    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    print(f"F1 Score: {f1:.4f}")

    if pit_now:
        auc = roc_auc_score(true_labels, predicted_labels)
        print(f"AUC Score: {auc:.4f}")
        return auc

    return accuracy

# Used to generate stats for any continous classes.
def continous_stats(true_values, predicted_values):
    mae_pred = mean_absolute_error(true_values, predicted_values)
    mae_naive = np.mean(np.abs(np.diff(true_values)))
    mase = mae_pred / mae_naive if mae_naive != 0 else np.inf
    print(f"Mean Absolute Error: {mase}")

    msd = np.mean(predicted_values - true_values)
    print(f"Mean Signed Deviation: {msd:.4f}")

    mse = np.mean((predicted_values - true_values) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")

    return mse

# Process a single drivers race through the model and return predictions.
def process_laps(model, laps, is_lstm=True, h_s=None, c_s=None):
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
    pit_decision_simulations = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        if is_lstm:
            h_s, c_s, compound_decisions, lap_times, speedi1, speedi2, speedfl, speedst, speed_telemetry, rpm_telemetry, gear_telemetry, throttle_telemetry, d_ahead, d_behind, pit_decision = model(laps[:, lap].unsqueeze(0), h_s, c_s)
        else:
            h_s, compound_decisions, lap_times, speedi1, speedi2, speedfl, speedst, speed_telemetry, rpm_telemetry, gear_telemetry, throttle_telemetry, d_ahead, d_behind, pit_decision = model(laps[:, lap].unsqueeze(0), h_s)

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
        pit_decision_simulations.append(pit_decision)

    return {
        "h_s": h_s,
        "c_s": c_s,
        "model_compound_decisions": model_compound_decisions,
        "lap_time_simulations": lap_time_simulations,
        "speedi1_simulations": speedi1_simulations,
        "speedi2_simulations": speedi2_simulations,
        "speedfl_simulations": speedfl_simulations,
        "speedst_simulations": speedst_simulations,
        "speedtelemetry_simulations": speedtelemetry_simulations,
        "rpm_telemetry_simulations": rpm_telemetry_simulations,
        "gear_telemetry_simulations": gear_telemetry_simulations,
        "throttle_telemetry_simulations": throttle_telemetry_simulations,
        "distance_ahead_simulations": distance_ahead_simulations,
        "distance_behind_simulations": distance_behind_simulations,
        "pit_decision_simulations": pit_decision_simulations,
    }

# Calculates the weighted loss.
def calculate_loss(model, laps, is_lstm=True):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device) if is_lstm else None

    results = process_laps(model, laps, is_lstm, h_s, c_s)

    return COMPOUND_LOSS_FN(torch.cat(results["model_compound_decisions"]).to(device), laps[:, 1:, -8].squeeze(0).long().to(device)) * COMPOUND_SCALE + \
        LAP_TIME_LOSS_FN(torch.cat(results["lap_time_simulations"]).view(-1).to(device), laps[:, 1:, 0].squeeze(0).to(device)) * CONTINOUS_SCALE + \
        SPEED_LOSS_FN(torch.cat(results["speedi1_simulations"]).view(-1).to(device), laps[:, 1:, 1].squeeze(0).to(device)) * CONTINOUS_SCALE + \
        SPEED_LOSS_FN(torch.cat(results["speedi2_simulations"]).view(-1).to(device), laps[:, 1:, 2].squeeze(0).to(device)) * CONTINOUS_SCALE + \
        SPEED_LOSS_FN(torch.cat(results["speedfl_simulations"]).view(-1).to(device), laps[:, 1:, 3].squeeze(0).to(device)) * CONTINOUS_SCALE + \
        SPEED_LOSS_FN(torch.cat(results["speedst_simulations"]).view(-1).to(device), laps[:, 1:, 4].squeeze(0).to(device)) * CONTINOUS_SCALE + \
        SPEED_LOSS_FN(torch.cat(results["speedtelemetry_simulations"]).view(-1).to(device), laps[:, 1:, 5].squeeze(0).to(device)) * CONTINOUS_SCALE + \
        TELEMETRY_RPM_FN(torch.cat(results["rpm_telemetry_simulations"]).view(-1).to(device), laps[:, 1:, 6].squeeze(0).to(device)) * CONTINOUS_SCALE + \
        TELEMETRY_GEAR_FN(torch.cat(results["gear_telemetry_simulations"]).to(device), laps[:, 1:, -10].squeeze(0).long().to(device)) * CONTINOUS_SCALE + \
        TELEMETRY_THROTTLE_FN(torch.cat(results["throttle_telemetry_simulations"]).view(-1).to(device), laps[:, 1:, 7].squeeze(0).to(device)) * GEAR_SCALE + \
        DISTANCE_LOSS_FN(torch.cat(results["distance_ahead_simulations"]).view(-1), laps[:, 1:, 8].squeeze(0).to(device)) * CONTINOUS_SCALE + \
        DISTANCE_LOSS_FN(torch.cat(results["distance_behind_simulations"]).view(-1), laps[:, 1:, 12].squeeze(0).to(device)) * CONTINOUS_SCALE + \
        PIT_NOW_LOSS_FN(torch.cat(results["pit_decision_simulations"]).view(-1), laps[:, 1:, 13].squeeze(0).to(device)) * PIT_IN_SCALE

# Returns predictions alongside the real values for testing purposes.
def testing_loop(model, laps, is_lstm=True):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device) if is_lstm else None

    results = process_laps(model, laps, is_lstm, h_s, c_s)

    compound_decisions = torch.cat(results["model_compound_decisions"], dim=0)  # Concatenate list of tensors
    gear_decisions = torch.cat(results["gear_telemetry_simulations"], dim=0)  # Concatenate list of tensors

    return torch.argmax(F.log_softmax(compound_decisions, dim=-1), dim=-1), laps[:, 1:, -8].squeeze(0).long().to(device), \
        results["lap_time_simulations"], laps[:, 1:, 0].squeeze(0).to(device), \
        results["speedi1_simulations"], laps[:, 1:, 1].squeeze(0).to(device), \
        results["speedi2_simulations"], laps[:, 1:, 2].squeeze(0).to(device), \
        results["speedfl_simulations"], laps[:, 1:, 3].squeeze(0).to(device), \
        results["speedst_simulations"], laps[:, 1:, 4].squeeze(0).to(device), \
        results["speedtelemetry_simulations"], laps[:, 1:, 5].squeeze(0).to(device), \
        results["rpm_telemetry_simulations"], laps[:, 1:, 6].squeeze(0).to(device), \
        torch.argmax(F.log_softmax(gear_decisions, dim=-1), dim=-1), laps[:, 1:, -10].squeeze(0).long().to(device), \
        results["throttle_telemetry_simulations"], laps[:, 1:, 7].squeeze(0).to(device), \
        results["distance_ahead_simulations"], laps[:, 1:, 11].squeeze(0).to(device), \
        results["distance_behind_simulations"], laps[:, 1:, 12].squeeze(0).to(device), \
        results["pit_decision_simulations"], laps[:, 1:, 13].squeeze(0).to(device)

# Returns the loss across the whoel valiudation set.
def validation_loss(model, validation_dataloader, is_lstm=True):
    validation_loss = torch.tensor([0.0], requires_grad=True).to(device)

    for race_data in validation_dataloader:
        validation_loss += calculate_loss(model, race_data, is_lstm)

    model.train()
    return validation_loss

# Returns the set of metrics for the whole set of data entered in the parameters.
def stats(testing_dataloader, model, is_lstm=True):
    model.eval()

    real_compound_labels = []
    predicted_compound_labels = []
    mandatory_stops_made = []
    mandatory_stops_counts = []

    real_lap_labels = []
    simulated_lap_labels = []
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

    real_pit_labels = []
    simulated_pit_labels = []

    for race_data in testing_dataloader:
        predicted_compound_decisions, real_compound_decisions, predicted_lap_times, real_lap_times, predicted_speed_i1, real_speed_i1, predicted_speed_i2, real_speed_i2, predicted_speed_fl, real_speed_fl, predicted_speed_st, real_speed_st, predicted_speed_telemetry, real_speed_telemetry, predicted_rpm_telemetry, real_rpm_telemetry, predicted_gear_telemetry, real_gear_telemetry, predicted_throttle_telemetry, real_throttle_telemetry, predicted_d_ahead, real_d_ahead, predicted_d_behind, real_d_behind, predicted_pit, real_pit = testing_loop(model, race_data, is_lstm)

        # Compound decision metrics
        real_compound_labels.append(real_compound_decisions.detach().cpu().numpy())
        predicted_compound_labels.append(predicted_compound_decisions.detach().cpu().numpy())
        mandatory_stops_temp = [predicted_compound_decisions[i].item() != predicted_compound_decisions[i - 1].item() for i in range(1, len(predicted_compound_decisions))]
        mandatory_stops_made.append(any(mandatory_stops_temp))
        mandatory_stops_counts.append(sum(mandatory_stops_temp))

        # Lap time metrics
        real_lap_labels.append(real_lap_times.detach().cpu().numpy())
        simulated_lap_labels.append(torch.cat(predicted_lap_times).detach().cpu().numpy().flatten())
        pit_in_indices = [i for i, stop in enumerate(mandatory_stops_temp) if stop]
        lap_time_differences_local = [predicted_lap_times[idx - 1].detach().cpu() - predicted_lap_times[idx].detach().cpu() for idx in pit_in_indices if idx > 0]
        average_pit_time_difference.append(np.mean(lap_time_differences_local) if lap_time_differences_local else 0)

        # Speed metrics
        real_speed_i1_labels.append(real_speed_i1.detach().cpu().numpy())
        simulated_speed_i1_labels.append(torch.cat(predicted_speed_i1).detach().cpu().numpy().flatten())
        real_speed_i2_labels.append(real_speed_i2.detach().cpu().numpy())
        simulated_speed_i2_labels.append(torch.cat(predicted_speed_i2).detach().cpu().numpy().flatten())
        real_speed_fl_labels.append(real_speed_fl.detach().cpu().numpy())
        simulated_speed_fl_labels.append(torch.cat(predicted_speed_fl).detach().cpu().numpy().flatten())
        real_speed_st_labels.append(real_speed_st.detach().cpu().numpy())
        simulated_speed_st_labels.append(torch.cat(predicted_speed_st).detach().cpu().numpy().flatten())
        real_speed_telemetry_labels.append(real_speed_telemetry.detach().cpu().numpy())
        simulated_speed_telemetry_labels.append(torch.cat(predicted_speed_telemetry).detach().cpu().numpy().flatten())

        # Telemetry metrics
        real_rpm_telemetry_labels.append(real_rpm_telemetry.detach().cpu().numpy())
        simulated_rpm_telemetry_labels.append(torch.cat(predicted_rpm_telemetry).detach().cpu().numpy().flatten())
        real_gear_telemetry_labels.append(real_gear_telemetry.detach().cpu().numpy())
        simulated_gear_telemetry_labels.append(predicted_gear_telemetry.detach().cpu().numpy())
        real_throttle_telemetry_labels.append(real_throttle_telemetry.detach().cpu().numpy())
        simulated_throttle_telemetry_labels.append(torch.cat(predicted_throttle_telemetry).detach().cpu().numpy().flatten())

        # Distance metrics
        real_distance_ahead_labels.append(real_d_ahead.detach().cpu().numpy())
        simulated_distance_ahead_labels.append(torch.cat(predicted_d_ahead).detach().cpu().numpy().flatten())
        real_distance_behind_labels.append(real_d_behind.detach().cpu().numpy())
        simulated_distance_behind_labels.append(torch.cat(predicted_d_behind).detach().cpu().numpy().flatten())

        # Pit decision metrics
        real_pit_labels.append((real_pit.detach().cpu().numpy() == 1.0))
        simulated_pit_labels.append((torch.cat(predicted_pit).detach().cpu().numpy().flatten() >= 0.5))

    # Aggregate metrics
    real_compound_labels = np.concatenate(real_compound_labels)
    predicted_compound_labels = np.concatenate(predicted_compound_labels).flatten()
    real_lap_labels = np.concatenate(real_lap_labels)
    simulated_lap_labels = np.concatenate(simulated_lap_labels).flatten()
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
    real_pit_labels = np.concatenate(real_pit_labels)
    simulated_pit_labels = np.concatenate(simulated_pit_labels).flatten()

    print("COMPOUND DECISION METRICS:")
    compound_precision = labeling_stats(real_compound_labels, predicted_compound_labels)
    print("MANDATORY STOPS MADE:")
    print(np.mean(mandatory_stops_made))
    print("AVERAGE STOPS MADE:")
    print(np.mean(mandatory_stops_counts))

    print("LAP TIME METRICS:")
    lap_time_mase = continous_stats(real_lap_labels, simulated_lap_labels)
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

    print("PIT DECISION METRICS:")
    pit_precision = labeling_stats(real_pit_labels, simulated_pit_labels, pit_now=True)

    model.train()
    return ([lap_time_mase, speedi1_mase, speedi2_mase, speedfl_mase, speedst_mase, \
        speedtelemetry_mase, rpm_mase, throttle_mase, distance_ahead, distance_behind], \
        [compound_precision, gear_precision, pit_precision])

# Generates a visualisation showing how loss changes alongside how the predictions change.
def plot_graph(experiment_id, losses, continous_preds, discrete_preds):
    continuous_values = np.vstack(continous_preds)
    discrete_values = np.vstack(discrete_preds)

    CONTINUOUS_LABELS = ["LapTime", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "Speed",
        "RPM",  "Throttle", "Distance Ahead", "Distance Behind"]
    DISCRETE_LABELS = ["Compound", "nGear", "PitNow"]
    CONTINUOUS_COLOURS = ['blue', 'red', 'green', 'orange', 'purple',
        'brown', 'pink', 'magenta', 'yellow',
        'black']
    DISCRETE_COLOURS = ['gray', 'lime', 'blue']

    plt.figure(figsize=(16, 6))

    #losses = np.array([loss.detach().cpu().numpy() for loss in losses]).squeeze()

    fig, (ax_cont, ax_disc, ax_loss_time) = plt.subplots(1, 3, figsize=(16, 6))

    for i, var in enumerate(CONTINUOUS_LABELS):
        ax_cont.plot(losses, continuous_values[:, i], marker='o', linestyle='-',
                     color=CONTINUOUS_COLOURS[i], label=var)

    ax_cont.set_title("Loss vs. MSE for Continuous Variables")
    ax_cont.set_xlabel("Loss")
    ax_cont.set_ylabel("MSE")
    ax_cont.legend()

    for i, var in enumerate(DISCRETE_LABELS):
        ax_disc.plot(losses, discrete_values[:, i], marker='o', linestyle='-',
                     color=DISCRETE_COLOURS[i], label=var)

    ax_disc.set_title("Loss vs. AUC Score for Pit Now / Loss vs. Accuracy for Other Discrete Variables")
    ax_disc.set_xlabel("Loss")
    ax_disc.set_ylabel("AUC Score / Accuracy")
    ax_disc.legend()

    ax_loss_time.plot(np.arange(len(losses)), losses, marker='o', linestyle='-', color='black', label="Loss vs Time")
    ax_loss_time.set_title("Loss vs Time")
    ax_loss_time.set_xlabel("Iteration (Index)")
    ax_loss_time.set_ylabel("Loss")

    plt.tight_layout()

    plt.savefig(experiment_id)
    plt.show()

# Where the model training takes place.
def train(model, dataset, experiment_id, is_lstm=True):
    model.to(device)

    loss_values = []
    continous_preds, discrete_preds = [], []

    optim = OPTIM(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    training_dataset, validation_dataset, _ = random_split(dataset, [0.8, 0.1, 0.1])
    training_dataloader = DataLoader(training_dataset, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset)

    model.train()
    total_loss = torch.tensor([0.0], requires_grad=True).to(device)
    iter_counter = 0  # Initialize iteration counter

    for epoch in range(EPOCHS):
        print(f"Epoch Number: {epoch}")
        for race_data in training_dataloader:
            total_loss = total_loss + calculate_loss(model, race_data, is_lstm)

            if iter_counter != 0 and iter_counter % BATCH_SIZE == 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optim.step()
                optim.zero_grad()

                print(f"\n LOSS: {total_loss.item()}")
                last_loss = total_loss.item()
                total_loss = torch.tensor([0.0], requires_grad=True).to(device)
            iter_counter += 1  # Increment iteration counter

        loss_values.append(last_loss)
        (continous, discrete) = stats(validation_dataloader, model, is_lstm)
        continous_preds.append(continous)
        discrete_preds.append(discrete)

    plot_graph(experiment_id, loss_values, continous_preds, discrete_preds)
