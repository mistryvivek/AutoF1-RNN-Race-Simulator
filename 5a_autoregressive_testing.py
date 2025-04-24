"""
Scipt used to carry out autoregressive testing.

I saved the testing partition through the final testing run and reloaded it with pickle.

I imported the model from the previous file for better modularity. Loaded the
weights that were saved after the training run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as sp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autoregressive_testing_loop(model, testing_dataloader, lstm=False, turn_off_variables=False):
    INDEX_MAPPING = {
      -8: 'compound_decision',
      0: 'lap_time',
      1: 'speedi1',
      2: 'speedi2',
      3: 'speedfl',
      4: 'speedst',
      5: 'speed_telemetry',
      -10: 'gear_telemetry',  
      6: 'rpm_telemetry',
      7: 'throttle_telemetry',
      11: 'd_ahead',
      12: 'd_behind',
      13: 'pit_decision',
    }

    if turn_off_variables:
       INDEX_MAPPING.pop(11)
       INDEX_MAPPING.pop(13)

    model.eval()  # Set the model to evaluation mode
    lap_time_differences = []
    num_pits = []
    rain_compound_detected = []
    qualified = []
    compound_switches = []
    pit_now_followed_by_pit_count = 0
    total_pit_now_decisions = 0

    for race_data in testing_dataloader:
        h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
        if lstm:
            c_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)

        laps_in_race = race_data.shape[1]
        current_lap = race_data[:, 0, :].unsqueeze(0).clone()  # Get initial lap data
        compound_decisions = []

        actual_lap_time = current_lap[:, 0, 0].item()
        simulated_lap_time = current_lap[:, 0, 0].item()
        pit_counter = 0
        rain_detected = current_lap[:, 0, 26].item() == 1.0
        required_stop_made = False

        for lap in range(laps_in_race - 1):
            # Pass current lap data to the model
            if lstm:
                h_s, c_s, compound_decision, lap_time, speedi1, speedi2, speedfl, speedst, speed_telemetry, rpm_telemetry, gear_telemetry, throttle_telemetry, d_ahead, d_behind, pit_decision = model(current_lap, h_s, c_s)
            else:
                h_s, compound_decision, lap_time, speedi1, speedi2, speedfl, speedst, speed_telemetry, rpm_telemetry, gear_telemetry, throttle_telemetry, d_ahead, d_behind, pit_decision = model(current_lap, h_s)

            # To check for disqualification and rain.
            compound_decisions.append(torch.argmax(F.softmax(compound_decision, dim=-1), dim=-1).item())

            # Need up to date static information.
            previous_lap = current_lap.clone()
            current_lap = race_data[:, lap+1, :].unsqueeze(0).clone()

            if previous_lap[:, 0, 13].item() == 1.0:
                total_pit_now_decisions += 1
                if torch.argmax(F.softmax(compound_decision, dim=-1), dim=-1).item() != previous_lap[:, 0, -8].item():
                    pit_now_followed_by_pit_count += 1

            # Compare lap time improvements at the end.
            simulated_lap_time += lap_time.item()
            actual_lap_time += current_lap[:, 0, 0].item()

            # Keep track of the amount of pit stops made.
            if F.sigmoid(pit_decision).item() >= 0.5:
              pit_counter += 1

            # Check for a rainy race.
            if current_lap[:, 0, 26].item() == 1.0:
              rain_detected = True

            # Start to prepare next laps inputs.
            for idx in INDEX_MAPPING.keys():
               autoregressed_value = locals()[INDEX_MAPPING[idx]]
               if idx == -8 or idx == -10:
                  current_lap[:, 0, idx] = torch.argmax(F.softmax(autoregressed_value, dim=-1), dim=-1).item()
               elif idx == 13:
                  autoregressed_value = int(F.sigmoid(autoregressed_value).item() >= 0.5)
               else:
                  current_lap[:, 0, idx] = autoregressed_value.item()

            # Required Pit Variable
            if previous_lap[:, 0, 30].item() == 1.0:
              current_lap[:, 0, 30] = 1.0
              required_stop_made = True
            elif previous_lap[:, 0, 26].item() == 1.0:
              current_lap[:, 0, 30] = 1.0
              required_stop_made = True
            elif len(set(compound_decisions)) > 1:
              current_lap[:, 0, 30] = 1.0
              required_stop_made = True
            else:
              current_lap[:, 0, 30] = 0.0

            # Tyre Life variable calculation
            if previous_lap[:, 0, 13].item() == 1.0:
              current_lap[:, 0, 14] = 1.0
            else:
              current_lap[:, 0, 14] = previous_lap[:, 0, 14].item() + 1

        lap_time_differences.append(actual_lap_time - simulated_lap_time)
        num_pits.append(pit_counter)
        if rain_detected:
          rain_compound_detected.append(3.0 in compound_decisions or 4.0 in compound_decisions)
        qualified.append(required_stop_made)
        compound_switches.append(np.count_nonzero(np.diff(compound_decisions)))

    total_simulations = len(lap_time_differences)
    faster_simulations = sum(1 for diff in lap_time_differences if diff < 0)  
    faster_percentage = (faster_simulations / total_simulations) * 100
    num_pits = np.array(num_pits)
    compound_switches = np.array(compound_switches)

    print(f"Total simulations {total_simulations}")

    print(f"Percentage of faster simulations: {faster_percentage:.2f}%")

    mode_pits = sp.mode(num_pits)[0]
    mean_pits = np.mean(num_pits)
    iqr_pits = np.quantile(num_pits, 0.75) - np.quantile(num_pits, 0.25)

    total_simulations = len(num_pits)

    print(f"Mode of pits: {mode_pits}")
    print(f"Mean of pits: {mean_pits:.2f}")
    print(f"IQR of pits: {iqr_pits}")

    mode_compound_switches = sp.mode(compound_switches)[0]
    mean_compound_switches = np.mean(compound_switches)
    iqr_compound_switches= np.quantile(compound_switches, 0.75) - np.quantile(compound_switches, 0.25)

    total_simulations = len(compound_switches)

    print(f"Mode of compound switches: {mode_compound_switches}")
    print(f"Mean of compound switches: {mean_compound_switches:.2f}")
    print(f"IQR of compound switches: {iqr_compound_switches}")

    total_wet_races = len(rain_compound_detected)
    wet_tyre_races = sum(rain_compound_detected) 
    wet_tyre_percentage = (wet_tyre_races / total_wet_races) * 100

    print(f"Percentage of wet races with wet tyre usage: {wet_tyre_percentage:.2f}% with the sample size being {total_wet_races}")    

    total_simulations = len(qualified)
    disqualifications = sum(1 for qual in qualified if not qual) 
    disqualification_percentage = (disqualifications / total_simulations) * 100

    print(f"Percentage of simulations leading to disqualification: {disqualification_percentage:.2f}%")

    percentage_of_compound_change_with_pit_decision = (pit_now_followed_by_pit_count / total_pit_now_decisions) * 100

    print(
        f"Percentage of simulations where a pit stop results in a tire change {percentage_of_compound_change_with_pit_decision} with the sample size being {total_pit_now_decisions}")

"""
Example how to run:

print("GRU - Dataset 1")
with open('../GRU - Dataset 1/dataset_state.pkl', 'rb') as f:
    dataset = pickle.load(f)

model = AutoF1GRU(INPUT_SIZE, HIDDEN_SIZE)
model.load_state_dict(torch.load('../GRU - Dataset 1/model_weights_9.pth'))
model.to(device)
"""

