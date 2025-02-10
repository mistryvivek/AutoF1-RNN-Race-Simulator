import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from FocalLoss import FocalLoss
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, mean_absolute_percentage_error, precision_score, recall_score, f1_score

from earth_movers_distance import torch_wasserstein_loss
from final_f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(1, "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 18
HIDDEN_SIZE = 100
EPOCHS = 2000
LR = 0.0001

OPTIM = torch.optim.Adam
PIT_DECISION_LOSS_FN = FocalLoss() # nn.BCEWithLogitsLoss(pos_weight=torch.tensor([35.0]))  # torch.nn.MSELoss() # # FocalLoss() # nn.BCEWithLogitsLoss(pos_weight=torch.tensor([60.0]))#torch_wasserstein_loss FocalLoss(alpha=0.4)#nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0])) #nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))# pos_weight=torch.tensor([30.0]))   
LAP_TIME_LOSS_FN = torch.nn.L1Loss()
COMPOUND_PREDICTION_LOSS_FN = torch.nn.CrossEntropyLoss()

NUM_COMPOUNDS = 5
NUM_TRACK_STATUS = 48
NUM_TEAMS = 10 # Since 2018
NUM_DRIVERS = 46
EMBEDDING_DIMS = 32

class AutoF1LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoF1LSTM, self).__init__()
        self.team_embedding = nn.Embedding(NUM_TEAMS, EMBEDDING_DIMS)
        self.track_status_embedding = nn.Embedding(NUM_TRACK_STATUS, EMBEDDING_DIMS)
        self.driver_embedding = nn.Embedding(NUM_DRIVERS, EMBEDDING_DIMS)
        self.compound_embedding = nn.Embedding(NUM_COMPOUNDS, EMBEDDING_DIMS)
        
        # Input size vector - any embeddings + what embeddings return
        self.lstm = nn.LSTM((input_size - 4) + (EMBEDDING_DIMS * 4), hidden_size) #- 1 + 5

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

        """self.compound_prediction = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_COMPOUNDS)
        )"""

    def forward(self, lap, h_s, c_s):
        """
        compound_tensor = lap[:, :, 2].long()
        compound_embedding = self.compound_encoding(compound_tensor) # Embed compounds first.
        lap = torch.cat((lap[:, :, :2], lap[:, :, 3:]), dim=2) # Remove value without embedding.
        lap = torch.cat((lap, compound_embedding), dim=2) # Add embedded values.
        """
        team_encoded = lap[:, :, -1:].long()
        track_status_encoded = lap[:, :, -2:-1].long()
        driver_encoded = lap[:, :, -3:-2].long()
        compound_encoded = lap[:, :, -4:-3].long()
        team_embedded = self.team_embedding(team_encoded).view(1, 1, EMBEDDING_DIMS)
        track_status_embedded = self.track_status_embedding(track_status_encoded).view(1, 1, EMBEDDING_DIMS)
        driver_embedded = self.driver_embedding(driver_encoded).view(1, 1, EMBEDDING_DIMS)
        compound_encoded = self.compound_embedding(compound_encoded).view(1, 1, EMBEDDING_DIMS)
        h_t, (h_s, c_s) = self.lstm(torch.cat((lap[:, :, :-4], team_embedded, track_status_embedded, driver_embedded, compound_encoded), dim=-1), (h_s, c_s))      

        pit_decision = self.pit_decision(h_t)
        lap_time_prediction = self.lap_time(h_t)
        # compound_prediction = self.compound_prediction(h_t)"""

        return h_s, c_s, pit_decision, lap_time_prediction #, compound_prediction

def testing_loop(model, laps):
    h_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    model_pit_decisions = []
    model_time_predictions = []
    # model_compound_decisions = []"""

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, c_s, pit_decision, time_prediction = model(laps[:,lap].unsqueeze(0), h_s, c_s)
        model_pit_decisions.append(torch.sigmoid(pit_decision))
        model_time_predictions.append(time_prediction)
        # model_compound_decisions.append(torch.argmax(compound_decision))"""
    
    return model_pit_decisions, laps[:,1:,0].squeeze(0), \
       model_time_predictions, laps[:,1:,1].squeeze(0)
    # model_compound_decisions, laps[:,1:,2].squeeze(0)"""

def labeling_stats(true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0.0, 1.0])
    print("Confusion Matrix:")
    print(conf_matrix)

    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    # Precision
    precision = precision_score(true_labels, predicted_labels, zero_division=1, average='weighted')
    print(f"Precision: {precision:.4f}")

    # Recall
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print(f"Recall: {recall:.4f}")

    # F1 Score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
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

    pit_true_labels = []
    pit_predicted_labels = []
    time_true_values = []
    time_predicted_values = []
    compound_true_values = []
    compound_predicted_values = []

    # Run through testing data.
    for race_data in testing_dataloader:
        model_pit_decisions, real_pit_decisions, model_time_outputs, real_time_outputs = testing_loop(model, race_data)
        pit_true_labels.append(real_pit_decisions.numpy())
        pit_predicted_labels.append(torch.stack(model_pit_decisions).detach().numpy().flatten())
        """time_true_values.append(real_time_outputs.numpy())
        time_predicted_values.append(torch.stack(model_time_outputs).detach().numpy().flatten())""
        compound_true_values.append(real_compound_outputs.numpy())
        compound_predicted_values.append(torch.stack(model_compound_outputs).detach().numpy().flatten())"""

    # PIT DECISIONS
    pit_true_labels = np.concatenate(pit_true_labels)
    pit_predicted_labels = np.concatenate(pit_predicted_labels).flatten()
    # Skilearn needs everything in same format.
    pit_predicted_labels = np.array([1.0 if prediction >= 0.5 else 0.0 for prediction in pit_predicted_labels])
    
    print("PIT DECISION METRICS:")
    labeling_stats(pit_true_labels, pit_predicted_labels)

    """
    # LAP TIME
    print("LAP TIME METRICS")
    time_true_values = np.concatenate(time_true_values)
    time_predicted_values = np.concatenate(time_predicted_values).flatten()
    continous_stats(time_true_values, time_predicted_values)

    # COMPOUND DECISIONS
    print("COMPOUND METRICS")
    time_true_values = np.concatenate(compound_true_values).astype(int)
    time_predicted_values = np.concatenate(compound_predicted_values).flatten().astype(int)
    # TODO: ADD STATS FOR COMPOUND DETECTION"""
   

"""def training_loop(model, laps):
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

    pit_decision_loss = pit_decision_loss = PIT_DECISION_LOSS_FN(torch.stack(model_pit_decisions).squeeze(1), laps[:, 1:, 0].squeeze(0))
    #lap_time_loss = LAP_TIME_LOSS_FN(torch.tensor(model_lap_time_predictions, requires_grad=True), laps[:,1:,1].squeeze(0))
    #compound_prediction_loss = COMPOUND_PREDICTION_LOSS_FN(torch.cat(model_compound_predictions, dim=0).squeeze(1), laps[:,1:,2].squeeze(0).to(torch.long))

    print(pit_decision_loss)
    print("==========")
    return pit_decision_loss"""

def training_loop(model, laps):
    h_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    c_s = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
    model_pit_decisions = []
    model_lap_time_predictions = []

    laps_in_race = laps.shape[1]
    
    """real_decisions = []

    #print("SHAPE OF LAPS")
    #print(laps.shape)

    stint_change = laps[:,:,0].squeeze(0)
    pit_stop_indices = (stint_change==1.0).nonzero()

    # print("SHAPE OF STINT CHANGE")
    # print(stint_change.shape)

    for pit_stop_index in pit_stop_indices:
        #print("PIT STOP INDEX")
        #print(pit_stop_index)
        pit_stop_index = pit_stop_index.squeeze(0).item()

        before_pit_stop = random.randint(2, 6)
        after_pit_stop = random.randint(2, 6)
        
        #print("CALCS")
        #print(pit_stop_index - before_pit_stop)
        #print(pit_stop_index + after_pit_stop + 1)
        start_before = max(0, pit_stop_index - before_pit_stop)
        end_after = min(laps.shape[1] - 2, pit_stop_index + after_pit_stop)

        laps_subset = laps[:,start_before:end_after]
        laps_in_race = laps_subset.shape[1]
        real_decisions.append(laps[:,start_before+1:end_after])

        #print("START BEFORE / AFTER")
        #print(start_before, end_after)

        for lap in range(laps_in_race - 1):
            h_s, c_s, pit_decision, lap_time_prediction = model(laps_subset[:,lap].unsqueeze(0), h_s, c_s)
            model_pit_decisions.append(pit_decision.view(-1))
            model_lap_time_predictions.append(lap_time_prediction.view(-1))

    real_decisions = torch.cat((real_decisions), dim=1)"""

    for lap in range(laps_in_race - 1):
        h_s, c_s, pit_decision, lap_time_prediction = model(laps[:,lap].unsqueeze(0), h_s, c_s)
        model_pit_decisions.append(pit_decision.view(-1))
        model_lap_time_predictions.append(lap_time_prediction.view(-1))

    print(model_pit_decisions)
    
    #print(model_pit_decisions)
    pit_decision_loss = PIT_DECISION_LOSS_FN(torch.stack(model_pit_decisions).squeeze(1), laps[:, 1:, 0].squeeze(0))
    # lap_time_loss = LAP_TIME_LOSS_FN(torch.stack(model_lap_time_predictions).squeeze(1), real_decisions[:, :, 1].squeeze(0))

    return pit_decision_loss #"""(lap_time_loss / 60) +""" 

def train():
    model = AutoF1LSTM(INPUT_SIZE, HIDDEN_SIZE)
    model.to(device)

    optim = OPTIM(model.parameters(), lr=LR)

    # Lets try oversampling.
    training_dataset, _, testing_dataset = random_split(DATASET, [0.8, 0.1, 0.1])

    # Extract the label at index 0 for each sequence
    labels = torch.tensor([int(data[1][0]) for data in training_dataset])  # Extract the label from each sequence

    # Step 1: Calculate class frequencies
    class_counts = torch.bincount(labels)  # Count occurrences of each class (0 and 1)
    class_weights = 1.0 / class_counts.float()  # Calculate inverse of class frequency for class weighting

    # Step 2: Assign weights to each sample based on its label
    sample_weights = class_weights[labels]  # Get the weight for each sample based on its class label

    # Step 3: Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(training_dataset), replacement=True)

    training_dataloader = DataLoader(training_dataset, sampler=sampler)
    testing_dataloader = DataLoader(testing_dataset)

    for epoch in range(EPOCHS):
        for race_data in training_dataloader:
            optim.zero_grad()
            
            loss = training_loop(model, race_data)
            print(loss)

            loss.backward()

            """# Print the gradients for each parameter
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradients for {name}: {param.grad}")
                else:
                    print(f"Gradients for {name}: No gradients computed.")"""


            optim.step()

            stats(testing_dataloader, model)
        exit()

train()