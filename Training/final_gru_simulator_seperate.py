import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from spacecutter.models import OrdinalLogisticModel
from spacecutter.losses import CumulativeLinkLoss

from final_f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(2, "../Data Gathering")

device = torch.device('cpu')#('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 41
HIDDEN_SIZE = 128
EPOCHS = 2
LR = 0.005
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 0.005
BATCH_SIZE = 25

OPTIM = torch.optim.Adam

NUM_COMPOUNDS = 5
NUM_TRACK_STATUS = 8
NUM_TEAMS = 10 # Since 2018
NUM_DRIVERS = 40
NUM_POSITIONS = 20 + 1 #(Due to indexing)
NUM_GEARS = 9 # (8 forward/1 reverse)
NUM_STATUS = 5
NUM_BINS = 9
EMBEDDING_DIMS = 8

LAPS_TILL_PIT_LOSS_FN = CumulativeLinkLoss()
BIN_EDGES = torch.tensor([0, 2, 4, 8, 12, 16, 20, 30])

class AutoF1GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoF1GRU, self).__init__()
        self.team_embedding = nn.Embedding(NUM_TEAMS, EMBEDDING_DIMS)
        self.track_status_embedding = nn.Embedding(NUM_TRACK_STATUS, EMBEDDING_DIMS)
        self.driver_embedding = nn.Embedding(NUM_DRIVERS, EMBEDDING_DIMS)
        self.compound_embedding = nn.Embedding(NUM_COMPOUNDS, EMBEDDING_DIMS)
        self.status_embedding = nn.Embedding(NUM_STATUS, EMBEDDING_DIMS)
        
        # Input size vector - any embeddings + what embeddings return
        self.gru = nn.GRU((input_size - 9) + (EMBEDDING_DIMS * 9), hidden_size, num_layers=NUM_LAYERS, dropout=DROPOUT)
        self.layer_norm = nn.LayerNorm(hidden_size) 

        self.laps_till_pit_predictor = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(64, 1) # Multiple outputs for multi-class classification
        )

        self.laps_till_pit = OrdinalLogisticModel(self.laps_till_pit_predictor, NUM_BINS)

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
        h_t, h_s = self.gru(
            torch.cat((lap[:, :, :-9].to(device),  # Exclude last 9 categorical columns
                    track_status1_embedded, track_status2_embedded, track_status3_embedded,
                    track_status4_embedded, track_status5_embedded, 
                    team_embedded, driver_embedded, compound_embedded, status_embedded), dim=-1), 
            h_s
        )


        h_t = self.layer_norm(h_t)

        laps_till_pit_decision = self.laps_till_pit(h_t.squeeze(0))

        return h_s, laps_till_pit_decision

def testing_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    laps_till_pit_decisions = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, lap_till_pit = model(laps[:,lap].unsqueeze(0), h_s)
        # Convert logits to actual class predictions
        print(lap_till_pit)
        print(torch.argmax(lap_till_pit, dim=-1))
        laps_till_pit_decisions.append(torch.argmax(lap_till_pit, dim=-1))
        exit()
    
    return laps_till_pit_decisions, laps[:,1:,1].squeeze(0)

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

def stats(testing_dataloader, model):
    model.eval()

    real_laps_till_pit_decisions = []
    pred_laps_till_pit_decisions = []
    mandatory_stops_made = []
    mandatory_stops_counts = []
    
    # Run through testing data.
    for race_data in testing_dataloader:
        race_data[:, :, 1] = torch.bucketize(race_data[:, :, 1].contiguous(), BIN_EDGES)
        predicted_till_pit, real_till_pit = testing_loop(model, race_data)
        predicted_till_pit = torch.cat(predicted_till_pit).cpu().detach().numpy().flatten()

        # ================== Laps Till Pit Prediction ==================
        real_laps_till_pit_decisions.append(real_till_pit.cpu().numpy())
        # Convert logits to actual class predictions
        pred_laps_till_pit_decisions.append(predicted_till_pit)
        # Checking for DSQs.
        """mandatory_stops_made.append(any(predicted_till_pit == 1.0))
        mandatory_stops_counts.append(sum(predicted_till_pit == 1.0))"""

    real_laps_till_pit_decisions = np.concatenate(real_laps_till_pit_decisions)
    pred_laps_till_pit_decisions = np.concatenate(pred_laps_till_pit_decisions).flatten()

    print("LAP TILL PIT DECISION METRICS:")
    laps_till_prec = labeling_stats(real_laps_till_pit_decisions, pred_laps_till_pit_decisions)
    """print("MANDATORY STOPS MADE:")
    print(np.mean(mandatory_stops_made))
    print("AVERAGE STOPS MADE:")
    print(np.mean(mandatory_stops_counts))"""

    model.train()
    return [laps_till_prec]

def training_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    model_laps_till_pit = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, lap_till_pit = model(laps[:,lap].unsqueeze(0), h_s)
        model_laps_till_pit.append(lap_till_pit)

    model_laps_till_pit = torch.cat(model_laps_till_pit).to(device)

    laps_till_loss = LAPS_TILL_PIT_LOSS_FN(
        model_laps_till_pit,
        laps[:, 1:, 1].view(model_laps_till_pit.shape[0], 1).long().to(device)
    )

    return laps_till_loss

def validation_loss(model, validation_dataloader):
    validation_loss = torch.tensor([0.0], requires_grad=True).to(device)

    for _, race_data in enumerate(validation_dataloader):
        race_data[:, :, 1] = torch.bucketize(race_data[:, :, 1].contiguous(), BIN_EDGES)
        laps_in_race = race_data.shape[1]

        h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
        model_laps_till_pit = []

        for lap in range(laps_in_race - 1):
            h_s, lap_till = model(race_data[:,lap].unsqueeze(0), h_s)
            model_laps_till_pit.append(lap_till)

        model_laps_till_pit = torch.cat(model_laps_till_pit).to(device)

        laps_till_loss = LAPS_TILL_PIT_LOSS_FN(
            model_laps_till_pit,
            race_data[:, 1:, 1].view(model_laps_till_pit.shape[0], 1).long().to(device)
        )
        validation_loss = validation_loss + laps_till_loss
    
    model.train()
    return validation_loss

def plot_graph(experiment_id, losses, laps_till_preds):
    # Convert lists of 1D arrays to 2D arrays
    laps_till_preds = np.vstack(laps_till_preds)  # Shape: (num_samples, num_features)

    # Ensure correct shape for losses
    losses = np.array(losses).squeeze()

    # Create a single plot
    plt.figure(figsize=(8, 6))

    # Plot loss vs. accuracy for "Compound"
    plt.plot(losses, laps_till_preds, marker='o', linestyle='-', color='blue', label="Laps Till Pit")

    # Set labels and title
    plt.title("Loss vs. Accuracy for Laps Till Pit Decision")
    plt.xlabel("Loss")
    plt.ylabel("Accuracy")
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(experiment_id)
    plt.show()

def train(experiment_id):
    model = AutoF1GRU(INPUT_SIZE, HIDDEN_SIZE)
    model.to(device)

    loss_values = []
    laps_till_preds = []

    optim = OPTIM(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2)

    training_dataset, validation_dataloader, _ = random_split(DATASET, [0.8, 0.1, 0.1])
    iter_counter = 1

    training_dataloader = DataLoader(training_dataset, shuffle=True)
    validation_dataloader = DataLoader(validation_dataloader)

    model.train()
    total_loss = torch.tensor([0.0], requires_grad=True).to(device)

    for epoch in range(EPOCHS):
        for _, race_data in enumerate(training_dataloader):
            # Explain why here
            race_data[:, :, 1] = torch.bucketize(race_data[:, :, 1].contiguous(), BIN_EDGES)
            total_loss = total_loss + training_loop(model, race_data)

            if iter_counter % BATCH_SIZE == 0 or (iter_counter == len(training_dataset) * EPOCHS):
                total_loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"No gradient found for {name}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optim.step()
                optim.zero_grad()
                if iter_counter % (BATCH_SIZE * 8) == 0:
                    loss_values.append(total_loss.cpu().detach().numpy().copy())
                    laps_till_pred = stats(validation_dataloader, model)
                    stats(training_dataloader, model)
                    laps_till_preds.append(laps_till_pred)
                print(f"\n LOSS: {total_loss}") 
                total_loss = torch.tensor([0.0], requires_grad=True).to(device)      

                val_loss = validation_loss(model, validation_dataloader)
                scheduler.step(val_loss)           
            
            iter_counter += 1

    plot_graph(experiment_id, loss_values, laps_till_preds)

"""for LR in [0.001, 0.005, 0.0001]:
    print(f"HIDDEN_SIZE {HIDDEN_SIZE}, EPOCHS {EPOCHS}, LR {LR}, NUM_LAYERS {NUM_LAYERS}, DROPOUT {DROPOUT}, WEIGHT_DECAY {WEIGHT_DECAY}, BATCH_SIZE {BATCH_SIZE}, EMBEDDING_DIMS {EMBEDDING_DIMS} \n")
    train(f"Laps_till_lr_{LR}_gru".replace(".", "_"))"""

"""for HIDDEN_SIZE in [64, 128, 256]:
    print(f"HIDDEN_SIZE {HIDDEN_SIZE}, EPOCHS {EPOCHS}, LR {LR}, NUM_LAYERS {NUM_LAYERS}, DROPOUT {DROPOUT}, WEIGHT_DECAY {WEIGHT_DECAY}, BATCH_SIZE {BATCH_SIZE}, EMBEDDING_DIMS {EMBEDDING_DIMS} \n")
    train(f"Laps_till_hs_{HIDDEN_SIZE}_gru".replace(".", "_"))"""

"""for NUM_LAYERS in [1, 2, 4]:
    print(f"HIDDEN_SIZE {HIDDEN_SIZE}, EPOCHS {EPOCHS}, LR {LR}, NUM_LAYERS {NUM_LAYERS}, DROPOUT {DROPOUT}, WEIGHT_DECAY {WEIGHT_DECAY}, BATCH_SIZE {BATCH_SIZE}, EMBEDDING_DIMS {EMBEDDING_DIMS} \n")
    train(f"Laps_till_layer_{NUM_LAYERS}_gru".replace(".", "_"))"""

for BATCH_SIZE in [12, 24, 48]:
    print(f"HIDDEN_SIZE {HIDDEN_SIZE}, EPOCHS {EPOCHS}, LR {LR}, NUM_LAYERS {NUM_LAYERS}, DROPOUT {DROPOUT}, WEIGHT_DECAY {WEIGHT_DECAY}, BATCH_SIZE {BATCH_SIZE}, EMBEDDING_DIMS {EMBEDDING_DIMS} \n")
    train(f"Laps_till_batch_{BATCH_SIZE}_gru".replace(".", "_"))
