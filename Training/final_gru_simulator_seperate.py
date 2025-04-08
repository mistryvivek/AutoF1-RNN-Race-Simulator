import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from earth_movers_distance import torch_wasserstein_loss

from final_f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(2, "../Data Gathering")

device = torch.device('cpu')#('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 41
HIDDEN_SIZE = 512  # Increased hidden size
EPOCHS = 2  # Increased number of epochs
LR = 0.05  # Adjusted learning rate
NUM_LAYERS = 2  # Increased number of layers
DROPOUT = 0.3  # Increased dropout
WEIGHT_DECAY = 0.01
BATCH_SIZE = 25

OPTIM = torch.optim.Adam  # Changed optimizer to AdamW

NUM_COMPOUNDS = 5
NUM_TRACK_STATUS = 8
NUM_TEAMS = 10 # Since 2018
NUM_DRIVERS = 40
NUM_POSITIONS = 20 + 1 #(Due to indexing)
NUM_GEARS = 9 # (8 forward/1 reverse)
NUM_STATUS = 5
NUM_BINS = 11
EMBEDDING_DIMS = 8

COMPOUND_LOSS_FN = nn.CrossEntropyLoss()

def PIT_NOW_LOSS_FN(output, target):
    # Create a weight tensor where class 0 gets weight 0 and class 1 gets weight 25.0
    weights = torch.where(target == 1.0, torch.tensor(34.0), torch.tensor(1.0)).to(device)

    # Compute the loss using the 'weight' parameter, applying 25.0 weight to class 1 and zeroing out class 0
    loss = torch_wasserstein_loss(output, target, weights=weights)
    return loss

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

        self.laps_till_pit = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_size, 128),  # Increased layer size
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(128, 1),  # Multiple outputs for multi-class classification,
            nn.ReLU()
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
        h_t, h_s = self.gru(
            torch.cat((lap[:, :, :-9].to(device),  # Exclude last 9 categorical columns
                    track_status1_embedded, track_status2_embedded, track_status3_embedded,
                    track_status4_embedded, track_status5_embedded, 
                    team_embedded, driver_embedded, compound_embedded, status_embedded), dim=-1), 
            h_s
        )

        h_t = self.layer_norm(h_t)

        laps_till_pit_decision = self.laps_till_pit(h_t)
        # compound_decision = self.compound_prediction(h_t)

        return h_s, laps_till_pit_decision

def testing_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    laps_till_pit_decisions = []
   # model_compound_decisions = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, lap_till_pit = model(laps[:,lap].unsqueeze(0), h_s)
        # Convert logits to actual class predictions
        laps_till_pit_decisions.append(F.sigmoid(lap_till_pit) >= 0.5)
       # model_compound_decisions.append(torch.argmax(F.log_softmax(compound_decisions, dim=-1), dim=-1))
    
    return laps_till_pit_decisions, laps[:,1:,14].squeeze(0) # , model_compound_decisions, laps[:,1:,-8].squeeze(0).long().to(device)

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

    return precision

def stats(testing_dataloader, model):
    model.eval()

    real_laps_till_pit_decisions = []
    pred_laps_till_pit_decisions = []
    
    """real_compound_labels = []
    predicted_compound_labels = []"""
    
    # Run through testing data.
    for race_data in testing_dataloader:
        predicted_till_pit, real_till_pit = testing_loop(model, race_data)
        predicted_till_pit = torch.cat(predicted_till_pit).cpu().detach().numpy().flatten()

        real_laps_till_pit_decisions.append(real_till_pit.cpu().numpy())
        # Convert logits to actual class predictions
        pred_laps_till_pit_decisions.append(predicted_till_pit)
        
        """real_compound_labels.append(real_compound_decisions.cpu().numpy())
        # Convert logits to actual class predictions
        predicted_compound_labels.append(torch.cat(predicted_compound_decisions).cpu().numpy().flatten())"""

    real_laps_till_pit_decisions = np.concatenate(real_laps_till_pit_decisions)
    pred_laps_till_pit_decisions = np.concatenate(pred_laps_till_pit_decisions).flatten()
    """real_compound_labels = np.concatenate(real_compound_labels)
    predicted_compound_labels = np.concatenate(predicted_compound_labels).flatten()"""

    print("LAP TILL PIT DECISION METRICS:")
    laps_till_pred = labeling_stats(real_laps_till_pit_decisions, pred_laps_till_pit_decisions)
    """print("COMPOUND DECISION METRICS:")
    compound_precision = labeling_stats(real_compound_labels, predicted_compound_labels)"""

    model.train()
    return [laps_till_pred] #, compound_precision]

def training_loop(model, laps):
    h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
    model_laps_till_pit = []
   # model_compound_decisions = []

    laps_in_race = laps.shape[1]

    for lap in range(laps_in_race - 1):
        h_s, lap_till_pit = model(laps[:,lap].unsqueeze(0), h_s)
        model_laps_till_pit.append(lap_till_pit)
        #model_compound_decisions.append(compound_decisions.view(-1, NUM_COMPOUNDS))

    laps_till_loss = PIT_NOW_LOSS_FN(
        torch.cat(model_laps_till_pit).view(-1).to(device),
        laps[:, 1:, 14].squeeze(0).to(device)
    )

    """compound_decision_loss = COMPOUND_LOSS_FN(torch.cat(model_compound_decisions).to(device), 
        laps[:, 1:, -8].squeeze(0).long().to(device))"""

    return laps_till_loss #+ compound_decision_loss

def validation_loss(model, validation_dataloader):
    validation_loss = torch.tensor([0.0], requires_grad=True).to(device)

    for _, race_data in enumerate(validation_dataloader):
        # race_data[:, :, 1] = torch.bucketize(race_data[:, :, 1].contiguous(), BIN_EDGES)
        laps_in_race = race_data.shape[1]

        h_s = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(device)
        model_laps_till_pit = []

        for lap in range(laps_in_race - 1):
            h_s, lap_till = model(race_data[:,lap].unsqueeze(0), h_s)
            model_laps_till_pit.append(lap_till)

        laps_till_loss = PIT_NOW_LOSS_FN(
            torch.cat(model_laps_till_pit).view(-1).to(device),
            race_data[:, 1:, 14].squeeze(0).to(device)
        )

        validation_loss = validation_loss + laps_till_loss
    
    model.train()
    return validation_loss

def plot_graph(experiment_id, losses, laps_till_preds):
    # Convert lists of 1D arrays to 2D arrays
    discrete_values = np.vstack(laps_till_preds)  # Shape: (num_samples, num_features)

    DISCRETE_LABELS = ["PitNow"] #["PitNow", "Compound"]
    DISCRETE_COLOURS = ['gray'] #['gray', 'blue']

    # Ensure correct shape for losses
    losses = np.array(losses).squeeze()

    # Create a single plot
    plt.figure(figsize=(8, 6))

    for i, var in enumerate(DISCRETE_LABELS):
        plt.plot(losses, discrete_values[:, i], marker='o', linestyle='-', 
                     color=DISCRETE_COLOURS[i], label=var)

    # Set labels and title
    plt.title("Loss vs. Accuracy for Discrete Variables")
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

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=True)  # Use Adam with amsgrad
    training_dataset, validation_dataloader, _ = random_split(DATASET, [0.8, 0.1, 0.1])
    iter_counter = 1

    training_dataloader = DataLoader(training_dataset, shuffle=True)
    validation_dataloader = DataLoader(validation_dataloader)

    model.train()
    total_loss = torch.tensor([0.0], requires_grad=True).to(device)

    for epoch in range(EPOCHS):
        for _, race_data in enumerate(training_dataloader):
            # Explain why here
            # race_data[:, :, 1] = torch.bucketize(race_data[:, :, 1].contiguous(), BIN_EDGES)
            total_loss = total_loss + training_loop(model, race_data)

            if iter_counter % BATCH_SIZE == 0 or (iter_counter == len(training_dataset) * EPOCHS):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjusted gradient clipping
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"No gradient found for {name}")
                optim.step()
                optim.zero_grad()
                if iter_counter % (BATCH_SIZE * 8) == 0 or (iter_counter == len(training_dataset) * EPOCHS):
                    loss_values.append(total_loss.cpu().detach().numpy().copy())
                    laps_till_pred = stats(validation_dataloader, model)
                    stats(training_dataloader, model)
                    laps_till_preds.append(laps_till_pred)
                print(f"\n LOSS: {total_loss}") 
                total_loss = torch.tensor([0.0], requires_grad=True).to(device)      
            
            iter_counter += 1

    plot_graph(experiment_id, loss_values, laps_till_preds)

# Loop of experiments with different configurations
for hidden_size in [256, 512, 1024]:
    for num_layers in [1, 2, 3]:
        for dropout in [0.2, 0.3, 0.4]:
            HIDDEN_SIZE = hidden_size
            NUM_LAYERS = num_layers
            DROPOUT = dropout
            experiment_id = f"gru_h{hidden_size}_l{num_layers}_d{dropout}".replace(".", "_")
            train(experiment_id)