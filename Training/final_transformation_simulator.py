import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from final_f1_dataset import CustomF1Dataloader
from earth_movers_distance import torch_wasserstein_loss
import math
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, mean_absolute_percentage_error, precision_score, recall_score, f1_score

DIM_MODEL = 2
LR = 0.0001
EPOCHS = 2000
NUM_TOKENS = 18
NUM_HEADS = 1
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DROPOUT_P = 0.1
BATCH_SIZE = 50
HIDDEN_SIZE = 100

OPTIM = torch.optim.Adam
PIT_DECISION_LOSS_FN = nn.BCEWithLogitsLoss() # FocalLoss() #BinaryPolyLoss(epsilon=0.9) # # nn.BCEWithLogitsLoss(pos_weight=torch.tensor([35.0]))
LAP_TIME_LOSS_FN = torch.nn.L1Loss()
COMPOUND_PREDICTION_LOSS_FN = torch.nn.CrossEntropyLoss()

DATASET = CustomF1Dataloader(4, "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    
# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
class AutoF1Tranformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p):
        super(AutoF1Tranformer, self).__init__()

        self.dim_model = dim_model
        
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )

        self.embedding = nn.Linear(num_tokens, dim_model)

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        
        self.fc_pit = nn.Sequential(
            nn.Linear(dim_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, src, tgt):
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        transformer_out = self.transformer(src, tgt)

        # Final output layers
        pit_output = self.fc_pit(transformer_out).permute(1, 0, 2)

        return pit_output
    
def testing_loop(model, laps):
    src = laps[:, :-1, :]
    tgt = torch.zeros_like(laps[:, 1:, :])

    pit_decision = model(src, tgt)
    pit_decision = torch.sigmoid(pit_decision)  # Convert logits to probabilities

    return pit_decision, tgt[:, :, 0]

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
        model_pit_decisions, real_pit_decisions = testing_loop(model, race_data)
        pit_true_labels.append(real_pit_decisions.numpy())
        pit_predicted_labels.append(model_pit_decisions.detach().numpy().flatten())
        """time_true_values.append(real_time_outputs.numpy())
        time_predicted_values.append(torch.stack(model_time_outputs).detach().numpy().flatten())""
        compound_true_values.append(real_compound_outputs.numpy())
        compound_predicted_values.append(torch.stack(model_compound_outputs).detach().numpy().flatten())"""

    print(pit_true_labels)
    print(pit_predicted_labels)
    # PIT DECISIONS
    pit_true_labels = np.stack(pit_true_labels)
    pit_predicted_labels = np.stack(pit_predicted_labels)
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
    model_pit_decisions = []
    src = laps[:, :-1, :]  # Input sequence (exclude last lap)
    tgt = laps[:, 1:, :]   # Target sequence (shifted by 1 lap)
    
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

    pit_decision = model(src, tgt)
        # model_lap_time_predictions.append(lap_time_prediction.view(-1))

    #print(model_pit_decisions)
    
    pit_decision_loss = PIT_DECISION_LOSS_FN(pit_decision.squeeze(-1), laps[:, 1:, 0])
    # lap_time_loss = LAP_TIME_LOSS_FN(torch.stack(model_lap_time_predictions).squeeze(1), real_decisions[:, :, 1].squeeze(0))

    return pit_decision_loss #"""(lap_time_loss / 60) +""" 

def train():
    model = AutoF1Tranformer(NUM_TOKENS, DIM_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT_P)
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
            print(epoch, loss)

            loss.backward()

            """# Print the gradients for each parameter
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradients for {name}: {param.grad}")
                else:
                    print(f"Gradients for {name}: No gradients computed.")"""


            optim.step()

            stats(testing_dataloader, model)

train()