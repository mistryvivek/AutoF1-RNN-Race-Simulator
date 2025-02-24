import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from f1_dataset import CustomF1Dataloader
from earth_movers_distance import torch_wasserstein_loss
import math

DIM_MODEL = 2
LR = 0.0001
EPOCHS = 2000
NUM_TOKENS = 2
NUM_HEADS = 1
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DROPOUT_P = 0.1
BATCH_SIZE = 50
MAE_LOSS = nn.L1Loss()
OPTIM = torch.optim.Adam

DATASET = CustomF1Dataloader(4, "TyreLife,Compound", "../Data Gathering")

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
class UnifiedModelTransformerMDN(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p):
        super(UnifiedModelTransformerMDN, self).__init__()

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
        
        self.fc_time = nn.Sequential(
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
        time_output = self.fc_time(transformer_out).permute(1, 0, 2)

        return pit_output, time_output
    
def train():
    model = UnifiedModelTransformerMDN(NUM_TOKENS, DIM_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT_P)

    model.to(device)

    optim = OPTIM(model.parameters(), lr=LR)

    training_dataset, testing_dataset, validation_dataset = random_split(DATASET, [0.8, 0.1, 0.1])

    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    data_iteration = iter(training_dataloader) 
    n = 0

    for epoch in range(EPOCHS):
        model.train()

        for inputs, pit_label, time_label in training_dataloader:
            #print(f"Inputs shape: {inputs.shape}")
            #print(f"Pit label shape: {pit_label.shape}")
            #print(f"Time label shape: {time_label.shape}")

            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("Inputs contain NaN or Inf")
                break

            optim.zero_grad()

            src = inputs
            tgt = torch.concat([pit_label, time_label], axis=2)

            pit_output, time_output = model(src, tgt)

            pit_loss = torch_wasserstein_loss(pit_output.squeeze(0), pit_label.squeeze(0))
            time_loss = MAE_LOSS(time_output, time_label)
            total_loss = pit_loss + time_loss

            total_loss.backward()

            optim.step()

            print(total_loss)    

train()
