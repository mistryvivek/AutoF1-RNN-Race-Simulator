import torch
import torch.nn as nn
from torch.utils.data import random_split
from f1_dataset import CustomF1Dataloader

HIDDEN_SIZE = 5
LR = 0.0001
ITERATIONS = 10000
INPUT_SIZE = 2
NUM_CLASSES = 5
OPTIM = torch.optim.Adagrad

DATASET = CustomF1Dataloader(1, "TyreLife,Compound", "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnifiedModelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(UnifiedModelRNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        self.fc_pit = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.fc_time = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        h_n = h_n.squeeze(0)
        
        pit_output = self.fc_pit(h_n)
        time_output = self.fc_time(h_n)
        
        return pit_output, time_output
    
def train():
    model = UnifiedModelRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
    model.to(device)

    optim = OPTIM(model.parameters(), lr=LR)

    train_dataset, testing_dataset, validation_dataset = random_split(DATASET, [0.8, 0.1, 0.1])

    print(train_dataset.__getitem__(0))
    print(testing_dataset.__getitem__(0))
    print(validation_dataset.__getitem__(0))

train()
