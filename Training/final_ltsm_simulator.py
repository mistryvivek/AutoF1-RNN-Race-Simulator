import torch
import torch.nn as nn

from f1_dataset import CustomF1Dataloader

DATASET = CustomF1Dataloader(4, "TyreLife,Compound,SpeedI1,SpeedI2,SpeedFL,SpeedST,DRS,DriverNumber,Team,TrackStatus,AirTemp,Humidity,Pressure,Rainfall,TrackTemp,WindDirection,WindSpeed", "../Data Gathering")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnifiedModelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pit_choices_num):
        super(UnifiedModelRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.ltsm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        pass

    