import torch 
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last time step
        return out
    

if __name__ == "__main__":
    # Example usage
    batch_size = 5
    sequence_length = 15
    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    num_layers = 2

    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    sample_input = torch.randn(batch_size, sequence_length, input_dim)  # (batch_size, sequence_length, input_dim)
    output = model(sample_input)
    print(output.shape)  # Expected output shape: (5, 1)