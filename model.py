#model.py
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_channels, output_size, hidden_size, num_layers, image_width):
        super(CRNN, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce size by half

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Further reduce size
        )

        # Calculate the width after CNN (each pooling reduces the size by half)
        cnn_output_width = image_width // 4  # Two pooling layers, so divide by 2 twice

        # LSTM layers (input_size is the number of features per time step)
        self.rnn = nn.LSTM(input_size=128 * cnn_output_width, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CNN forward pass
        x = self.cnn(x)

        # Rearrange tensor for LSTM
        batch_size, channels, height, width = x.size()  # Get dimensions after CNN
        x = x.permute(0, 2, 3, 1)  # Change shape to [batch_size, height, width, channels]
        x = x.reshape(batch_size, height, -1)  # Reshape to [batch_size, height, width * channels]

        # Pass through LSTM
        rnn_out, _ = self.rnn(x)  # rnn_out shape: [batch_size, height, hidden_size]

        # Pass through fully connected layer for output
        # out = self.fc(rnn_out[:, -1, :])
        out = self.fc(rnn_out)

        return out
