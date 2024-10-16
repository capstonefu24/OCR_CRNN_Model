# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VietnameseOCRDataset, char_to_index
from model import CRNN
import os


class OCRTrainer:
    def __init__(self, resume_training=False, checkpoint_path="crnn_vietnamese_ocr.pth"):
        # Set up dataset paths
        self.image_folder = "C:/Users/DELL/PycharmProjects/SE173082_ChauMinhNhat/OCR/dataset/img"
        self.label_folder = "C:/Users/DELL/PycharmProjects/SE173082_ChauMinhNhat/OCR/dataset/annotations"

        # Set input parameters
        self.input_channels = 1  # Grayscale images
        self.output_size = 36  # Number of output classes (characters)
        self.hidden_size = 256  # Hidden size of the RNN layers
        self.num_layers = 2  # Number of layers in RNN
        self.image_width = 128  # Assuming images are resized to 128 width in the transforms

        # Load dataset
        self.dataset = VietnameseOCRDataset(self.image_folder, self.label_folder)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        # Initialize model
        self.model = CRNN(self.input_channels, self.output_size, self.hidden_size, self.num_layers, self.image_width)

        # Check if a GPU is available and use it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move the model to the specified device

        # Initialize loss and optimizer
        self.criterion = nn.CTCLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Set checkpoint path
        self.checkpoint_path = checkpoint_path

        # Variables for continuing training
        self.start_epoch = 0
        self.resume_training = resume_training

        # Load from checkpoint if needed
        if resume_training and os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

    def train(self):
        epochs = 10
        self.model.train()

        for epoch in range(self.start_epoch, epochs):
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images = images.float().to(self.device)  # Move images to device
                self.optimizer.zero_grad()

                # Forward pass through the entire model
                logits = self.model(images)  # logits shape: [batch_size, height, output_size]

                # Transpose logits to shape: [height, batch_size, output_size]
                logits = logits.permute(1, 0, 2)  # Now logits is [height, batch_size, output_size]



                # Calculate batch size and logits dimensions
                batch_size = images.size(0)  # Dynamic batch size (should be 1 for your case)
                cnn_height = logits.size(0)  # Number of time steps (height) from the CNN output

                # Compute input lengths based on the actual height
                input_lengths = torch.full(size=(batch_size,), fill_value=cnn_height, dtype=torch.long).to(self.device)

                # Generate target lengths
                target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(self.device)

                # Generate target sequences (convert characters to numerical indices)
                targets = []
                for label in labels:
                    if label:  # Ensure the label is not empty
                        targets.append(torch.tensor([ord(c) - ord('a') + 1 for c in label]))
                    else:
                        targets.append(torch.tensor([]))  # Append an empty tensor if the label is empty

                # Concatenate the targets
                targets = torch.cat(targets).to(self.device)

                # Compute the loss
                try:
                    loss = self.criterion(logits, targets, input_lengths, target_lengths)
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue  # Skip this batch if there is an error

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.dataloader):.4f}")

            # Save checkpoint at the end of each epoch
            self.save_checkpoint(epoch, epoch_loss)

        # Save the final trained model
        torch.save(self.model.state_dict(), "crnn_vietnamese_ocr.pth")
        print("Model saved as crnn_vietnamese_ocr.pth")

    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch + 1,  # Next epoch to start training
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {self.start_epoch} with saved loss {checkpoint['loss']:.4f}")
if __name__ == "__main__":
    print("Starting script...")
    trainer = OCRTrainer()  # Create an instance of the class
    trainer.train()  # Call the train method

