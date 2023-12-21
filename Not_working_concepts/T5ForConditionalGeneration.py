import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, T5ForConditionalGeneration
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Define paths
'''
data_dir = "/fhome/gia03/Images_split"
train_txt = "/fhome/gia03/Images_split/train/train_images.txt"
val_txt = "/fhome/gia03/Images_split/validation/validation_images.txt"
train_image_dir = "/fhome/gia03/Images_split/train"
val_image_dir = "/fhome/gia03/Images_split/validation"
model_path = "/fhome/gia03/all_models/T5ForCOnditionalGeneration.pth"
loss_output_dir = "/fhome/gia03/loss_plots"
''' 
data_dir = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/testing"
train_txt = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/testing/train/train_images.txt"
val_txt = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/testing/validation/validation_images.txt"
train_image_dir = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/testing/train"
val_image_dir = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/testing/validation"
model_path = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/Models/T5ForCOnditionalGeneration.pth"
loss_output_dir = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/Loss_output/loss_plots"
  
# Hyperparameters
batch_size = 8
max_seq_len = 50
lr = 0.001
epochs = 10

# Preprocess data


def preprocess_data(data_file, image_dir):
  with open(data_file, "r") as f:
    lines = f.readlines()

  images, captions = [], []
  for line in lines:
    image_name, caption = line.strip().split(',', 1)
    images.append(os.path.join(image_dir, image_name))
    captions.append(caption)

  return images, captions

# Tokenize text
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def tokenize(text):
  return tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len)

# Image preprocessing


def preprocess_image(image_path):
  image = Image.open(image_path).convert("RGB")
  image = image.resize((224, 224))
  image = torch.from_numpy(np.array(image)).float()
  image = image.permute(2, 0, 1)
  image = (image / 255.0).unsqueeze(0)
  return image

# Dataset class


class CaptionDataset(Dataset):
  def __init__(self, images, captions):
    self.images = images
    self.captions = captions

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image_path = self.images[idx]
    image = preprocess_image(image_path)
    image = image.long()
    caption = self.captions[idx]
    tokenized_caption = tokenize(caption)


    return image, tokenized_caption

# Create datasets and dataloaders
train_images, train_captions = preprocess_data(train_txt, train_image_dir)
val_images, val_captions = preprocess_data(val_txt, val_image_dir)

train_dataset = CaptionDataset(train_images, train_captions)
val_dataset = CaptionDataset(val_images, val_captions)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()



if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

else:
  device = torch.device("cpu")

model.encoder.to(device)
model.decoder.to(device)
# Train model
train_losses, val_losses = [], []

  # Train loop

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, captions in tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training"):
        images = images.to(device)
        captions = {key: torch.stack(value, dim=0).to(device) for key, value in captions.items()}

        optimizer.zero_grad()

        # Move encoder and decoder to GPU if not done outside the loop
        # model.encoder.to(device)
        # model.decoder.to(device)

        image_features = model.encoder(images)
        decoder_outputs = model.decoder(
            input_ids=captions["input_ids"],
            attention_mask=captions["attention_mask"],
            encoder_hidden_states=image_features.detach()
        )

        # Calculate loss
        loss = loss_fn(decoder_outputs.logits.view(-1, decoder_outputs.logits.shape[-1]),
                       captions["labels"].view(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Save training loss
    train_losses.append(train_loss / len(train_loader))

    # Validation loop
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, captions in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation"):
            images = images.to(device)
            captions = {key: value.to(device) for key, value in captions.items()}

            image_features = model.encoder(images)
            decoder_outputs = model.decoder(
                input_ids=captions["input_ids"],
                attention_mask=captions["attention_mask"],
                encoder_hidden_states=image_features.detach()
            )

            loss = loss_fn(decoder_outputs.logits.view(-1, decoder_outputs.logits.shape[-1]),
                           captions["labels"].view(-1))

            val_loss += loss.item()

    # Save validation loss
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Train and Validation Loss")

# Save plot as jpg
plt.savefig(f"T5ForConditionalGeneration_loss_{epoch}.png") 



# Plot and save loss in specified directory
plt.savefig(os.path.join(loss_output_dir, f"loss_{epoch}.jpg"), format="jpg")



# Save model
  
torch.save(model.state_dict(), os.path.join(model_path, f"model_{epoch}.pth"))
