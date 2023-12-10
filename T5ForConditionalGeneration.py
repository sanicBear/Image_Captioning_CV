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
data_dir = "/fhome/gia03/Images_split"
train_txt = "/fhome/gia03/Images_split/train/train_images.txt"
val_txt = "/fhome/gia03/Images_split/validation/validation_images.txt"
train_image_dir = "/fhome/gia03/Images_split/train"
val_image_dir = "/fhome/gia03/Images_split/validation"
model_path = "/fhome/gia03/all_models/T5ForCOnditionalGeneration.pth"
loss_output_dir = "/fhome/gia03/loss_plots"

# Hyperparameters
batch_size = 32
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
    caption = self.captions[idx]
    tokenized_caption = tokenize(caption)

    image = torch.Tensor(image)
    tokenized_caption = torch.Tensor(tokenized_caption)
    if torch.cuda.is_available():
      image.cuda()
      tokenized_caption.cuda()

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
if torch.cuda.is_available():
  model.cuda()
# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

# Train model
train_losses, val_losses = [], []
for epoch in range(epochs):
  train_loss = 0
  val_loss = 0

  # Train loop
  model.train()
  for images, captions in tqdm(train_loader):

    # Move data to GPU if available
  
    optimizer.zero_grad()

    # Encode images
    image_features = model.encoder(input_ids=images)

    # Decode captions
    decoder_outputs = model.decoder(input_ids=captions["input_ids"], attention_mask=captions["attention_mask"], encoder_hidden_states=image_features)

    # Calculate loss
    loss = loss_fn(decoder_outputs.logits.view(-1, decoder_outputs.logits.shape[-1]), captions["labels"].view(-1))

    # Backpropagation
    loss.backward()
    optimizer.step()

    train_loss += loss.item()


  # Save loss
  train_losses.append(train_loss / len(train_loader))
  val_losses.append(val_loss / len(val_loader))


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
