import torch
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
path_save_model = '/fhome/gia03/all_models/CLIP_fine_tuning.pth'
path_save_plot = '/fhome/gia03/loss_plots/CLIP_fine_tuning.png'
path = '/fhome/gia03/Image_Captioning_CV/testing/train'
val_path = '/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/testing/validation/fhome/gia03/Image_Captioning_CV/testing/validation'

# Define the Dataset
class DataSet(Dataset):
    def __init__(self, folder_path, split="train", target_size=(256, 256), transform=None):
        self.folder_path = folder_path
        self.split = split
        self.target_size = target_size
        self.transform = transform
        self.image_paths, self.captions = self.load_data()


    def load_data(self):
        image_paths = []
        captions = []
        with open(f"{self.folder_path}/{self.split}.txt", "r") as file:
            for line in file:
                line = line.strip().split(",")
                image_paths.append(line[0].strip())
                captions.append(line[1])
                
        return image_paths, captions
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = f"{self.folder_path}/{self.image_paths[idx]}"
        caption = self.captions[idx]
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, caption
# Initialize the Dataset and DataLoader
transform = preprocess
train_loader = DataSet(folder_path=path,split = 'train', transform=transform)
train_loader = DataLoader(train_loader, batch_size=64, shuffle=True)
# Initialize the Validation Dataset and DataLoader
val_dataset = DataSet(folder_path=val_path, split='validation', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define Loss Function and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 5
train_losses = []
val_losses = []

# Training Loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for images, captions in tqdm(train_loader):
        images = images.to(device)
        
        # Tokenize and move captions to the correct device
        captions = clip.tokenize(captions).to(device)

        # Forward pass: Obtain features from the model
        image_features, text_features = model(images, captions)

        # Normalize features to compute similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        # Dot product between all image and text features (batch_size x batch_size matrix)
        logits_per_image = image_features @ text_features.t()
        logits_per_text = text_features @ image_features.t()

        # Labels for the contrastive loss (diagonal elements are positives)
        ground_truth = torch.arange(images.shape[0], dtype=torch.long, device=device)

        # Compute the loss
        loss_img = loss_fn(logits_per_image, ground_truth)
        loss_txt = loss_fn(logits_per_text, ground_truth)
        total_loss = (loss_img + loss_txt) / 2

        # Backward pass and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, captions in val_loader:
            # ... existing validation code ...
            total_val_loss += total_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
torch.save(model, path_save_model)
# Specify the directory where you want to save the plot
save_dir = path_save_plot # Replace with your actual save path


# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot to the specified directory
plt.savefig(path_save_plot)
