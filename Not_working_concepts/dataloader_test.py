import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchtext.data.utils import get_tokenizer
from torchvision import models, transforms
from torchtext.data import  get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from PIL import Image
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch.nn as nn

train_data = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/testing/train"
captions = '/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/captions.csv'


def adapt_bert(csv_path, caption_column='caption', tokenizer_name='bert-base-uncased'):
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Load the Flickr8k dataset
    df = pd.read_csv(csv_path)

    # Extract captions from the specified column
    captions = df[caption_column].tolist()

    # Tokenize each caption
    tokenized_captions = [tokenizer.encode(caption, add_special_tokens=True) for caption in tqdm(captions)]

    return tokenized_captions, tokenizer

class ImageCaptionDataset(Dataset):
    def __init__(self, folder_path, split="train", target_size=(256, 256), tokenizer=None):
        self.folder_path = folder_path
        self.split = split
        self.target_size = target_size
        self.tokenizer = get_tokenizer("basic_english") if tokenizer is None else tokenizer
        self.image_paths, self.captions = self.load_data()

        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

    def load_data(self):
        image_paths = []
        captions = []
        with open(f"{self.folder_path}/{self.split}.txt", "r") as file:
            for line in file:
                print('****************************+++')
                print(line)
                line = line.strip().split(",")
                print(line)
                image_paths.append(line[0].strip())
                captions.append(self.tokenizer(",".join(line[1:]).strip(),add_special_tokens=True,max_length=256,
    padding="max_length",
    truncation=True))
        return image_paths, captions

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = f"{self.folder_path}/{self.image_paths[idx]}"
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        image = self.transform(image)

        # Convert caption to numerical format if needed
        caption = self.captions[idx]
        # For example, you can use a vocabulary to convert words to indices

        return image, torch.tensor(caption['input_ids']).type(torch.LongTensor)







def visualize_dataset(dataset, num_samples=5):
    fig, axes = plt.subplots(nrows=num_samples, ncols=1, figsize=(8, 2 * num_samples))

    for i in range(num_samples):
        # Randomly select an example from the dataset
        idx = random.randint(0, len(dataset) - 1)
        image, caption = dataset[idx]

        # Display the image
        axes[i].imshow(image.permute(1, 2, 0))
        axes[i].axis('off')

        # Display the caption
        axes[i].set_title("Caption: {}".format(caption), fontsize=10)

    plt.tight_layout()
    plt.show()



#_, tokenizer = adapt_bert(captions)
# Example usage
#folder_path = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/testing/micro"
#dataset = ImageCaptionDataset(folder_path, split="micro",tokenizer=tokenizer)

#Visualize 2 random samples from the dataset
#visualize_dataset(dataset, num_samples=2)






class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()

        # Load pre-trained ResNet-50 as the image feature extractor
        resnet = models.resnet50(pretrained=True)
        # Remove the last fully connected layer of ResNet
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Get the number of features in the output of the ResNet
        num_features = resnet.fc.in_features

        # Linear layer to map ResNet output to the desired embedding size
        self.embedding_layer = nn.Linear(num_features, embed_size)

        # Transformer model for caption generation
        self.transformer = nn.Transformer(d_model=embed_size, nhead=8, num_encoder_layers=num_layers)

        # Linear layer to map transformer output to vocabulary size
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, images, captions):
            
        # Extract image features using ResNet
        features = self.resnet(images)
        features = features.view(features.size(0), -1)

        # Map the features to the desired embedding size
        features = self.embedding_layer(features)

        # Initialize the input for the transformer
        inputs = features.unsqueeze(1)  # Add a sequence dimension

        # Transformer input should be of shape (sequence_length, batch_size, embed_size)
        inputs = inputs.permute(1, 0, 2)

        # Embedding captions
        embeddings = captions.float()

        # Loop through the transformer for autoregressive generation
        for i in range(embeddings.size(1)):
            # Get the current input for this step
            current_input = inputs[-1, :].unsqueeze(0)
            print('························3')
            print(embeddings.size())
            print(current_input.squeeze(-1).size())
            # Apply transformer for the current step
            output_step = self.transformer(current_input.squeeze(-1), embeddings[:, i].unsqueeze(0))

            # Concatenate the output step to the inputs
            inputs = torch.cat([inputs, output_step], dim=0)

        # Permute the output back to (batch_size, sequence_length, embed_size)
        outputs = inputs.permute(1, 0, 2)

        # Fully connected layer
        outputs = self.fc(outputs)


_,tokenizer = adapt_bert(csv_path=captions)
embed_size = 256
hidden_size = 512
vocab_size =  len(tokenizer.vocab)
num_layers = 2 
batch_size = 2
num_epochs = 5
# Instantiate the model
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)

# Instantiate your dataset
dataset = ImageCaptionDataset(train_data, tokenizer = tokenizer)

# Instantiate DataLoader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    # Initialize the epoch loss
    epoch_loss = 0.0

    # Loop over the batches of images and captions
    for images, captions in data_loader:
        # Move the images and captions to the device
        images = images.to(device)
        captions = captions.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        print(captions.size())
        print(images.size())
        # Forward pass
        outputs = model(images, captions)
        print('######################')
        print(outputs.size())
        # Compute the loss
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update the epoch loss
        epoch_loss += loss.item()

    # Print the epoch loss
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(data_loader)}")
