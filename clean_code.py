import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

# Read and display data
image_captions_file = '/fhome/gia03/Image_Captioning_CV/testing/train/train.txt'
caption_data = pd.read_csv(image_captions_file)
print("There are {} image captions".format(len(caption_data)))
caption_data.head(7)

# Display an image
selected_image_idx = 1
image_file_path = '/fhome/gia03/Image_Captioning_CV/testing/train/'+caption_data.iloc[selected_image_idx,0]
image = mpimg.imread(image_file_path)
plt.imshow(image)
plt.show()

# Display captions for an image
for index in range(selected_image_idx, selected_image_idx+2):
    print("Caption:", caption_data.iloc[index,1])

# NLP processing
nlp = spacy.load('en_core_web_sm')

def tokenize_text(text):
    return [token.text.lower() for token in nlp.tokenizer(text)]

# Vocabulary class
class TextVocabulary:
    def __init__(self, min_freq=0):
        self.index_to_word = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.word_to_index = {v:k for k,v in self.index_to_word.items()}
        self.min_freq = min_freq

    def __len__(self):
        return len(self.index_to_word)

    def build_vocabulary(self, text_list):
        word_freq = Counter()
        idx = 4

        for text in text_list:
            for word in tokenize_text(text):
                word_freq[word] += 1
                if word_freq[word] == self.min_freq:
                    self.word_to_index[word] = idx
                    self.index_to_word[idx] = word
                    idx += 1

    def convert_to_index(self, text):
        tokenized_text = tokenize_text(text)
        return [self.word_to_index.get(token, self.word_to_index["<UNK>"]) for token in tokenized_text]


# Image dataset class
class ImageDataset(Dataset):
    def __init__(self, directory, caption_file, img_transform=None, freq_threshold=1):
        self.directory = directory
        self.dataframe = pd.read_csv(caption_file)
        self.img_transform = img_transform
        self.images = self.dataframe["image"]
        self.captions = self.dataframe["caption"]
        self.vocabulary = TextVocabulary(freq_threshold)
        self.vocabulary.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_caption = self.captions[idx]
        img_name = self.images[idx]
        img_path = os.path.join(self.directory, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            img = self.img_transform(img)
        caption_vector = [self.vocabulary.word_to_index["<SOS>"]]
        caption_vector += self.vocabulary.convert_to_index(img_caption)
        caption_vector += [self.vocabulary.word_to_index["<EOS>"]]
        return img, torch.tensor(caption_vector)

# Data transformation and loading
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_dataset = ImageDataset(
    directory='/fhome/gia03/Image_Captioning_CV/testing/train/',
    caption_file='/fhome/gia03/Image_Captioning_CV/testing/train/train.txt',
    img_transform=image_transforms
)

# Define a custom collate function
class CustomCollate:
    def __init__(self, padding_idx, batch_first=False):
        self.padding_idx = padding_idx
        self.batch_first = batch_first

    def __call__(self, batch_data):
        images = [item[0].unsqueeze(0) for item in batch_data]
        images = torch.cat(images, dim=0)
        captions = [item[1] for item in batch_data]
        captions = pad_sequence(captions, batch_first=self.batch_first, padding_value=self.padding_idx)
        return images, captions


# Display batch of images
def display_batch_images(batch_images, epoch='', title=None):
    batch_images = batch_images.numpy().transpose((1, 2, 0))
    plt.imshow(batch_images)
    if title:
        plt.title(f"{epoch} {title}")
    plt.pause(0.001)

# Neural network models
class ImageEncoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)  # Pre-trained on ImageNet
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        features = self.adaptive_pool(features)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        features = features.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return features

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class CaptionDecoder(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(CaptionDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, image_features, captions):
        embedded_captions = self.embedding_layer(captions)
        h, c = self.init_hidden_state(image_features)  # Initialize hidden and cell states

        seq_length = len(captions[0]) - 1  # Exclude the <EOS> token
        batch_size = captions.size(0)
        num_features = image_features.size(1)

        predictions = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        for t in range(seq_length):
            context_vector, attention_weights = self.attention_layer(image_features, h)
            lstm_input = torch.cat((embedded_captions[:, t], context_vector), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.decoder_fc(self.dropout(h))
            predictions[:, t] = output

        return predictions

# Encoder-Decoder Model
class EncoderDecoderModel(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout=0.4):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = ImageEncoder()
        self.decoder = CaptionDecoder(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    


# Function to generate captions from an image
def generate_image_captions(encoder, decoder, image, vocabulary, max_length=20):
    with torch.no_grad():
        encoder_out = encoder(image)
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(-2)

        # Flatten encoding
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        h, c = decoder.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # Start token
        start_token = vocabulary.word_to_index['<SOS>']
        captions = torch.LongTensor([[start_token]]).to(device)

        generated_captions = []
        for i in range(max_length):
            embeddings = decoder.embedding_layer(captions).squeeze(1)  # (batch_size, embed_dim)
            awe, _ = decoder.attention_layer(encoder_out, h)  # (batch_size, encoder_dim), (batch_size, num_pixels)
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (batch_size, encoder_dim)
            awe = gate * awe
            h, c = decoder.lstm_cell(torch.cat([embeddings, awe], dim=1), (h, c))  # (batch_size, decoder_dim)
            preds = decoder.decoder_fc(h)  # (batch_size, vocab_size)
            predicted_word_idx = preds.argmax(1)
            captions = predicted_word_idx.unsqueeze(1)
            generated_captions.append(predicted_word_idx.item())
            if predicted_word_idx.item() == vocabulary.word_to_index['<EOS>']:
                break

        return [vocabulary.index_to_word[idx] for idx in generated_captions]

# Visualization function
def visualize_captioned_image(image_path, generated_caption):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(' '.join(generated_caption))
    plt.show()


def visualize_attention(image_path, generated_caption, attention_weights, save_path=None):
    image = Image.open(image_path)
    image = image.resize((224, 224), Image.LANCZOS)

    fig = plt.figure(figsize=(10, 10))

    for i, (word, weight) in enumerate(zip(generated_caption, attention_weights)):
        ax = fig.add_subplot(len(generated_caption)//2, len(generated_caption)//2, i+1)
        ax.set_title(word)
        img_ax = ax.imshow(image)
        ax.imshow(weight.reshape(7, 7), cmap='gray', alpha=0.6, extent=img_ax.get_extent())

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()



# Update generate_image_captions function to return attention weights
def generate_image_captions_with_attention(encoder, decoder, image, vocabulary, max_length=20):
    with torch.no_grad():
        encoder_out = encoder(image)
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(-2)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        h, c = decoder.init_hidden_state(encoder_out)

        start_token = vocabulary.word_to_index['<SOS>']
        captions = torch.LongTensor([[start_token]]).to(device)

        generated_captions = []
        attention_weights = []
        for i in range(max_length):
            embeddings = decoder.embedding_layer(captions).squeeze(1)
            awe, alpha = decoder.attention_layer(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe
            h, c = decoder.lstm_cell(torch.cat([embeddings, awe], dim=1), (h, c))
            preds = decoder.decoder_fc(h)
            predicted_word_idx = preds.argmax(1)
            captions = predicted_word_idx.unsqueeze(1)
            generated_captions.append(predicted_word_idx.item())
            attention_weights.append(alpha.cpu().numpy())

            if predicted_word_idx.item() == vocabulary.word_to_index['<EOS>']:
                break

        return [vocabulary.index_to_word[idx] for idx in generated_captions], attention_weights





#Instantitate de vocabulary
vocab = TextVocabulary(min_freq=1)
vocab.build_vocabulary(["This is a good place to find a city"])

# DataLoader setup
BATCH_SIZE = 16
WORKERS = 1
padding_index = image_dataset.vocabulary.word_to_index["<PAD>"]



# Initialize dataloader
image_loader = DataLoader(
    dataset=image_dataset,
    batch_size=BATCH_SIZE,
    num_workers=WORKERS,
    shuffle=True,
    collate_fn=CustomCollate(padding_idx=padding_index, batch_first=True)
)




# Instantiate the model
embed_size = 300
vocab_size = len(vocab)
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EncoderDecoderModel(
    embed_size=embed_size,
    vocab_size=vocab_size,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim
).to(device)

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=image_dataset.vocabulary.word_to_index["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for idx, (imgs, captions) in enumerate(image_loader):
        imgs, captions = imgs.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if idx % 100 == 0:
            print(f"Epoch: {epoch}, Step: {idx}, Loss: {loss.item()}")



# Save model
torch.save(model.state_dict(), 'image_captioning_model.pth')


# Visualize some images

sample_image_path = '/fhome/gia03/Image_Captioning_CV/testing/test/testing/test/17273391_55cfc7d3d4.jpg'
sample_image = Image.open(sample_image_path)
sample_image = image_transforms(sample_image).unsqueeze(0).to(device)

generated_caption = generate_image_captions(model.encoder, model.decoder, sample_image, image_dataset.vocab)
visualize_captioned_image(sample_image_path, generated_caption)




sample_image_path = '/fhome/gia03/Image_Captioning_CV/testing/test/testing/test/17273391_55cfc7d3d4.jpg'
sample_image = Image.open(sample_image_path)
sample_image = image_transforms(sample_image).unsqueeze(0).to(device)

generated_caption, attention_weights = generate_image_captions_with_attention(model.encoder, model.decoder, sample_image, image_dataset.vocab)
visualize_attention(sample_image_path, generated_caption, attention_weights, save_path='attention_visualization.png')
