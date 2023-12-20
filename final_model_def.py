
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T


class Vocabulary:
    def __init__(self, freq_threshold=0):
        # Initializing a dictionary to convert integer tokens to their string representation.
        # <PAD>, <SOS>, <EOS>, and <UNK> are special tokens for padding, start of sentence, end of sentence, and unknown words, respectively.
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"} 
        
        # Creating a reverse dictionary to convert strings back to their integer representation.
        # This dictionary is the inverse of self.itos.
        self.stoi = {v: k for k, v in self.itos.items()}
        
        # Setting a frequency threshold for words to be added to the vocabulary.
        # Words must appear at least this many times to be included.
        self.freq_threshold = freq_threshold
        
    # Method to get the number of items in the vocabulary.
    def __len__(self): return len(self.itos)
    
    # Static method to tokenize a given text into a list of lowercase words.
    # This method uses the 'spacy_eng.tokenizer' which is assumed to be a pre-defined tokenizer.
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    # Method to build the vocabulary from a list of sentences.
    def build_vocab(self, sentence_list):
        frequencies = Counter()  # Creating a counter to keep track of word frequencies.
        idx = 4  # Starting index for new words, as indices 0-3 are reserved for special tokens.
        
        for sentence in sentence_list:  # Iterating over each sentence.
            for word in self.tokenize(sentence):  # Tokenizing the sentence and iterating over each word.
                frequencies[word] += 1  # Incrementing the count for each word.
                
                # If a word reaches the frequency threshold, it's added to the vocabulary.
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx  # Adding the word to the string-to-integer dictionary.
                    self.itos[idx] = word  # Adding the word to the integer-to-string dictionary.
                    idx += 1  # Incrementing the index for the next word.
    
    # Method to convert a text into a list of corresponding integer tokens.
    def numericalize(self, text):
        tokenized_text = self.tokenize(text)  # Tokenizing the text.
        # Converting each token to its corresponding integer value.
        # If a word is not in the vocabulary, it's replaced by the <UNK> token.
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
    

    



class FlickrDataset(Dataset):
    """
    FlickrDataset
    This class represents a dataset for images and their corresponding captions, presumably for a task like image captioning.
    It inherits from Dataset, a common base class used in PyTorch for handling datasets.
    """ 
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=1):
        self.root_dir = root_dir  # Directory where images are stored.
        self.df = pd.read_csv(captions_file)  # Reading the captions file which links images to their captions.
        self.transform = transform  # Transformation to apply to images (e.g., resizing, normalization).
        
        # Extracting image file names and captions from the DataFrame.
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Initializing the vocabulary for the captions and building it based on the frequency threshold.
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
    
    # Method to get the number of items in the dataset.
    def __len__(self):
        return len(self.df)
    
    # Method to retrieve an image and its corresponding caption by index.
    def __getitem__(self, idx):
        caption = self.captions[idx]  # Getting the caption for the given index.
        img_name = self.imgs[idx]  # Getting the image file name for the given index.
        img_location = os.path.join(self.root_dir, img_name)  # Creating the full path to the image.
        img = Image.open(img_location).convert("RGB")  # Opening and converting the image to RGB.
        
        # Applying the specified transformation to the image if any.
        if self.transform is not None:
            img = self.transform(img)
        
        # Preparing the caption for processing:
        # Start with <SOS> token, add the numericalized caption, and end with <EOS> token.
        caption_vec = [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        # Returning the processed image and the numericalized caption as a tensor.
        return img, torch.tensor(caption_vec)


class CapsCollate:
    """
    CapsCollate
    A custom collate function for a DataLoader that processes batches of image-caption pairs.
    It pads the captions to ensure they are all the same length within a batch.
    """
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx  # Index used for padding the captions.
        self.batch_first = batch_first  # Determines the dimension order in the output tensors.
    
    def __call__(self, batch):
        # Extracting and concatenating the images from the batch.
        # Each image tensor is unsqueezed to add a batch dimension, then all are concatenated along this dimension.
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        # Extracting the caption vectors from the batch.
        targets = [item[1] for item in batch]
        # Padding the caption sequences to make them of equal length within this batch.
        # The padding value and whether the batch dimension is first are specified by the instance variables.
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        
        # Returning the batch of images and the corresponding padded captions.
        return imgs, targets



class EncoderCNN(nn.Module):
    """
    EncoderCNN
    A Convolutional Neural Network (CNN) encoder, typically used in image captioning models.
    This class uses a pre-trained ResNet-50 model as the backbone, with some modifications.
    """
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # Load a pre-trained ResNet-50 model.
        resnet = models.resnet50(pretrained=True)
        # Freezing the parameters of the ResNet-50 model to prevent them from being updated during training.
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Removing the last two layers of the ResNet-50 model (usually the global average pooling layer and the fully connected layer).
        # This modification is to use the model as a feature extractor, rather than for classification.
        modules = list(resnet.children())[:-2]
        # Re-structuring the ResNet-50 model with the modified layers.
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        # Forward pass for the images through the modified ResNet-50 model.
        features = self.resnet(images)  # (batch_size, 2048, 7, 7) - The shape of the output feature map.

        # Permuting the dimensions of the output to rearrange the feature maps.
        # This is usually done to align with the expected input format for subsequent parts of an image captioning model.
        features = features.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 2048)

        # Reshaping the features to flatten the spatial dimensions while keeping the feature dimension.
        # This step prepares the data for processing by the subsequent components of the model (like an RNN or a transformer).
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size, 49, 2048)

        return features





class Attention(nn.Module):
    """
    Attention
    This class implements the Bahdanau attention mechanism, which is a type of attention used in neural network models,
    particularly for sequence-to-sequence tasks like machine translation and image captioning.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        # Initializing the attention dimension size.
        self.attention_dim = attention_dim
        
        # Linear transformation for the hidden state from the decoder.
        self.W = nn.Linear(decoder_dim, attention_dim)
        # Linear transformation for the features (encoder outputs).
        self.U = nn.Linear(encoder_dim, attention_dim)
        
        # Linear layer to compute the attention scores.
        self.A = nn.Linear(attention_dim, 1)
        
    def forward(self, features, hidden_state):
        # Transforming the encoder's features to the attention dimension.
        u_hs = self.U(features)  # (batch_size, num_layers, attention_dim)
        
        # Transforming the decoder's hidden state to the attention dimension.
        w_ah = self.W(hidden_state)  # (batch_size, attention_dim)
        
        # Combining the transformed features and hidden state and applying tanh activation.
        # This combination is the core of the attention mechanism.
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))  # (batch_size, num_layers, attention_dim)
        
        # Computing the attention scores from the combined states.
        attention_scores = self.A(combined_states)  # (batch_size, num_layers, 1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size, num_layers)
        
        # Applying softmax to get the attention weights (probabilities).
        alpha = F.softmax(attention_scores, dim=1)  # (batch_size, num_layers)
        
        # Computing the weighted sum of the features, which is the context vector.
        # This vector represents the attended features.
        attention_weights = features * alpha.unsqueeze(2)  # (batch_size, num_layers, features_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size, num_layers)
        
        # Returning both the attention weights and the context vector.
        return alpha, attention_weights

# Attention Decoder
class DecoderRNN(nn.Module):
    """
    DecoderRNN
    This class represents the decoder part of an image captioning model using an LSTM with attention.
    """
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        
        # Storing model parameters.
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        # Embedding layer to convert word indices into dense vectors of a fixed size.
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Attention model instance.
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        # Linear layers to initialize the hidden and cell state of the LSTM using encoder's output.
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        # LSTMCell to process sequences step by step with attention mechanism.
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        # Linear layer to create a learnable scalar to weigh the context vector.
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        # Fully connected layer to predict the next word in the sequence.
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        # Dropout layer for regularization.
        self.drop = nn.Dropout(drop_prob)
    
    def forward(self, features, captions):
        # Embedding the input captions.
        embeds = self.embedding(captions)
        
        # Initialize the LSTM state.
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        # Preparing to collect predictions and attention weights.
        seq_length = len(captions[0]) - 1  # Excluding the last one for prediction.
        batch_size = captions.size(0)
        num_features = features.size(1)
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)
        
        # Iterating over each word in the sequence.
        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Predicting the next word in the sequence.
            output = self.fcn(self.drop(h))
            preds[:, s] = output
            alphas[:, s] = alpha
        
        return preds, alphas
    
    def generate_caption(self, features, max_len=20, vocab=None):
        # Generation of caption in inference mode.
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        # Starting with <SOS> token.
        word = torch.tensor(vocab.stoi['<SOS>']).view(1, -1).to(device)
        embeds = self.embedding(word)
        captions = []
        alphas = []
        
        # Generating caption up to a maximum length.
        for i in range(max_len):
            alpha, context = self.attention(features, h)
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)
            
            # Choosing the word with the highest probability.
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())
            
            # Stop if <EOS> is detected.
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            # Preparing the next input word.
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        
        # Converting the predicted indices to words.
        return [vocab.itos[idx] for idx in captions], alphas
    
    def init_hidden_state(self, encoder_out):
        # Initialize LSTM hidden and cell states using the encoder's output.
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c



class EncoderDecoder(nn.Module):
    """
    EncoderDecoder
    This class encapsulates an image captioning model which consists of two main components: 
    an encoder that processes images and a decoder that generates captions.
    """
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        # Initializing the encoder part of the model.
        # This part is responsible for processing images and extracting feature representations.
        self.encoder = EncoderCNN()

        # Initializing the decoder part of the model.
        # The decoder uses the features provided by the encoder to generate captions.
        # It incorporates an attention mechanism and an RNN (LSTM) for this purpose.
        self.decoder = DecoderRNN(
            embed_size=embed_size,      # Size of the word embeddings.
            vocab_size=vocab_size,      # The size of the vocabulary.
            attention_dim=attention_dim,# Dimensionality of the attention space.
            encoder_dim=encoder_dim,    # Dimensionality of the encoder's output features.
            decoder_dim=decoder_dim,    # Dimensionality of the decoder's LSTM.
            drop_prob=drop_prob         # Dropout probability for regularization.
        )
        
    def forward(self, images, captions):
        # Forward pass for the model.
        # First, images are passed through the encoder to get feature representations.
        features = self.encoder(images)

        # These features along with the captions are then passed to the decoder.
        # The decoder uses these features to generate the captions.
        outputs = self.decoder(features, captions)
        
        # Returning the outputs from the decoder.
        # These outputs typically include predicted word probabilities for each position in the caption sequence.
        return outputs




def show_image(img, title=None, epoch_num=' '):
    """
    This function displays an image and optionally adds a title. It also saves the image to a file.
    'img' is expected to be a PyTorch tensor representing an image.
    """
    
    # Unnormalize the image. The values 0.229, 0.224, 0.225, 0.485, 0.456, 0.406 are standard normalization means and stds for pre-trained models.
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    # Convert the tensor image to a NumPy array and change the order of dimensions for plotting.
    img = img.numpy().transpose((1, 2, 0))
    
    # Display the image using matplotlib.
    plt.imshow(img)
    output = '/fhome/gia03/Image_Captioning_CV/testing/plots/'+str(epoch_num)+'_epoch_num'+'.png'
    
    # Add a title if provided.
    if title is not None:
        plt.title(str(epoch_num)+' '+title)
    
    # Save the image to a file and pause briefly to update plots.
    plt.savefig(output)
    plt.pause(0.001)  # Pause to update the plot.


def plot_and_save_loss(training_loss, file_name='training_loss.png'):
    """
    This function plots the training loss over epochs and saves the plot to a file.
    'training_loss' should be a list or array-like object containing loss values.
    """
    # Create a matplotlib figure and axis.
    fig, ax = plt.subplots()

    # Plot the training loss on the axis.
    ax.plot(training_loss, label='Training Loss')

    # Set title and axis labels.
    ax.set_title('Training Loss Over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    # Add a legend to the plot.
    ax.legend()

    # Save the plot to a file.
    plt.savefig(file_name)


# Helper function to save the model state.
def save_model(model, num_epochs):
    """
    This function saves the state of a PyTorch model to a file.
    'model' is the model to be saved, and 'num_epochs' indicates the number of training epochs completed.
    """
    model_state = {
        'num_epochs': num_epochs,
        'embed_size': embed_size,
        'vocab_size': len(dataset.vocab),
        'attention_dim': attention_dim,
        'encoder_dim': encoder_dim,
        'decoder_dim': decoder_dim,
        'state_dict': model.state_dict()
    }
    # Save the model state to a file.
    torch.save(model_state, 'attention_model_state.pth')


def get_caps_from(features_tensors):
    """
    This function generates captions for a given set of image features.
    'features_tensors' is a tensor of image features.
    It returns the generated captions and the attention weights.
    """
    # Switch model to evaluation mode.
    model.eval()
    with torch.no_grad():
        # Get image features using the encoder part of the model.
        features = model.encoder(features_tensors.to(device))
        # Generate captions using the decoder part of the model.
        caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
        # Convert the list of word indices to a sentence.
        caption = ' '.join(caps)
        # Display the image with the generated caption.
        show_image(features_tensors[0], title=caption)
    
    # Return the captions and the attention weights.
    return caps, alphas

def plot_attention(img, result, attention_plot, name):
    """
    This function visualizes the attention weights for each word in the generated caption over the image.
    'img' is the input image tensor, 'result' is a list of words in the generated caption,
    'attention_plot' contains the attention weights, and 'name' is the filename for saving the plot.
    """
    # Unnormalize the image using standard means and stds.
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406
    
    # Convert the image tensor to a numpy array and transpose the dimensions for display.
    img = img.numpy().transpose((1, 2, 0))
    temp_image = img  # Storing the unnormalized image for plotting.

    # Creating a figure for plotting.
    fig = plt.figure(figsize=(15, 15))

    # Length of the caption result.
    len_result = len(result)
    
    # Looping over each word in the caption.
    for l in range(len_result):
        # Reshaping the attention weights for the l-th word in the caption.
        temp_att = attention_plot[l].reshape(7, 7)
        
        # Adding a subplot for each word in the caption.
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])  # Setting the title of the subplot to the word.
        
        # Displaying the image.
        img = ax.imshow(temp_image)
        # Overlaying the attention weights on the image with some transparency.
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())
    
    # Adjusting the layout to prevent overlap of subplots.
    plt.tight_layout()
    
    # Saving the figure with the given filename.
    plt.savefig(name)










caption_file = '/fhome/gia03/Image_Captioning_CV/testing/train/train.txt'
df = pd.read_csv(caption_file)
print("There are {} image to captions".format(len(df)))
print(df.head(7))

spacy_eng = spacy.load('en_core_web_sm')


#defing the transform to be applied
transforms = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

#testing the dataset class
dataset =  FlickrDataset(
    root_dir ='/fhome/gia03/Image_Captioning_CV/testing/train/',
    captions_file = '/fhome/gia03/Image_Captioning_CV/testing/train/train.txt',
    transform=transforms, freq_threshold=1
)


img, caps = dataset[0]


#setting the constants
BATCH_SIZE = 16
NUM_WORKER = 1

#token to represent the padding
pad_idx = dataset.vocab.stoi["<PAD>"]

data_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
)


#generating the iterator from the dataloader
dataiter = iter(data_loader)

#getting the next batch
batch = next(dataiter)

#unpacking the batch
images, captions = batch

#showing info of image in single batch
for i in range(BATCH_SIZE):
    img,cap = images[i],captions[i]
    caption_label = [dataset.vocab.itos[token] for token in cap.tolist()]
    eos_index = caption_label.index('<EOS>')
    caption_label = caption_label[1:eos_index]
    caption_label = ' '.join(caption_label)                      
    show_image(img,title = caption_label)
    plt.show()


#location of the training data 
data_location =  '/fhome/gia03/Image_Captioning_CV/testing/train/'

BATCH_SIZE = 32
# BATCH_SIZE = 6
NUM_WORKER = 4

#defining the transform to be applied
transforms = T.Compose([
    T.Resize(226),                     
    T.RandomCrop(224),                 
    T.ToTensor(),                               
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])


#testing the dataset class
dataset =  FlickrDataset(
    root_dir = '/fhome/gia03/Image_Captioning_CV/testing/train/',
    captions_file = '/fhome/gia03/Image_Captioning_CV/testing/train/train.txt',
    transform=transforms
)



# Initialazing the dataloader
data_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKER,
    shuffle=True,
    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
)
# Vocabulary size determination.
vocab_size = len(dataset.vocab)  # Getting the size of the vocabulary from the dataset.
print(vocab_size)  # Printing the vocabulary size.

# Setting up the device for training (using CUDA if available).
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model hyperparameters.
embed_size = 300       # The size of the embedding layer.
vocab_size = len(dataset.vocab)  # Reconfirming the vocabulary size.
attention_dim = 256    # The size of the attention layer.
encoder_dim = 2048     # The dimensionality of the encoder's output.
decoder_dim = 512      # The dimensionality of the decoder.
learning_rate = 3e-4   # Learning rate for the optimizer.

# Initializing the model.
model = EncoderDecoder(
    embed_size=embed_size,
    vocab_size=vocab_size,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim
).to(device)  # Moving the model to the appropriate device (GPU or CPU).

# Loss function and optimizer.
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])  # Cross-entropy loss, ignoring the padding index.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer with the defined learning rate.

# Training loop setup.
num_epochs = 15  # Number of epochs to train for.
print_every = 100  # Frequency of printing the training loss.
train_loss_list = []  # List to store the loss values for plotting later.

# Training loop.
for epoch in tqdm(range(1, num_epochs + 1)):  # Looping over each epoch.
    for idx, (image, captions) in tqdm(enumerate(iter(data_loader))):  # Looping over each batch in the dataset.
        image, captions = image.to(device), captions.to(device)  # Moving the batch to the appropriate device.

        optimizer.zero_grad()  # Zeroing the gradients to prevent accumulation.

        # Forward pass.
        outputs, attentions = model(image, captions)  # Getting the model's predictions and attention weights.

        # Loss computation.
        targets = captions[:, 1:]  # Target captions (excluding the <SOS> token).
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))  # Reshaping and computing loss.

        # Backward pass and optimization.
        loss.backward()  # Backpropagation.
        optimizer.step()  # Updating the model parameters.

        # Printing and visualization.
        if (idx + 1) % print_every == 0:
            print("Epoch: {} loss: {:.5f}".format(epoch, loss.item()))  # Printing the loss.
            
            # Generating a caption to visualize the model's performance.
            model.eval()  # Switching to evaluation mode.
            with torch.no_grad():
                dataiter = iter(data_loader)
                img, _ = next(dataiter)
                features = model.encoder(img[0:1].to(device))
                caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
                caption = ' '.join(caps)
                show_image(img[0], title=caption, epoch_num=epoch)  # Showing the image with the generated caption.
                train_loss_list.append(loss.item)
                
            model.train()  # Switching back to training mode.
        
    # Saving the model after each epoch.
    save_model(model, epoch)

# Plotting the training loss after the training loop.
plot_and_save_loss(train_loss_list, '/fhome/gia03/Image_Captioning_CV/testing/plots/my_training_loss_plot.png')



dataiter = iter(data_loader)
images,_ = next(dataiter)

img = images[0].detach().clone()
img1 = images[0].detach().clone()
caps,alphas = get_caps_from(img.unsqueeze(0))
name =  '/fhome/gia03/Image_Captioning_CV/testing/plots/atention_1.png'
plot_attention(img1, caps, alphas,name)


dataiter = iter(data_loader)
images,_ = next(dataiter)

img = images[0].detach().clone()
img1 = images[0].detach().clone()
caps,alphas = get_caps_from(img.unsqueeze(0))
name =  '/fhome/gia03/Image_Captioning_CV/testing/plots/atention_2.png'
plot_attention(img1, caps, alphas,name)


dataiter = iter(data_loader)
images,_ = next(dataiter)

img = images[0].detach().clone()
img1 = images[0].detach().clone()
caps,alphas = get_caps_from(img.unsqueeze(0))
name =  '/fhome/gia03/Image_Captioning_CV/testing/plots/atention_3.png'
plot_attention(img1, caps, alphas,name)

