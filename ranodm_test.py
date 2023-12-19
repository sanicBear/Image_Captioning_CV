from transformers import BertTokenizer
import pandas as pd
'''
def adapt_bert(csv_path, caption_column='caption', tokenizer_name='bert-base-uncased'):
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Load the Flickr8k dataset
    df = pd.read_csv(csv_path)

    # Extract captions from the specified column
    captions = df[caption_column].tolist()

    # Tokenize each caption
    tokenized_captions = [tokenizer.encode(caption, add_special_tokens=True) for caption in captions]

    return tokenized_captions, tokenizer

# Example usage:
csv_location = '/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/captions.csv'
_, tokenizer = adapt_bert(csv_location)

input_text = "Example input text"
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
print("Input Text:", input_text)
print("Token IDs:", input_ids)

# Example of converting token IDs back to tokens
decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
print("Decoded Tokens:", decoded_tokens)

import torch
from torchtext.legacy.data import Field
from torchtext.vocab import FastText

def word2indx(words, max_vocab_size=1000):
    # Use FastText vectors for English
    vectors = FastText(language='en', max_vectors=max_vocab_size)
    
    # Create a Field to handle the vocabulary
    caption_field = Field(sequential=True, tokenize='spacy', lower=True, use_vocab=False)
    caption_field.vocab = vectors.vocab
    
    # Add special tokens to the vocabulary
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    for token in special_tokens:
        if token not in caption_field.vocab.stoi:
            caption_field.vocab.stoi[token] = len(caption_field.vocab.stoi)
            caption_field.vocab.itos.append(token)
    
    # Map words to indices
    word_to_index = {word: index for index, word in enumerate(caption_field.vocab.itos)}
    
    # Map words in the provided list to indices
    indices = [word_to_index.get(word, word_to_index['<unk>']) for word in words]
    
    return indices, word_to_index

def indx2word(indices, word_to_index):
    # Create an inverse mapping from indices to words
    index_to_word = {index: word for word, index in word_to_index.items()}
    
    # Map indices to words
    words = [index_to_word.get(index, '<unk>') for index in indices]
    
    return words

# Example usage
word_list = ["cat", "dog", "apple", "zebra", "common"]
word_indices, word_to_index = word2indx(word_list)

print("Word List:", word_list)
print("Word Indices:", word_indices)

# Example: Convert indices back to words
reconstructed_words = indx2word(word_indices, word_to_index)
print("Reconstructed Words:", reconstructed_words)
'''



from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torchvision import transforms
# Load the pre-trained Show and Tell model and tokenizer
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
import random
import matplotlib.pyplot as plt

# Fine-tune the model on your specific dataset
# Assume you have a custom dataset class named CustomDataset
# You need to implement this part based on your dataset loading and processing logic



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, split="train", target_size=(256, 256), tokenizer=None):
        self.folder_path = folder_path
        self.split = split
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.image_paths, self.captions = self.load_data()

        self.transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
        ])
    def load_data(self):
        image_paths = []
        captions = []
        with open(f"{self.folder_path}/{self.split}.txt", "r") as file:
            for line in file:
                line = line.strip().split(",")
                image_paths.append(line[0].strip())
                captions.append(line[1])
                print(captions)
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
captions = "/home/sisard/Documents/year_semester_1/Computer_Vision/Image_Captioning/git_repo/Image_Captioning_CV/testing/train"
dataset = CustomDataset(captions, split="train",tokenizer=processor)

#Visualize 2 random samples from the dataset
visualize_dataset(dataset, num_samples=2)
