from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from torch import nn
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from transformers import ResNetModel
from torchvision.transforms import v2
import torch
import pandas as pd
import evaluate
from torch.utils.data import DataLoader
import zipfile
from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
import random
import torch
from torchtext.vocab import FastText

from metrics2 import Model, Data

DEVICE = 'cuda'


base_path = '/fhome/gia03/'
zip_file_path = f'{base_path}Dataset.zip'
extracted_folder_path = f'{base_path}extracted_data/'

# Extraer el contenido del archivo zip
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

# Ahora, puedes cargar tus datos desde la carpeta extraída
img_path = f'{extracted_folder_path}Images/'
cap_path = f'{extracted_folder_path}captions.txt'

data = pd.read_csv(cap_path)
partitions = np.load("/fhome/gia03/flickr8k_partitions.npy", allow_pickle=True).item()

chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']



from gensim.models import FastText
from torchtext.data import Field

def char2idx(words, max_vocab_size=1000):
    # Use FastText vectors for English
    vectors = FastText(language='en', max_vectors=max_vocab_size)

    # Create a Field to handle the vocabulary
    caption_field = Field(sequential=True, tokenize='spacy', lower=True, use_vocab=False)
    caption_field.vocab = vectors.wv

    # Add the special tokens to the vocabulary
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    for token in special_tokens:
        if token not in caption_field.vocab.stoi:
            caption_field.vocab.stoi[token] = len(caption_field.vocab.stoi)
            caption_field.vocab.itos.append(token)

    # Map the words to the indices
    word_to_index = {word: index for index, word in enumerate(caption_field.vocab.itos)}

    # Map the words in the provided list to the indices
    indices = [word_to_index.get(word, word_to_index['<UNK>']) for word in words]

    return indices, word_to_index

def idx2char(indices, word_to_index):
    # Create the inverse mapping, from indices to words
    index_to_word = {index: word for word, index in word_to_index.items()}

    # Map the indices to words
    words = [index_to_word.get(index, '<UNK>') for index in indices]

    return words

    
# Obtener todas las captions de entrenamiento
train_captions = data['caption'].tolist()

# Dividir las captions en palabras
words = [caption.split() for caption in train_captions]

# Obtener el conjunto único de palabras
unique_words = set([word for caption_words in words for word in caption_words])
vocabulary = list(unique_words)

# Actualizar NUM_CHAR y los diccionarios idx2char, char2idx
NUM_CHAR = 1004
#idx2char = {k: v for k, v in enumerate(vocabulary)}
#char2idx = {v: k for k, v in enumerate(vocabulary)}


    
def id2str(seq):
    return ("".join([idx2char[item] for item in seq]))
 

def text_to_indices(text, char2idx):
    return [char2idx[word] for word in text.split()]



def test_model(model, test_dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for img_batch, caption_batch in test_dataloader:
            img_batch, caption_batch = img_batch.to(DEVICE), caption_batch.to(DEVICE)

            pred = model(img_batch)

            # Antes de la línea con la pérdida, agrega estas líneas para verificar las dimensiones
            print("Dimensiones de pred:", pred.shape)
            print("Dimensiones de caption_batch:", caption_batch.shape)
              
            
            loss = criterion(pred.reshape(-1, NUM_CHAR, caption_batch.reshape(-1).contiguous()))


            # Antes de la línea con la pérdida, agrega estas líneas para verificar las dimensiones
            print("Dimensiones de pred:", pred.shape)
            print("Dimensiones de caption_batch:", caption_batch.shape)
              

            total_loss += loss.item()

            # Postprocesamiento de predicciones
            pred_text = [id2str(seq) for seq in pred.argmax(dim=1).tolist()]
            pred_text = [sentence.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip() for sentence in pred_text]

            # Postprocesamiento de frases reales
            real_text = [id2str(seq) for seq in caption_batch.tolist()]
            real_text = [sentence.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip() for sentence in real_text]
            
                    
           
    avg_loss = total_loss / len(test_dataloader)
    print(f'Test loss: {avg_loss:.2f}')


def generate_predictions(model, image_path):
    model.eval()

    # Cargar la imagen
    img = Image.open(image_path).convert('RGB')
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img).unsqueeze(0).to(DEVICE)

    # Obtener la predicción del modelo
    with torch.no_grad():
        pred = model(img)

        # Postprocesamiento de predicciones
        pred_words = [idx2char[idx.item()] for idx in pred.argmax(dim=1)]
        pred_text = ' '.join(pred_words)

    return pred_text


# Cargargamos el modelo
model = Model().to(DEVICE)
model.load_state_dict(torch.load('/fhome/gia03/Gloria/model_epoch35.pth')) 
model.eval()  
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)  
criterion = nn.CrossEntropyLoss()


test_dataloader = DataLoader(Data(data, partitions['test']), batch_size=16, shuffle=True)

# Ejecutar la prueba del modelo
test_model(model, test_dataloader)

# Generar predicciones para una imagen específica 
image_path_to_test = "/export/fhome/gia03/Images/99171998_7cc800ceef.jpg"
predictions = generate_predictions(model, image_path_to_test)
print("Predicciones para la imagen:")
print(predictions)
