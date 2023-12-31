# -*- coding: utf-8 -*-
"""Metrics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OkdyuBmF29wBN9saZTyhPJGL5Yt7n-FR
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import zipfile
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

NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 201




def text_to_indices(text, char2idx):
    return [char2idx[char] for char in text]


class Data(Dataset):
    def __init__(self, data, partition):
        self.data = data
        self.partition = partition
        self.num_captions = 5
        self.max_len = TEXT_MAX_LEN
        self.img_proc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #NORMALIZACION 
        ])

    def __len__(self):
        return len(self.partition)
        
    def __getitem__(self, idx):
        real_idx = self.num_captions * self.partition[idx]
        item = self.data.iloc[real_idx: real_idx + self.num_captions]
        ## image processing
        img_name = item.image.reset_index(drop=True)[0]
        img = Image.open(f'{img_path}{img_name}').convert('RGB')
        img = self.img_proc(img)

        ## caption processing
        caption = item.caption.reset_index(drop=True)[random.choice(list(range(self.num_captions)))]
        cap_str = caption  # Obtén la cadena completa de la leyenda
        cap_list = list(cap_str)
        final_list = [chars[0]]
        final_list.extend(cap_list)
        final_list.extend([chars[2]])
        gap = self.max_len - len(final_list)
        final_list.extend([chars[2]] * gap)
        cap_idx = text_to_indices(final_list, char2idx)
        
        return img, torch.tensor(cap_idx, dtype=torch.long)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(num_classes, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc = nn.Linear(d_model, num_classes)
        
       
    def forward(self, img):
 
        # Nos aseguramos que sea tipo torch.LongTensor para que no de error.
        img = img.type(torch.LongTensor).to(DEVICE)
    
        # Capa de embedding: convierte las etiquetas en vectores 
        x = self.embedding(img)
    
        # Cambiar la forma del tensor para tener (sequence_length, batch_size, features)
        x = x.permute(1, 0, 2)
    
        # Asegurarse de que el tensor sea 3D
        x = x.unsqueeze(0)
 
        # Capa Transformer
        x = self.transformer(x, x)
    
        # Asegurarse de que el tensor sea 3D 
        x = x.unsqueeze(0)
    
        # Transponer de nuevo a la forma original
        x = x.squeeze(0).permute(1, 0, 2)
    
        # Fully-connected para la salida final
        x = self.fc(x)
    
        return x


 
    def calculate_regularization_loss(self):
        reg_loss = 0.0
        for param in self.parameters():
            reg_loss += self.l2_regularization(param)
        return reg_loss * self.weight_decay



    
def id2str(seq):
    return ("".join([idx2char[item] for item in seq]))
 

def train_one_epoch(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0

    for img_batch, caption_batch in tqdm(dataloader):
        img_batch, caption_batch = img_batch.to(DEVICE), caption_batch.to(DEVICE)
        optimizer.zero_grad()

        # Solo pasa la imagen al modelo Transformer
        pred = model(img_batch)

        # Realiza el cálculo de la pérdida como antes
        loss = criterion(pred.reshape(-1, NUM_CHAR), caption_batch.reshape(-1).contiguous())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        pred = pred.argmax(dim=1).tolist()
        pred = [id2str(seq) for seq in pred]
        
        for element in pred:
            if (element == chars[0]) or (element == chars[1]) or (element == chars[2]):
                pred.remove(element)
          
        caption_batch = caption_batch.tolist()
        caption_batch = [id2str(seq) for seq in caption_batch]
        
        for element in caption_batch:
            if (element == chars[0]) or (element == chars[1]) or (element == chars[2]):
                caption_batch.remove(element)

    avg_loss = total_loss / len(dataloader)
    return avg_loss


   
def eval_epoch(model, criterion, dataloader,optimizer):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for img_batch, caption_batch in dataloader:
            img_batch, caption_batch = img_batch.to(DEVICE), caption_batch.to(DEVICE)
        
            pred = model(img_batch)

            #print(f"Shape of input batch: {img_batch.shape}")
            #print(f"Shape of caption batch: {caption_batch.shape}")
            #print(f"Shape of model output: {pred.shape}")

            #pred = pred.permute(0, 2, 1).contiguous().view(-1, NUM_CHAR)
            
            loss = criterion(pred.reshape(-1, NUM_CHAR), caption_batch.reshape(-1).contiguous())
            #loss.backward()
            total_loss += loss.item()

    
            pred = pred.argmax(dim=1).tolist()
            pred = [id2str(seq) for seq in pred]
            
            for element in pred:
              if (element == chars[0]) or (element == chars[1]) or (element == chars[2]):
                pred.remove(element)
              
            caption_batch = caption_batch.tolist()
            caption_batch = [id2str(seq) for seq in caption_batch]
            
            for element in caption_batch:
              if (element == chars[0]) or (element == chars[1]) or (element == chars[2]):
                caption_batch.remove(element)
           

    avg_loss = total_loss / len(dataloader)
    return avg_loss
    
    

def train(EPOCHS, model, optimizer, criterion, train_dataloader, valid_dataloader):
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, optimizer, criterion, train_dataloader)
        print(f'train loss: {train_loss:.2f}, epoch: {epoch}')
       

        valid_loss = eval_epoch(model, criterion, valid_dataloader,optimizer)
        print(f'valid loss: {valid_loss:.2f}')
       
# Define the parameters
num_layers = 2
d_model = 512
nhead = 8
dim_feedforward = 2048
num_classes = NUM_CHAR 


# Create an instance of the TransformerModel
transformer_model = TransformerModel(
    num_layers=num_layers,
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    num_classes=num_classes
).to(DEVICE)

# Print the model architecture
print(transformer_model)

optimizer = torch.optim.SGD(transformer_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

train_dataloader = DataLoader(Data(data, partitions['train']), batch_size=16, shuffle=True)
valid_dataloader = DataLoader(Data(data, partitions['valid']), batch_size=16, shuffle=True)

train(EPOCHS=5, model=transformer_model, optimizer=optimizer, criterion=criterion, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)

#torch.save(transformer_model.state_dict(), '/fhome/gia03/Gloria/transformer_model.pth')
