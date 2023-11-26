
import numpy as np
import random
from transformers import ResNetModel
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torch
import pandas as pd
import evaluate
from torch.utils.data import DataLoader


DEVICE = 'cpu'

base_path = '/home/sisard/Documents/universitat/year_3/semester_1/Computer_Vision/Image_Captioning/'
img_path = f'{base_path}Images/'
cap_path = f'{base_path}captions.txt'

data = pd.read_csv(cap_path)
partitions = np.load('flickr8k_partitions.npy', allow_pickle=True).item()


chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 201


class Data(Dataset):
    def __init__(self, data, partition):
        self.data = data
        self.partition = partition
        self.num_captions = 5
        self.max_len = TEXT_MAX_LEN
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)

    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        real_idx = self.num_captions*self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        ## image processing
        img_name = item.image.reset_index(drop=True)[0]
        img = Image.open(f'{img_path}{img_name}').convert('RGB')
        img = self.img_proc(img)
    
        ## caption processing
        caption = item.caption.reset_index(drop=True)[random.choice(list(range(self.num_captions)))]
        cap_list = list(caption)
        final_list = [chars[0]]
        final_list.extend(cap_list)
        final_list.extend([chars[1]])
        gap = self.max_len - len(final_list)
        final_list.extend([chars[2]]*gap)
        cap_idx = [char2idx[i] for i in final_list]
        return img, cap_idx


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512
        start = torch.tensor(char2idx['<SOS>']).to(DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        hidden = feat
        for t in range(TEXT_MAX_LEN-1): # rm <SOS>
            out, hidden = self.gru(inp, hidden)
            inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 512
    
        res = inp.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq, 80
        res = res.permute(0, 2, 1) # batch, 80, seq
        return res


'''A simple example to calculate loss of a single batch (size 2)'''
dataset = Data(data, partitions['train'])
img1, caption1 = next(iter(dataset))
img2, caption2 = next(iter(dataset))
caption1 = torch.tensor(caption1)
caption2 = torch.tensor(caption2)
img = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)))
caption = torch.cat((caption1.unsqueeze(0), caption2.unsqueeze(0)))
img, caption = img.to(DEVICE), caption.to(DEVICE)
model = Model().to(DEVICE)
pred = model(img)
crit = nn.CrossEntropyLoss()
loss = crit(pred, caption)
print(loss)



'''metrics'''
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

reference = [['A child in a pink dress is climbing up a set of stairs in an entry way .', 'A girl going into a wooden building .']]
prediction = ['A girl goes into a wooden building .']

res_b = bleu.compute(predictions=prediction, references=reference)
res_r = rouge.compute(predictions=prediction, references=reference)
res_m = meteor.compute(predictions=prediction, references=reference)

res_b, res_r, res_m


ref = [['A child is running in the campus']]
pred1 = ['A child is running']

res_b = bleu.compute(predictions=pred1, references=ref)
res_r = rouge.compute(predictions=pred1, references=ref)
res_m = meteor.compute(predictions=pred1, references=ref)

res_b, res_r, res_m


ref = [['A child is running in the campus']]
pred1 = ['A child is']

res_b = bleu.compute(predictions=pred1, references=ref)
res_r = rouge.compute(predictions=pred1, references=ref)
res_m = meteor.compute(predictions=pred1, references=ref)

res_b, res_r, res_m


ref = [['A child is running in the campus']]
pred1 = ['A child campus']

res_b = bleu.compute(predictions=pred1, references=ref)
res_r = rouge.compute(predictions=pred1, references=ref)
res_m = meteor.compute(predictions=pred1, references=ref)
res_m_sin = meteor.compute(predictions=pred1, references=ref, gamma=0) # no penalty by setting gamma to 0

res_b, res_r, res_m, res_m_sin


# Final metric we use for challenge 3: BLEU1, BLEU2, ROUGE-L, METEOR


ref = [['A child is running in the campus']]
pred1 = ['A child campus']

bleu1 = bleu.compute(predictions=pred1, references=ref, max_order=1)
bleu2 = bleu.compute(predictions=pred1, references=ref, max_order=2)
res_r = rouge.compute(predictions=pred1, references=ref)
res_m = meteor.compute(predictions=pred1, references=ref)

f"BLEU-1:{bleu1['bleu']*100:.1f}%, BLEU2:{bleu2['bleu']*100:.1f}%, ROUGE-L:{res_r['rougeL']*100:.1f}%, METEOR:{res_m['meteor']*100:.1f}%"


# Now it is your turn! Try to finish the code below to run the train function


def train(EPOCHS):
    data_train = Data(data, partitions['train'])
    data_valid = Data(data, partitions['valid'])
    data_test = Data(data, partitions['test'])
    dataloader_train = DataLoader(data_train, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(data_valid, batch_size=32, shuffle=True)
    dataloader_valid = DataLoader(data_test, batch_size=32, shuffle=True)
    
    model = Model().to(DEVICE)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    metric = rouge#
    for epoch in range(EPOCHS):
        loss, res = train_one_epoch(model, optimizer, crit, metric, dataloader_train)
        print(f'train loss: {loss:.2f}, metric: {res:.2f}, epoch: {epoch}')
        loss_v, res_v = eval_epoch(model, crit, metric, dataloader_valid)
        print(f'valid loss: {loss:.2f}, metric: {res:.2f}')
    loss_t, res_t = eval_epoch(model, crit, metric, dataloader_test)
    print(f'test loss: {loss:.2f}, metric: {res:.2f}')
    
def train_one_epoch(model, optimizer, crit, metric, dataloader):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = len(dataloader)

    for batch_idx, (img, caption) in enumerate(dataloader):
        print('························')
        print(caption)
        img, caption = img.to(DEVICE), caption.to(DEVICE)
        print('························')
        print(caption)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(img)

        # Compute the loss
        loss = crit(output, caption)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate the total loss
        total_loss += loss.item()

        # Compute the metric (e.g., Rouge)
        # You might need to convert the model predictions and captions to strings
        # and pass them to the Rouge metric, adjust accordingly
        # For example:
        #predictions = decode_predictions(output)  # Implement a decode_predictions function
        #references = decode_captions(caption)  # Implement a decode_captions function
        batch_metric = metric.compute(predictions=output, references=caption)
        total_metric += batch_metric

    # Calculate average loss and metric over all batches
    average_loss = total_loss / num_batches
    average_metric = total_metric / num_batches

    return average_loss, average_metric


def eval_epoch(model, crit, metric, dataloader):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (img, caption) in enumerate(dataloader):
            img, caption = img.to(DEVICE), caption.to(DEVICE)

            # Forward pass
            output = model(img)

            # Compute the loss
            loss = crit(output, caption)

            # Accumulate the total loss
            total_loss += loss.item()

            # Compute the metric (e.g., Rouge)
            #predictions = decode_predictions(output)  # Implement a decode_predictions function
            #references = decode_captions(caption)  # Implement a decode_captions function
            batch_metric = metric.compute(predictions=output, references=caption)
            total_metric += batch_metric

    # Calculate average loss and metric over all batches
    average_loss = total_loss / num_batches
    average_metric = total_metric / num_batches

    return average_loss, average_metric

train(EPOCHS=5)


