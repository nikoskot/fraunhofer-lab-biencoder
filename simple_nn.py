import numpy as np
import pickle
import torch
from torch import nn
from torch.utils import data
import tqdm
import random
import matplotlib.pyplot as plt

class Word_Sense_Embeddings_Dataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, word_emb_dict, sense_emb_dict, device='cuda'):
        """
        """
        assert word_emb_dict.keys() == sense_emb_dict.keys()

        self.word_emb_dict = word_emb_dict
        self.sense_emb_dict = sense_emb_dict
        self.device = device

    def __len__(self):
        return len(self.word_emb_dict)

    def __getitem__(self, idx):
        
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        instance = list(self.word_emb_dict.keys())[idx]

        rad = torch.rand(1, device=self.device)

        return instance, torch.tensor(self.word_emb_dict[instance], device=self.device), torch.tensor(self.sense_emb_dict[instance], device=self.device), rad
    
class MLP(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        """
            Convolutional Block with a residual connection
            
            Parameters
            ----------
            in_channels  : int
                number of input channels
            out_channels : int
                number of output channels
        """

        #--- YOUR CODE HERE ---#
        # TODO
        self.in_dims = in_dims
        self.out_dims = out_dims

        # Convolution Sub-block 1
        self.linear1 = nn.Linear(in_features=in_dims, out_features=2* in_dims)
        self.linear2 = nn.Linear(in_features=2* in_dims, out_features=out_dims)


    def forward(self, x: torch.Tensor):
        """
            x : tensor [N, C, H, W]
        """

        # --- YOUR CODE HERE ---#
        # TODO

        # Pass the input through the two blocks
        x = self.linear1(x)
        x = self.linear2(x)
        
        return x
    
    # def forward(self, x: torch.Tensor):
    #     """
    #         x : tensor [N, C, H, W]
    #     """

    #     # --- YOUR CODE HERE ---#
    #     # TODO

    #     # Pass the input through the two blocks
    #     norm = torch.linalg.norm(x)

    #     x = self.linear(x)

    #     x = x / torch.linalg.norm(x) * norm
        
    #     return x

def my_loss(pred, target, rad):
    
    dist = (pred - target).pow(2).sum(1).sqrt()

    return torch.clamp(dist - torch.squeeze(rad, 1), min=0)

def my_loss2(pred, target, rad):

    return torch.acos(torch.dot(pred, target) / (torch.linalg.norm(pred) * torch.linalg.norm(pred)))

def train_loop(dataloader, model, optimizer):

    size = len(dataloader.dataset)

    epoch_loss = 0

    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    for batch, (instance, word_emb, sense_emb, rad) in enumerate(tqdm.tqdm(dataloader)):

        # Compute prediction and loss
        pred = model(word_emb)

        loss = my_loss(pred, sense_emb, rad)

        loss = loss.sum()

        epoch_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return epoch_loss / size

def test_loop(dataloader, model):

    size = len(dataloader.dataset)

    test_loss = 0

    num_correct_preds = 0
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    with torch.no_grad():
        for batch, (instance, word_emb, sense_emb, rad) in enumerate(tqdm.tqdm(dataloader)):

            # Compute prediction and loss
            pred = model(word_emb)

            loss = my_loss(pred, sense_emb, rad)

            num_correct_preds += (loss.shape[0] - loss.count_nonzero()).item()

            loss = loss.sum()

            test_loss += loss.item()

    return test_loss / size, num_correct_preds / size 

print('Loading data from disk')
# Get the instance - embedding dictionary
with open('D:\Documents\Master\FraunhoferLab\\training_dataset_sense_embeddings.pkl', 'rb') as f:
    training_sense_emb = pickle.load(f)
    f.close()

with open('D:\Documents\Master\FraunhoferLab\\training_dataset_word_embeddings.pkl', 'rb') as f:
    training_word_emb = pickle.load(f)
    f.close()

for key in training_word_emb.keys():
    training_word_emb[key] = np.squeeze(training_word_emb[key])

print('Removing duplicate senses')
# Get the unique semcor instance - gold key decitonary
# semcor_gold_keys = {}
# with open('D:\Documents\Master\FraunhoferLab\WSD_Evaluation_Framework\Training_Corpora\SemCor\semcor.gold.key.txt') as f:
#     for line in f:
#         line = line.strip().split()
#         if line[1] not in semcor_gold_keys.values():
#             semcor_gold_keys[line[0]] = line[1]
#     f.close()

# Keep only the embeddings of the unique instances
assert training_sense_emb.keys() == training_word_emb.keys()
# instances_to_keep = list(semcor_gold_keys.keys())

# keys = list(training_sense_emb.keys())
# for inst in keys:
#     if inst not in instances_to_keep:
#         training_sense_emb.pop(inst)
#         training_word_emb.pop(inst) 

print('Splitting to training/testing')
# Split to train/test
num_samples = len(training_word_emb)
num_training_samples = int(num_samples * 0.7)
keys = list(training_word_emb.keys())
random.shuffle(keys)
training_keys = keys[:num_training_samples]
testing_keys = keys[num_training_samples:]

print('Normalizing by dividing with max norm of training part')
# Normalize
a = np.array([v for k, v in training_word_emb.items() if k in training_keys])
word_emb_norms = np.linalg.norm(a, axis=1)
max_word_emb_norm = max(np.linalg.norm(a, axis=1))

a = np.array([v for k, v in training_sense_emb.items() if k in training_keys])
sense_emb_norms = np.linalg.norm(a, axis=1)
max_sense_emb_norm = max(np.linalg.norm(a, axis=1))

max_norm = max(max_word_emb_norm, max_sense_emb_norm)

training_word_emb = {k: v/max_norm for k, v in training_word_emb.items()}
training_sense_emb = {k: v/max_norm for k, v in training_sense_emb.items()}

print('Creating datasets/dataloaders')
# Create dataset
training_dataset = Word_Sense_Embeddings_Dataset({k: v for k, v in training_word_emb.items() if k in training_keys}, {k: v for k, v in training_sense_emb.items() if k in training_keys})
test_dataset = Word_Sense_Embeddings_Dataset({k: v for k, v in training_word_emb.items() if k in testing_keys}, {k: v for k, v in training_sense_emb.items() if k in testing_keys})

# Create dataloaders
training_dataloader = data.DataLoader(training_dataset, batch_size=256, num_workers=0)
testing_dataloader = data.DataLoader(test_dataset, batch_size=256, num_workers=0)

# Create model
device = 'cuda'
net = MLP(768, 768)
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

epochs = 50
train_losses = []
test_losses = []
test_accuracies = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch_train_loss = train_loop(training_dataloader, net, optimizer)
    train_losses.append(epoch_train_loss)
    print('Train loss: {} \n'.format(epoch_train_loss))
    test_loss, test_accuracy = test_loop(testing_dataloader, net)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print('Test loss: {} , Test accuracy: {} \n'.format(test_loss, test_accuracy))

# Create count of the number of epochs
range_epochs = range(1, len(train_losses) + 1)
# Visualize loss history
plt.plot(range_epochs, train_losses, 'r-', label='Train Loss')
plt.plot(range_epochs, test_losses, 'b-', label='Test Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('D:\Documents\Master\FraunhoferLab\\loss_0.0001_all.png', format='png', dpi=300, bbox_inches='tight')
plt.close()

# Visualize accuracy
plt.plot(range_epochs, test_accuracies, 'r-', label='Test Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('D:\Documents\Master\FraunhoferLab\\acc_0.0001_all.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
pass