import torch
from VQModel import VQModel
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
import scipy
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torch.nn.functional as F
from lpips import LPIPS
from discriminator import Discriminator
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AdamW


# set hyper parameters
latent_dim = 128
learning_rate = 0.00001
batch_size = 32
num_vectors = 128
device = 'cuda'
num_epochs = 50
hidden_dim = 32
beta = .25
EOS_TOKEN = 1024

n = latent_dim
k = num_vectors

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load the Flowers102 dataset
train_dataset = Flowers102(
    root='data',
    split='train',
    transform=transform,
    download=False
)

# Create a DataLoader for the training set
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,  # Number of samples per batch
    shuffle=True,  # Shuffle the dataset
    num_workers=2  # Number of subprocesses for data loading
)
if __name__ == '__main__':

    model = VQModel(n,k,hidden_dim,beta).to(device)
    model.train()
    model.load_state_dict(torch.load('VQ46.pth'))

    # Initialize GPT-2 model
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Modify GPT-2 to handle your specific codebook size
    codebook_size = num_vectors
    gpt2.resize_token_embeddings(codebook_size)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(gpt2.parameters(), lr=5e-5, weight_decay=0.01)

    for epoch in range(num_epochs):

        cur_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs, loss2, min_encoding_indices = model(images)
            min_encoding_indices = min_encoding_indices.to(device)
            # print(min_encoding_indices.shape)
            input_ids = min_encoding_indices.clone()  # Keep original sequence for input
            labels = torch.cat([min_encoding_indices[:, 1:], torch.full((min_encoding_indices.size(0), 1), EOS_TOKEN, dtype=torch.long).to(device)], dim=1)

            labels = labels[1:2]
            min_encoding_indices = min_encoding_indices[1:2]

            transformer_outputs = gpt2(input_ids=min_encoding_indices, labels=labels)

            loss = transformer_outputs.loss  # Cross-entropy loss
            # logits = outputs.logits  # Predictions for next token

            # print(transformer_outputs)

            loss.backward()
            optimizer.step()


            # if cur_loss == 0:
            #     save_image(images[:16], f"imgs4/epoch_{epoch + 1+50}_samples.png", nrow=4, normalize=True)
            #     save_image(outputs[:16], f"imgs4/epoch_{epoch + 1+50}_ours2.png", nrow=4, normalize=True)


            cur_loss += loss.item()


        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {cur_loss / len(train_loader):.4f}")

