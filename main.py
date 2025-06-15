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


# set hyper parameters
latent_dim = 128
learning_rate = 0.001
batch_size = 32
num_vectors = 128
device = 'cuda'
num_epochs = 50
hidden_dim = 32
beta = .25

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
    discrimin = Discriminator(3).to(device)

    model.load_state_dict(torch.load('VQ46.pth'))

    v1 = sum(p.numel() for p in model.parameters())
    v2 = sum(p.numel() for p in discrimin.parameters())
    print(v1 +v2)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    perceptual_loss = LPIPS().eval().to(device=device)

    for epoch in range(num_epochs):

        cur_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs, loss, __ = model(images)

            # better reconsruction loss
            p_loss = perceptual_loss(images, outputs).mean()

            disc_real = discrimin(images)
            disc_fake = discrimin(outputs)

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))

            gan_loss = d_loss_fake + d_loss_real
            #todo if i add back gan i have to add the disc_fake.mean() loss to the total_loss

            # Reconstruction loss (pixel-wise)
            # recon_loss = F.mse_loss(outputs, images)

            total_loss = loss + p_loss
            # print(disc_fake.mean())
            total_loss += -disc_fake.mean()

            total_loss.backward(retain_graph=True)
            gan_loss.backward()
            optimizer.step()


            if cur_loss == 0:
                save_image(images[:16], f"imgs3/epoch_{epoch + 1+50}_samples.png", nrow=4, normalize=True)
                save_image(outputs[:16], f"imgs3/epoch_{epoch + 1+50}_ours2.png", nrow=4, normalize=True)


            cur_loss += total_loss.item()


        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {cur_loss / len(train_loader):.4f}")

        if epoch % 5 == 0:
            print("Save")
            torch.save(model.state_dict(), f'VQ{epoch + 1}.pth')


