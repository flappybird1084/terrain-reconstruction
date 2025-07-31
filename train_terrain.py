
import torch
import torch.optim as optim
import torch.nn as nn
from util.unet import UNet
import torchvision.transforms as transforms
import util.dataset as ds
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.models as models


dataset_path = "../../Other/cosmos/data/terrain_reconstruction/_dataset/"


transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

dataset = ds.TerrainDataset(dataset_path, transform=transform_pipeline)

# Example: 80% train, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

# from unet import UNet
device = torch.device("mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu")

# initialize dataloaders
numworkers = 0
batchsize = 8
train_loader = DataLoader(
    dataset_train, batch_size=batchsize, shuffle=True, num_workers=numworkers)
test_loader = DataLoader(dataset_test, batch_size=batchsize,
                         shuffle=False, num_workers=numworkers)


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=9):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT).features[:feature_layer].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, pred, target):
        pred = self.transform(pred)
        target = self.transform(target)
        return nn.functional.mse_loss(self.vgg(pred), self.vgg(target))


def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
        torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


# Initialize UNet model
unet_model = UNet(in_channels=3, out_channels=3).to(device)
# criterion = nn.MSELoss()
mse_loss = nn.MSELoss()
perceptual_loss = PerceptualLoss().to(device)
perceptual_loss_scaling_factor = 0.1  # Adjust this factor based on your needs
optimizer = optim.Adam(unet_model.parameters(), lr=0.001)


train_previous = False
if train_previous:
    unet_model.load_state_dict(torch.load(
        './models/terrain/turbo_terrain_unet_model.pth'))
    print("Loaded previous model state from './models/terrain/turbo_terrain_unet_model.pth'")

num_epochs = 5
for epoch in range(num_epochs):
    # save model to checkpoints
    # torch.save(unet_model.state_dict(
    # ), f'./models/checkpoints/terrain/turbo_terrain_unet_model_epoch_{epoch + 1}.pth')
    # unet_model.train()
    running_loss = 0.0

    for i, (height, terrain, segmentation) in enumerate(train_loader):
        terrain = (terrain * 2) - 1   # if originally ∈ [0,1]
        # CHECK ABOVE LINE
        images = segmentation
        images = images.to(device)
        target_images = terrain
        target_images = target_images.to(device)

        # Forward pass
        outputs = unet_model(images)
        # loss = criterion(outputs, target_images)
        loss = mse_loss(outputs, target_images) + perceptual_loss_scaling_factor * \
            perceptual_loss(outputs, target_images)
        running_loss += loss.item()
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            # Use end='' to avoid new line
            # print(f'\rEpoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{
            #       len(train_loader)}], Loss: {loss.item():.4f}', end='', flush=True)
            print(f"epoch: {epoch+1}")
            print(f"step: {i+1}/{len(train_loader)}")
            print(f"loss: {loss.item():.4f}")


torch.save(unet_model.state_dict(),
           './models/terrain/turbo_terrain_unet_model.pth')
print("Model saved to './models/terrain/turbo_terrain_unet_model.pth'")
