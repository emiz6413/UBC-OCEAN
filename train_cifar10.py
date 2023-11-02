from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore
from torchvision.datasets import CIFAR10  # type: ignore

from src.models.bi_sngan import BiSNGAN, Discriminator32, Encoder32, Generator32

BATCH_SIZE = 64
EPOCHS = 200

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=(0.5, 0.5, 0.5))])


if __name__ == "__main__":
    dataset = CIFAR10(root="dataset", download=True, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator32(output_channels=3)
    encoder = Encoder32(input_channels=3)
    discriminator = Discriminator32(input_channels=3)

    model = BiSNGAN(encoder=encoder, generator=generator, discriminator=discriminator)

    for epoch in range(EPOCHS):
        ge_loss, disc_loss = model.train_single_epoch(train_loader=dataloader)
