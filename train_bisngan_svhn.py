import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import SVHN
from tqdm.auto import tqdm

from src.models.bi_sngan import BiSNGAN, Discriminator32, Encoder32, Generator32

BATCH_SIZE = 256
EPOCHS = 110
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambd=lambda i: i.to(DEVICE)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=(0.5, 0.5, 0.5)),
    ]
)

reverse_transform = transforms.Compose(
    [transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]), transforms.ToPILImage()]
)

if __name__ == "__main__":
    dataset = SVHN(root="dataset", download=True, split="extra", transform=transform)
    eval_dataset = SVHN(root="dataset", download=True, split="test", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    generator = Generator32(output_channels=3)
    encoder = Encoder32(input_channels=3)
    discriminator = Discriminator32(input_channels=3)

    model = BiSNGAN(encoder=encoder, generator=generator, discriminator=discriminator, amp=torch.cuda.is_available())
    model = model.to(DEVICE)

    pbar = tqdm(total=EPOCHS)
    for epoch in range(EPOCHS):
        ge_loss, disc_loss = model.train_single_epoch(train_loader=dataloader)

        if epoch % 10 == 0:
            rec_loss = model.evaluate(eval_loader=eval_loader)
            pbar.set_description(f"epoch: {epoch} Reconstruction loss: {rec_loss:.3f}")

            with torch.no_grad():
                gen = model.generate(torch.randn(64, 128, 1, 1, device=DEVICE))
                rec = model.reconstruct(next(iter(eval_loader))[0][:64])

            img = reverse_transform(vutils.make_grid(gen, nrow=8))
            img.save(f"asset/epoch_{epoch}_generated.png")

            img = reverse_transform(vutils.make_grid(rec, nrow=8))
            img.save(f"asset/epoch_{epoch}_reconstructed.png")

            torch.save(model.state_dict(), f"asset/bigan_epoch{epoch}.pth")

        pbar.update()
