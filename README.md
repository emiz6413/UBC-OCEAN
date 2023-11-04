# UBC-OCEAN

This repository was developped for a kaggle competition ["UBC-OCEAN"](https://www.kaggle.com/competitions/UBC-OCEAN)

The difficulty of this competition comes from the size of whole slide images (WSI) which are typically gigapixel.

Neural image compression method ([Tellez et al., 2020](https://arxiv.org/abs/1811.02840)) is employed to compress WSI to a much smaller feature space followed by a typical CNN to classify cancer subtype.

Following [Tellez et al. (2020)](https://arxiv.org/abs/1811.02840), Bidirectional GAN ([Donahue et al., 2017](https://arxiv.org/abs/1605.09782)) is used to compress WSI. To stabilize the BiGAN's training, spectral normalization ([Miyato et al., 2018](https://arxiv.org/abs/1802.05957)) is applied to every convolutional layer in the discriminator.

# Preliminary results

### BiGAN on SVHN

| generated | reconstructed |
| - | - |
| <img src="asset/svhn_generated.png"> | <img src="asset/svhn_reconstructed.png"> |
