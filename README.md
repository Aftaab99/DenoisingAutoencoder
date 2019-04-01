### Denoising Autoencoder
Implementation of a denoising autoencoder trained on the RENOIR dataset. 

#### Known Issues
Currently training doesn't converge and outputs all zeros due to large negative weights and dying ReLUs(leaky ReLU doesn't help much). If someone figures out whats wrong with my architecture please open a PR.
