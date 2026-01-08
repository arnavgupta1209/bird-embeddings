"""
Variational Autoencoder (VAE) for eBird Checklist Embeddings

This module implements a VAE architecture designed to learn compressed embeddings
of bird observation checklists from eBird data. The model encodes high-dimensional
binary/one-hot encoded species presence-absence data into a lower-dimensional 
continuous latent space.

Architecture:
    - Encoder: Input -> 3-layer MLP -> Latent mean & log-variance
    - Decoder: Latent sample -> 3-layer MLP -> Reconstructed input
    - Latent space: Gaussian distribution parameterized by mean and log-variance
    
The VAE is trained to minimize:
    1. Reconstruction loss (BCE between input and reconstruction)
    2. KL divergence (regularization to keep latent space close to N(0,1))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for learning embeddings from eBird checklists.
    
    The model learns to compress species presence-absence data into a continuous
    latent representation that captures meaningful ecological patterns.
    
    Args:
        input_dimension (int): Number of input features (e.g., number of species)
        latent_dimension (int): Dimensionality of the latent embedding space.
            Lower values = more compression, higher values = more information retained.
            Default: 16
        hidden_dimension (int): Number of neurons in hidden layers of encoder/decoder.
            Controls model capacity. Default: 512
            
    Example:
        >>> # For 500 bird species, create 16-dimensional embeddings
        >>> vae = VariationalAutoencoder(input_dimension=500, latent_dimension=16)
        >>> reconstructed, mu, logvar = vae(input_tensor)
    """
    
    def __init__(
        self,
        input_dimension: int,
        latent_dimension: int = 16,
        hidden_dimension: int = 512
    ):
        super().__init__()

        # -----------------------
        # Encoder network
        # -----------------------
        # Maps input (species presence/absence) to hidden representation
        # Architecture: 3-layer MLP with ReLU activations
        self.encoder_network = nn.Sequential(
            nn.Linear(in_features=input_dimension, out_features=hidden_dimension),
            nn.ReLU(),
            
            nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension),
            nn.ReLU(),
            
            nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension),
            nn.ReLU(),
        )
        
        # Latent space parameterization
        # VAE learns a distribution (not a single point) in latent space
        # Mean vector: center of the latent distribution for this input
        self.latent_mean_layer = nn.Linear(
            in_features=hidden_dimension,
            out_features=latent_dimension
        )
        
        # Log-variance vector: controls spread of latent distribution
        # Using log-variance for numerical stability (ensures positive variance)
        self.latent_log_variance_layer = nn.Linear(
            in_features=hidden_dimension,
            out_features=latent_dimension
        )

        # -----------------------
        # Decoder network
        # -----------------------
        # Maps latent sample back to reconstructed input space
        # Architecture: 3-layer MLP with ReLU + final Sigmoid
        self.decoder_network = nn.Sequential(
            nn.Linear(in_features=latent_dimension, out_features=hidden_dimension),
            nn.ReLU(),
            
            nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension),
            nn.ReLU(),
            
            nn.Linear(in_features=hidden_dimension, out_features=hidden_dimension),
            nn.ReLU(),
            
            nn.Linear(in_features=hidden_dimension, out_features=input_dimension),
            # Sigmoid ensures output is in [0, 1] range (probabilities for binary data)
            nn.Sigmoid()
        )

    def encode(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            input_tensor: Input data [batch_size, input_dimension]
            
        Returns:
            latent_mean: Mean of latent distribution [batch_size, latent_dimension]
            latent_log_variance: Log-variance of latent distribution [batch_size, latent_dimension]
        """
        encoder_output = self.encoder_network(input_tensor)
        latent_mean = self.latent_mean_layer(encoder_output)
        latent_log_variance = self.latent_log_variance_layer(encoder_output)
        return latent_mean, latent_log_variance

    def reparameterize(
        self, 
        latent_mean: torch.Tensor, 
        latent_log_variance: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, sigma^2) using N(0,1).
        
        This allows gradients to flow through the sampling operation during training.
        Instead of sampling z ~ N(mu, sigma^2) directly (which is non-differentiable),
        we sample epsilon ~ N(0,1) and compute z = mu + sigma * epsilon.
        
        Args:
            latent_mean: Mean of the distribution [batch_size, latent_dimension]
            latent_log_variance: Log-variance [batch_size, latent_dimension]
            
        Returns:
            latent_sample: Sampled latent vector [batch_size, latent_dimension]
        """
        # Convert log-variance to standard deviation
        latent_standard_deviation = torch.exp(0.5 * latent_log_variance)
        
        # Sample random noise from standard normal
        random_noise = torch.randn_like(latent_standard_deviation)
        
        # Reparameterization: z = mu + sigma * epsilon
        latent_sample = latent_mean + random_noise * latent_standard_deviation
        return latent_sample

    def decode(self, latent_sample: torch.Tensor) -> torch.Tensor:
        """
        Decode latent sample back to input space.
        
        Args:
            latent_sample: Latent vector [batch_size, latent_dimension]
            
        Returns:
            reconstructed_input: Reconstructed data [batch_size, input_dimension]
        """
        reconstructed_input = self.decoder_network(latent_sample)
        return reconstructed_input

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the VAE.
        
        Args:
            input_tensor: Input data [batch_size, input_dimension]
            
        Returns:
            reconstructed_input: Reconstructed data [batch_size, input_dimension]
            latent_mean: Mean of latent distribution [batch_size, latent_dimension]
            latent_log_variance: Log-variance of latent distribution [batch_size, latent_dimension]
        """
        # Encode input to latent distribution parameters
        latent_mean, latent_log_variance = self.encode(input_tensor)
        
        # Sample from the latent distribution
        latent_sample = self.reparameterize(latent_mean, latent_log_variance)
        
        # Decode the sample back to input space
        reconstructed_input = self.decode(latent_sample)
        
        return reconstructed_input, latent_mean, latent_log_variance


def variational_autoencoder_loss_function(
    reconstructed_input: torch.Tensor,
    original_input: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_log_variance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the VAE loss (ELBO - Evidence Lower Bound).
    
    The VAE optimizes the Evidence Lower Bound (ELBO), which consists of:
    1. Reconstruction loss: How well can we reconstruct the input?
    2. KL divergence: How different is our latent distribution from N(0,1)?
    
    Loss = Reconstruction Loss + KL Divergence
    
    Args:
        reconstructed_input: Model's reconstruction [batch_size, input_dimension]
        original_input: Original input data [batch_size, input_dimension]
        latent_mean: Mean of latent distribution [batch_size, latent_dimension]
        latent_log_variance: Log-variance of latent distribution [batch_size, latent_dimension]
        
    Returns:
        total_loss: Combined loss (scalar)
        reconstruction_loss: BCE reconstruction loss (scalar)
        kl_divergence_loss: KL divergence regularization term (scalar)
        
    Note:
        All losses use 'sum' reduction to match standard VAE formulation.
        Divide by batch_size if you want mean per-sample loss.
    """
    
    # --------------------------------------
    # 1. Reconstruction loss
    # --------------------------------------
    # Binary Cross Entropy: measures how well we reconstruct binary/probabilistic data
    # Assumes input is binary or in [0, 1] range (species presence/absence)
    reconstruction_loss = F.binary_cross_entropy(
        input=reconstructed_input,
        target=original_input,
        reduction='sum'  # Sum over all features and batch samples
    )
    
    # --------------------------------------
    # 2. KL divergence
    # --------------------------------------
    # KL(N(mu, sigma^2) || N(0, 1))
    # Closed-form solution for KL divergence between two Gaussians:
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    #
    # This regularizes the latent space to be close to a standard normal distribution,
    # which helps with:
    # - Smooth interpolation in latent space
    # - Generating new samples
    # - Preventing overfitting
    kl_divergence_loss = -0.5 * torch.sum(
        1 + latent_log_variance
        - latent_mean.pow(2)
        - torch.exp(latent_log_variance)
    )
    
    # --------------------------------------
    # 3. Total VAE loss (negative ELBO)
    # --------------------------------------
    # The total loss balances reconstruction quality with latent space regularization
    total_loss = reconstruction_loss + kl_divergence_loss
    
    return total_loss, reconstruction_loss, kl_divergence_loss
