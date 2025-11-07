"""
CAC-CycleGAN-WGP PyTorch Implementation
Conditional Auxiliary Classifier CycleGAN with Wasserstein Gradient Penalty
For Fault Diagnosis

Converted from TensorFlow to PyTorch
Compatible with: PyTorch 2.5.1, Python 3.12, CUDA 12.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import datetime
import os
from sklearn.utils import shuffle
from cyclegan_sample_generation_new_and_svm import samlpe_generation_feed_svm

# ==================== Custom Layers ====================

class RandomWeightedAverage(nn.Module):
    """Provides a (random) weighted average between real and generated samples"""
    def __init__(self, batch_size):
        super(RandomWeightedAverage, self).__init__()
        self.batch_size = batch_size
    
    def forward(self, real_samples, fake_samples):
        alpha = torch.rand(self.batch_size, 1, device=real_samples.device)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        return interpolated


# ==================== Discriminator ====================

class Discriminator(nn.Module):
    """
    Discriminator with Auxiliary Classifier
    Input: (batch, 512) -> Reshape to (batch, 1, 32, 16)
    Outputs: validity (real/fake) and class label
    """
    def __init__(self, data_length=512, df=32, num_classes=10):
        super(Discriminator, self).__init__()
        self.data_length = data_length
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, df, kernel_size=5, stride=2, padding=2)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(df, df*2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(df*2, momentum=0.8)
        self.leaky2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(df*2, df*4, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(df*4, momentum=0.8)
        self.leaky3 = nn.LeakyReLU(0.2)
        self.dropout3 = nn.Dropout(0.25)
        
        self.conv4 = nn.Conv2d(df*4, df*8, kernel_size=5, stride=2, padding=2)
        self.leaky4 = nn.LeakyReLU(0.2)
        self.dropout4 = nn.Dropout(0.25)
        
        # Calculate flattened size: 32x16 -> 16x8 -> 8x4 -> 4x2 -> 2x1
        self.flatten_size = df * 8 * 2 * 1
        
        # Output layers
        self.validity_layer = nn.Linear(self.flatten_size, 1)
        self.label_layer = nn.Linear(self.flatten_size, num_classes)
        
    def forward(self, x):
        # Reshape: (batch, 512) -> (batch, 1, 32, 16)
        x = x.view(-1, 1, 32, 16)
        
        # Conv blocks
        x = self.dropout1(self.leaky1(self.conv1(x)))
        x = self.dropout2(self.leaky2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.leaky3(self.bn3(self.conv3(x))))
        x = self.dropout4(self.leaky4(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Outputs
        validity = self.validity_layer(x)
        label = self.label_layer(x)
        
        return validity, label


# ==================== Generator ====================

class Generator(nn.Module):
    """
    Conditional U-Net Generator
    Input: sample (batch, 512) + label (batch, 1)
    Output: generated sample (batch, 512)
    """
    def __init__(self, data_length=512, gf=32, num_classes=10):
        super(Generator, self).__init__()
        self.data_length = data_length
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 32*16)
        
        # Encoder (Downsampling)
        # Input: (batch, 2, 32, 16) - sample + embedded label concatenated
        self.down1 = self._conv_block(2, gf)           # -> (batch, gf, 16, 8)
        self.down2 = self._conv_block(gf, gf*2)        # -> (batch, gf*2, 8, 4)
        self.down3 = self._conv_block(gf*2, gf*4)      # -> (batch, gf*4, 4, 2)
        self.down4 = self._conv_block(gf*4, gf*8)      # -> (batch, gf*8, 2, 1)
        
        # Decoder (Upsampling)
        self.up1 = self._deconv_block(gf*8, gf*4)      # -> (batch, gf*4, 4, 2)
        self.up2 = self._deconv_block(gf*4*2, gf*2)    # -> (batch, gf*2, 8, 4) (concat skip)
        self.up3 = self._deconv_block(gf*2*2, gf)      # -> (batch, gf, 16, 8) (concat skip)
        
        # Final upsampling
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Conv2d(gf*2, 1, kernel_size=5, stride=1, padding=2)
        
        # Final dense layer
        self.final_dense = nn.Linear(32*16, 512)
        
    def _conv_block(self, in_channels, out_channels, kernel_size=5):
        """Encoder block: Conv2d -> LeakyReLU -> BatchNorm"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=2, padding=kernel_size//2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels)
        )
    
    def _deconv_block(self, in_channels, out_channels, kernel_size=5, dropout_rate=0):
        """Decoder block: Upsample -> Conv2d -> ReLU -> BatchNorm"""
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=1, padding=kernel_size//2),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, sample, label):
        # Reshape sample: (batch, 512) -> (batch, 1, 32, 16)
        reshaped_sample = sample.reshape(-1, 1, 32, 16)
        
        # Embed and reshape label: (batch, 1) -> (batch, 32*16) -> (batch, 1, 32, 16)
        embedded_label = self.label_embedding(label.squeeze(-1))
        embedded_label = embedded_label.reshape(-1, 1, 32, 16)
        
        # Concatenate sample and label
        x = torch.cat([reshaped_sample, embedded_label], dim=1)  # (batch, 2, 32, 16)
        
        # Encoder with skip connections
        d1 = self.down1(x)      # (batch, gf, 16, 8)
        d2 = self.down2(d1)     # (batch, gf*2, 8, 4)
        d3 = self.down3(d2)     # (batch, gf*4, 4, 2)
        d4 = self.down4(d3)     # (batch, gf*8, 2, 1)
        
        # Decoder with skip connections
        u1 = self.up1(d4)                              # (batch, gf*4, 4, 2)
        u1 = torch.cat([u1, d3], dim=1)                # (batch, gf*4*2, 4, 2)
        
        u2 = self.up2(u1)                              # (batch, gf*2, 8, 4)
        u2 = torch.cat([u2, d2], dim=1)                # (batch, gf*2*2, 8, 4)
        
        u3 = self.up3(u2)                              # (batch, gf, 16, 8)
        u3 = torch.cat([u3, d1], dim=1)                # (batch, gf*2, 16, 8)
        
        # Final upsampling
        u4 = self.up4(u3)                              # (batch, gf*2, 32, 16)
        output = self.final_conv(u4)                   # (batch, 1, 32, 16)
        
        # Flatten and project to output dimension
        output = output.view(-1, 32*16)                # (batch, 512)
        output = self.final_dense(output)              # (batch, 512)
        
        return output


# ==================== CycleGAN Model ====================

class CycleGAN:
    """
    CAC-CycleGAN-WGP: Conditional Auxiliary Classifier CycleGAN 
    with Wasserstein Gradient Penalty
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.data_length = 512
        self.sample_shape = (self.data_length,)
        self.num_classes = 10
        self.batch_size = 45
        self.n_critic = 1
        
        # Architecture parameters
        self.gf = 32
        self.df = 32
        
        # Loss weights
        self.lambda_adv = 1
        self.lambda_cycle = 10
        self.lambda_id = 0.1 * self.lambda_cycle
        self.lambda_fake_A_classification = 1
        self.lambda_fake_B_classification = 1
        self.lambda_gp = 10
        self.label_weight = 1
        
        # Build models
        self.d_1 = Discriminator(self.data_length, self.df, self.num_classes).to(device)
        self.d_2 = Discriminator(self.data_length, self.df, self.num_classes).to(device)
        self.g_AB = Generator(self.data_length, self.gf, self.num_classes).to(device)
        self.g_BA = Generator(self.data_length, self.gf, self.num_classes).to(device)
        
        # Optimizers
        self.d_optimizer_1 = torch.optim.RMSprop(self.d_1.parameters(), lr=0.001)
        self.d_optimizer_2 = torch.optim.RMSprop(self.d_2.parameters(), lr=0.001)
        self.g_optimizer = torch.optim.RMSprop(
            list(self.g_AB.parameters()) + list(self.g_BA.parameters()), 
            lr=0.001
        )
        
        # Random weighted average layer
        self.rwa = RandomWeightedAverage(self.batch_size)
        
        print(f"Models initialized on {device}")
        print(f"Discriminator 1 parameters: {sum(p.numel() for p in self.d_1.parameters()):,}")
        print(f"Discriminator 2 parameters: {sum(p.numel() for p in self.d_2.parameters()):,}")
        print(f"Generator AB parameters: {sum(p.numel() for p in self.g_AB.parameters()):,}")
        print(f"Generator BA parameters: {sum(p.numel() for p in self.g_BA.parameters()):,}")
    
    def wasserstein_loss(self, y_true, y_pred):
        """Wasserstein loss"""
        return torch.mean(y_true * y_pred)
    
    def gradient_penalty_loss(self, gradients):
        """Calculate gradient penalty for WGAN-GP"""
        gradients_sqr = gradients ** 2
        gradients_sqr_sum = torch.sum(gradients_sqr, dim=1)
        gradient_l2_norm = torch.sqrt(gradients_sqr_sum)
        gradient_penalty = (gradient_l2_norm - 1) ** 2
        return torch.mean(gradient_penalty)
    
    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP"""
        # Interpolate between real and fake samples
        interpolated = self.rwa(real_samples, fake_samples)
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        d_interpolated, _ = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = self.gradient_penalty_loss(gradients)
        
        return gradient_penalty
    
    def train_discriminator_1(self, batch_samples_A, batch_labels_A, batch_samples_B):
        """Train discriminator 1 (for domain A - healthy samples)"""
        self.d_optimizer_1.zero_grad()
        
        batch_size = batch_samples_A.size(0)
        valid = -torch.ones(batch_size, 1, device=self.device)
        fake = torch.ones(batch_size, 1, device=self.device)
        
        # Generate fake samples
        with torch.no_grad():
            fake_A = self.g_BA(batch_samples_B, batch_labels_A)
        
        # Get discriminator predictions
        d_pred_real, d_label_real = self.d_1(batch_samples_A)
        d_pred_fake, d_label_fake = self.d_1(fake_A)
        
        # Compute Wasserstein loss
        d_loss_real = self.wasserstein_loss(valid, d_pred_real)
        d_loss_fake = self.wasserstein_loss(fake, d_pred_fake)
        
        # Compute gradient penalty
        gradient_penalty = self.compute_gradient_penalty(
            self.d_1, batch_samples_A, fake_A
        )
        
        # Classification losses
        label_loss_real = F.cross_entropy(d_label_real, batch_labels_A.squeeze(-1).long())
        label_loss_fake = F.cross_entropy(d_label_fake, batch_labels_A.squeeze(-1).long())
        
        # Total discriminator loss
        d_loss_1 = (d_loss_real + d_loss_fake + 
                   self.lambda_gp * gradient_penalty + 
                   self.label_weight * (label_loss_real + label_loss_fake))
        
        d_loss_1.backward()
        self.d_optimizer_1.step()
        
        return d_loss_1.item()
    
    def train_discriminator_2(self, batch_samples_A, batch_samples_B, batch_labels_B):
        """Train discriminator 2 (for domain B - fault samples)"""
        self.d_optimizer_2.zero_grad()
        
        batch_size = batch_samples_B.size(0)
        valid = -torch.ones(batch_size, 1, device=self.device)
        fake = torch.ones(batch_size, 1, device=self.device)
        
        # Generate fake samples
        with torch.no_grad():
            fake_B = self.g_AB(batch_samples_A, batch_labels_B)
        
        # Get discriminator predictions
        d_pred_real, d_label_real = self.d_2(batch_samples_B)
        d_pred_fake, d_label_fake = self.d_2(fake_B)
        
        # Compute Wasserstein loss
        d_loss_real = self.wasserstein_loss(valid, d_pred_real)
        d_loss_fake = self.wasserstein_loss(fake, d_pred_fake)
        
        # Compute gradient penalty
        gradient_penalty = self.compute_gradient_penalty(
            self.d_2, batch_samples_B, fake_B
        )
        
        # Classification losses
        label_loss_real = F.cross_entropy(d_label_real, batch_labels_B.squeeze(-1).long())
        label_loss_fake = F.cross_entropy(d_label_fake, batch_labels_B.squeeze(-1).long())
        
        # Total discriminator loss
        d_loss_2 = (d_loss_real + d_loss_fake + 
                   self.lambda_gp * gradient_penalty + 
                   self.label_weight * (label_loss_real + label_loss_fake))
        
        d_loss_2.backward()
        self.d_optimizer_2.step()
        
        return d_loss_2.item()
    
    def train_generators(self, batch_samples_A, batch_labels_A, 
                        batch_samples_B, batch_labels_B):
        """Train both generators"""
        self.g_optimizer.zero_grad()
        
        batch_size = batch_samples_A.size(0)
        valid = -torch.ones(batch_size, 1, device=self.device)
        
        # Translate samples to the other domain
        fake_B = self.g_AB(batch_samples_A, batch_labels_B)
        fake_A = self.g_BA(batch_samples_B, batch_labels_A)
        
        # Translate samples back to original domain (cycle consistency)
        reconstr_A = self.g_BA(fake_B, batch_labels_A)
        reconstr_B = self.g_AB(fake_A, batch_labels_B)
        
        # Identity mapping
        sample_A_id = self.g_BA(batch_samples_A, batch_labels_A)
        sample_B_id = self.g_AB(batch_samples_B, batch_labels_B)
        
        # Discriminator evaluation
        valid_A_for_fake_A, label_A_for_fake_A = self.d_1(fake_A)
        valid_B_for_fake_B, label_B_for_fake_B = self.d_2(fake_B)
        
        # Adversarial losses
        adv_loss_A = self.wasserstein_loss(valid, valid_A_for_fake_A)
        adv_loss_B = self.wasserstein_loss(valid, valid_B_for_fake_B)
        
        # Cycle consistency losses
        cycle_loss_A = F.l1_loss(batch_samples_A, reconstr_A)
        cycle_loss_B = F.l1_loss(batch_samples_B, reconstr_B)
        
        # Identity losses
        id_loss_A = F.l1_loss(batch_samples_A, sample_A_id)
        id_loss_B = F.l1_loss(batch_samples_B, sample_B_id)
        
        # Classification losses
        class_loss_A = F.cross_entropy(label_A_for_fake_A, batch_labels_A.squeeze(-1).long())
        class_loss_B = F.cross_entropy(label_B_for_fake_B, batch_labels_B.squeeze(-1).long())
        
        # Total generator loss
        g_loss = (self.lambda_adv * (adv_loss_A + adv_loss_B) +
                 self.lambda_cycle * (cycle_loss_A + cycle_loss_B) +
                 self.lambda_id * (id_loss_A + id_loss_B) +
                 self.lambda_fake_A_classification * class_loss_A +
                 self.lambda_fake_B_classification * class_loss_B)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        losses = {
            'g_total': g_loss.item(),
            'g_adv': (adv_loss_A + adv_loss_B).item() / 2,
            'g_cycle': (cycle_loss_A + cycle_loss_B).item() / 2,
            'g_id': (id_loss_A + id_loss_B).item() / 2,
            'g_class_A': class_loss_A.item(),
            'g_class_B': class_loss_B.item()
        }
        
        return losses
    
    def train(self, epochs, data_path, save_dir='./saved_models', best_model_name = 'best_cyclegan_model.pth', add_quantity=None,
              save_interval=100, svm_function=None):
        """
        Train the CycleGAN model
        
        Args:
            epochs: Number of training epochs
            data_path: Path to .npz data file
            save_dir: Directory to save models
            save_interval: Save model every N epochs
            svm_function: Optional function for SVM evaluation
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Load data
        print(f"Loading data from {data_path}")
        data = np.load(data_path)
        
        domain_A_train_X = data['domain_A_train_X']
        domain_A_train_Y = data['domain_A_train_Y']
        
        # Load domain B data from multiple fault types
        domain_B_train_X = np.concatenate([
            data[f'domain_B_train_X_{i}'][:5] for i in range(9)
        ], axis=0)
        
        domain_B_train_Y = np.concatenate([
            data[f'domain_B_train_Y_{i}'][:5] for i in range(9)
        ], axis=0)
        
        test_X = data['test_X']
        test_Y = data['test_Y']
        
        # Use first 900 samples from domain A
        domain_A_train_X_900 = domain_A_train_X[:900]
        domain_A_train_Y_900 = domain_A_train_Y[:900]
        
        num_batches = int(domain_A_train_X_900.shape[0] / self.batch_size)
        
        print(f"Training data loaded:")
        print(f"  Domain A (for training): {domain_A_train_X_900.shape}")
        print(f"  Domain A (for evaluation): {domain_A_train_X.shape}")
        print(f"  Domain B (for training): {domain_B_train_X.shape}")
        print(f"  Domain B (for evaluation): {domain_B_train_X.shape}")
        print(f"  Test: {test_X.shape}")
        print(f"  Batches per epoch: {num_batches}")

        # 初始化最佳准确率跟踪
        best_accuracy = 0.0
        accuracy_list = []
        best_model_path = os.path.join(save_dir, best_model_name)

        print(f"\nBest model will be saved to: {best_model_path}")
        print("="*80)

        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = datetime.datetime.now()
            
            for batch_i in range(num_batches):
                start_i = batch_i * self.batch_size
                end_i = (batch_i + 1) * self.batch_size
                
                # Prepare batch data
                batch_samples_A = torch.FloatTensor(
                    domain_A_train_X_900[start_i:end_i]
                ).to(self.device)
                batch_labels_A = torch.LongTensor(
                    domain_A_train_Y_900[start_i:end_i].reshape(-1, 1)
                ).to(self.device)
                
                batch_samples_B = torch.FloatTensor(domain_B_train_X).to(self.device)
                batch_labels_B = torch.LongTensor(
                    domain_B_train_Y.reshape(-1, 1)
                ).to(self.device)
                
                # Train discriminators
                for _ in range(self.n_critic):
                    d_loss_1 = self.train_discriminator_1(
                        batch_samples_A, batch_labels_A, batch_samples_B
                    )
                    d_loss_2 = self.train_discriminator_2(
                        batch_samples_A, batch_samples_B, batch_labels_B
                    )
                
                # Train generators
                g_losses = self.train_generators(
                    batch_samples_A, batch_labels_A,
                    batch_samples_B, batch_labels_B
                )
                
                elapsed_time = datetime.datetime.now() - epoch_start_time
                
                # Print progress
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {batch_i+1}/{num_batches}] "
                      f"[D_1 loss: {d_loss_1:.5f}] [D_2 loss: {d_loss_2:.5f}] "
                      f"[G loss: {g_losses['g_total']:.5f}, "
                      f"adv: {g_losses['g_adv']:.5f}, "
                      f"recon: {g_losses['g_cycle']:.5f}, "
                      f"id: {g_losses['g_id']:.5f}, "
                      f"classA: {g_losses['g_class_A']:.5f}, "
                      f"classB: {g_losses['g_class_B']:.5f}] "
                      f"time: {elapsed_time}")

            # ========== 每个 epoch 都进行 SVM 评估 ==========
            print("\n" + "="*80)
            print(f"Evaluating model at epoch {epoch+1}...")

            # 调用 SVM 评估函数
            current_accuracy = samlpe_generation_feed_svm(
                add_quantity=add_quantity,
                test_x=test_X,
                test_y=test_Y,
                generator=self.g_AB,
                domain_A_train_x=domain_A_train_X,  # 使用完整的 domain A
                domain_A_train_y=domain_A_train_Y,
                domain_B_train_x=domain_B_train_X,  # 使用完整的 domain B
                domain_B_train_y=domain_B_train_Y,
                c = 0.2,
                g = 0.01
            )

            accuracy_list.append(current_accuracy)
            
            # 打印当前准确率和最佳准确率
            print(f"\n[Epoch {epoch+1}/{epochs}] Current Accuracy: {current_accuracy:.4f}")
            print(f"[Epoch {epoch+1}/{epochs}] Best Accuracy So Far: {best_accuracy:.4f}")
            
            # 如果当前准确率优于最佳准确率，保存模型
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy

                # 保存最佳模型（覆盖）
                torch.save({
                    'epoch': epoch + 1,
                    'best_accuracy': best_accuracy,
                    'g_AB_state_dict': self.g_AB.state_dict(),
                    'g_BA_state_dict': self.g_BA.state_dict(),
                    'd_1_state_dict': self.d_1.state_dict(),
                    'd_2_state_dict': self.d_2.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_1_state_dict': self.d_optimizer_1.state_dict(),
                    'd_optimizer_2_state_dict': self.d_optimizer_2.state_dict(),
                }, best_model_path)

                print(f"🎉 NEW BEST MODEL SAVED! Accuracy improved: {best_accuracy:.4f}")
                print(f"   Model saved to: {best_model_path}")
            else:
                print(f"   No improvement. Best accuracy remains: {best_accuracy:.4f}")
            
            print(f"\nAccuracy history: {[f'{acc:.4f}' for acc in accuracy_list[-10:]]}")  # 显示最近10个
            print("="*80 + "\n")
        
        print("\n" + "="*80)
        print("Training completed!")
        print(f"Final best accuracy: {best_accuracy:.4f}")
        print(f"Best model saved at: {best_model_path}")
        print("="*80)
        
        return accuracy_list, best_accuracy

        
    
    def generate_samples(self, samples, labels, generator='g_AB'):
        """
        Generate samples using specified generator
        
        Args:
            samples: Input samples (numpy array or tensor)
            labels: Target labels (numpy array or tensor)
            generator: 'g_AB' or 'g_BA'
        
        Returns:
            Generated samples as numpy array
        """
        # Convert to tensors if needed
        if isinstance(samples, np.ndarray):
            samples = torch.FloatTensor(samples).to(self.device)
        if isinstance(labels, np.ndarray):
            labels = torch.LongTensor(labels.reshape(-1, 1)).to(self.device)
        
        # Select generator
        gen = self.g_AB if generator == 'g_AB' else self.g_BA
        
        # Generate
        gen.eval()
        with torch.no_grad():
            generated = gen(samples, labels)
        gen.train()
        
        return generated.cpu().numpy()
    
    def save_model(self, path):
        """Save complete model"""
        torch.save({
            'g_AB_state_dict': self.g_AB.state_dict(),
            'g_BA_state_dict': self.g_BA.state_dict(),
            'd_1_state_dict': self.d_1.state_dict(),
            'd_2_state_dict': self.d_2.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load complete model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.g_AB.load_state_dict(checkpoint['g_AB_state_dict'])
        self.g_BA.load_state_dict(checkpoint['g_BA_state_dict'])
        self.d_1.load_state_dict(checkpoint['d_1_state_dict'])
        self.d_2.load_state_dict(checkpoint['d_2_state_dict'])
        print(f"Model loaded from {path}")


# ==================== Main ====================

if __name__ == '__main__':
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Initialize model
    gan = CycleGAN(device=device)
    
    # Example: Train model
    # You'll need to provide the correct data path
    data_path = './dataset/dataset_fft_for_cyclegan_case1_512.npz'
    
    # Check if data exists
    if os.path.exists(data_path):
        gan.train(
            epochs=5000,
            data_path=data_path,
            save_dir='./saved_models',
            save_interval=100
        )
        
        # Save final model
        gan.save_model('./saved_models/cyclegan_final.pth')
    else:
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable with your actual data path")
        print("\nModel architecture initialized successfully!")
        print("To train, provide the correct data path in the script.")
