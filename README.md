# CAC-CycleGAN-WGP
The implementation of CAC-CycleGAN-WGP for bearing fault daignosis based on imbalanced dataset

# Paper
W. Liao, L. Wu, S. Xu and S. Fujimura, "A Novel Approach for Intelligent Fault Diagnosis in Bearing With Imbalanced Data Based on Cycle-Consistent GAN," in IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-16, 2024, Art no. 3525416, doi: 10.1109/TIM.2024.3427866

# Dataset
This example is based on the Case Western Reserve University bearing dataset

# Requirements
```
python = 3.9.20
Tensorflow = 2.10.0
pandas = 1.3.5
scikit-learn = 1.0.2                                                                                                     `
```
The codes was tested with tensorflow 2.10.0, CUDA driver 12.7, CUDA toolkit 11.8, Ubuntu 22.04.

# Quick strat
```
python CAC_CycleGAN_WGP.py
```
The dataset has been convert to fresuency domain by FFT and each sample contains 512 points.
