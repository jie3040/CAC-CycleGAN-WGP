"""
Training Script for CAC-CycleGAN-WGP
PyTorch Implementation

This script demonstrates how to train the CycleGAN model on fault diagnosis data
with evaluation every epoch and automatic best model saving
"""

import torch
import numpy as np
import os
from CAC_CycleGAN_WGP_pytorch import CycleGAN


def main():
    # ==================== Configuration ====================
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    
    # Training parameters
    EPOCHS = 5000
    BATCH_SIZE = 45
    ADD_QUANTITY = 995  # 每个故障类别生成的样本数用于评估
    
    # Paths
    DATA_PATH = './dataset/dataset_fft_for_cyclegan_case1_512.npz'
    SAVE_DIR = './saved_models'
    BEST_MODEL_NAME = 'best_cyclegan_model.pth'  # 最佳模型文件名
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ==================== Initialize Model ====================
    
    print("\n" + "="*50)
    print("Initializing CAC-CycleGAN-WGP Model")
    print("="*50)
    
    gan = CycleGAN(device=device)
    
    # ==================== Check Data ====================
    
    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Data file not found: {DATA_PATH}")
        print("\nPlease update DATA_PATH with your actual data file location.")
        print("Expected format: .npz file with the following keys:")
        print("  - domain_A_train_X: Healthy samples")
        print("  - domain_A_train_Y: Healthy labels")
        print("  - domain_B_train_X_0 to domain_B_train_X_8: Fault samples for each class")
        print("  - domain_B_train_Y_0 to domain_B_train_Y_8: Fault labels for each class")
        print("  - test_X: Test samples")
        print("  - test_Y: Test labels")
        return
    
    # ==================== Train Model ====================
    
    print("\n" + "="*50)
    print("Starting Training with Evaluation Every Epoch")
    print("="*50)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Generated samples per fault class: {ADD_QUANTITY}")
    print(f"Data path: {DATA_PATH}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Best model name: {BEST_MODEL_NAME}")
    print("="*50 + "\n")
    
    try:
        accuracy_list, best_accuracy = gan.train(
            epochs=EPOCHS,
            data_path=DATA_PATH,
            save_dir=SAVE_DIR,
            best_model_name=BEST_MODEL_NAME,
            add_quantity=ADD_QUANTITY
        )
        
        # ==================== Training Summary ====================
        
        print("\n" + "="*50)
        print("Training Completed Successfully!")
        print("="*50)
        print(f"Best accuracy achieved: {best_accuracy:.4f}")
        print(f"Best model saved to: {os.path.join(SAVE_DIR, BEST_MODEL_NAME)}")
        print(f"\nFull accuracy history:")
        for i, acc in enumerate(accuracy_list, 1):
            marker = " ← BEST" if acc == best_accuracy else ""
            print(f"  Epoch {i:4d}: {acc:.4f}{marker}")
        
        # Save accuracy history
        accuracy_file = os.path.join(SAVE_DIR, 'accuracy_history.txt')
        np.savetxt(accuracy_file, accuracy_list)
        print(f"\nAccuracy history saved to: {accuracy_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
