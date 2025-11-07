"""
Load Model and Test Script
加载训练好的 CycleGAN 模型并进行测试

功能：
1. 加载保存的最佳模型
2. 使用 g_AB 从健康样本生成故障样本
3. 使用 g_BA 从故障样本生成健康样本
4. 调用 SVM 评估函数进行诊断
5. 可视化生成结果
"""

import torch
import numpy as np
import os
from CAC_CycleGAN_WGP_pytorch import CycleGAN
from cyclegan_sample_generation_new_and_svm import samlpe_generation_feed_svm
import matplotlib.pyplot as plt


class ModelTester:
    """模型测试类"""
    
    def __init__(self, model_path, data_path, device='cuda'):
        """
        初始化模型测试器
        
        Args:
            model_path: 模型文件路径
            data_path: 数据文件路径
            device: 运行设备 ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # 初始化模型
        print("\nInitializing CycleGAN model...")
        self.gan = CycleGAN(device=self.device)
        
        # 加载模型权重
        print(f"Loading model from: {model_path}")
        self.load_model(model_path)
        
        # 加载数据
        print(f"Loading data from: {data_path}")
        self.load_data(data_path)
        
        print("\n✓ Model and data loaded successfully!\n")
    
    def load_model(self, model_path):
        """加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 加载生成器和判别器权重
        self.gan.g_AB.load_state_dict(checkpoint['g_AB_state_dict'])
        self.gan.g_BA.load_state_dict(checkpoint['g_BA_state_dict'])
        self.gan.d_1.load_state_dict(checkpoint['d_1_state_dict'])
        self.gan.d_2.load_state_dict(checkpoint['d_2_state_dict'])
        
        # 设置为评估模式
        self.gan.g_AB.eval()
        self.gan.g_BA.eval()
        self.gan.d_1.eval()
        self.gan.d_2.eval()
        
        # 打印模型信息
        if 'epoch' in checkpoint:
            print(f"  Model from epoch: {checkpoint['epoch']}")
        if 'best_accuracy' in checkpoint:
            print(f"  Best accuracy: {checkpoint['best_accuracy']:.4f}")
    
    def load_data(self, data_path):
        """加载数据"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = np.load(data_path)
        
        # Domain A (健康样本)
        self.domain_A_train_X = data['domain_A_train_X']
        self.domain_A_train_Y = data['domain_A_train_Y']
        
        # Domain B (故障样本) - 完整数据
        self.domain_B_train_X = np.concatenate([
            data[f'domain_B_train_X_{i}'][:5] for i in range(9)
        ], axis=0)
        
        self.domain_B_train_Y = np.concatenate([
            data[f'domain_B_train_Y_{i}'][:5] for i in range(9)
        ], axis=0)
        
        # 测试数据
        self.test_X = data['test_X']
        self.test_Y = data['test_Y']
        
        print(f"  Domain A (healthy): {self.domain_A_train_X.shape}")
        print(f"  Domain B (fault): {self.domain_B_train_X.shape}")
        print(f"  Test set: {self.test_X.shape}")
    
    def generate_fault_samples(self, num_samples=10, fault_label=1, save_path=None):
        """
        使用 g_AB 从健康样本生成故障样本
        
        Args:
            num_samples: 要生成的样本数量
            fault_label: 目标故障类别 (0-8)
            save_path: 保存路径（可选）
        
        Returns:
            original_samples: 原始健康样本
            generated_samples: 生成的故障样本
        """
        print(f"\n{'='*80}")
        print(f"Generating Fault Samples using g_AB")
        print(f"{'='*80}")
        print(f"Number of samples: {num_samples}")
        print(f"Target fault label: {fault_label}")
        
        # 随机选择健康样本
        indices = np.random.choice(len(self.domain_A_train_X), num_samples, replace=False)
        original_samples = self.domain_A_train_X[indices]
        
        # 创建目标标签
        target_labels = np.full((num_samples,), fault_label, dtype=np.int32)
        
        # 生成故障样本
        print("Generating samples...")
        generated_samples = self.gan.generate_samples(
            original_samples,
            target_labels,
            generator='g_AB'
        )
        
        print(f"✓ Generated {num_samples} fault samples (label {fault_label})")
        
        # 保存
        if save_path:
            np.savez(save_path,
                    original=original_samples,
                    generated=generated_samples,
                    labels=target_labels)
            print(f"✓ Saved to: {save_path}")
        
        return original_samples, generated_samples
    
    def generate_healthy_samples(self, num_samples=10, fault_label=1, save_path=None):
        """
        使用 g_BA 从故障样本生成健康样本
        
        Args:
            num_samples: 要生成的样本数量
            fault_label: 源故障类别 (0-8)
            save_path: 保存路径（可选）
        
        Returns:
            original_samples: 原始故障样本
            generated_samples: 生成的健康样本
        """
        print(f"\n{'='*80}")
        print(f"Generating Healthy Samples using g_BA")
        print(f"{'='*80}")
        print(f"Number of samples: {num_samples}")
        print(f"Source fault label: {fault_label}")
        
        # 选择特定故障类别的样本
        fault_indices = np.where(self.domain_B_train_Y.flatten() == fault_label)[0]
        
        if len(fault_indices) < num_samples:
            print(f"Warning: Only {len(fault_indices)} samples available for fault label {fault_label}")
            num_samples = len(fault_indices)
        
        selected_indices = np.random.choice(fault_indices, num_samples, replace=False)
        original_samples = self.domain_B_train_X[selected_indices]
        
        # 创建目标标签（健康类别为0）
        target_labels = np.zeros((num_samples,), dtype=np.int32)
        
        # 生成健康样本
        print("Generating samples...")
        generated_samples = self.gan.generate_samples(
            original_samples,
            target_labels,
            generator='g_BA'
        )
        
        print(f"✓ Generated {num_samples} healthy samples from fault label {fault_label}")
        
        # 保存
        if save_path:
            np.savez(save_path,
                    original=original_samples,
                    generated=generated_samples,
                    source_labels=np.full((num_samples,), fault_label))
            print(f"✓ Saved to: {save_path}")
        
        return original_samples, generated_samples
    
    def evaluate_with_svm(self, add_quantity=995):
        """
        使用 SVM 评估模型性能
        
        Args:
            add_quantity: 每个故障类别生成的样本数量
        
        Returns:
            accuracy: 分类准确率
        """
        print(f"\n{'='*80}")
        print(f"Evaluating Model with SVM")
        print(f"{'='*80}")
        print(f"Generating {add_quantity} samples per fault class (9 classes total)")
        print(f"Total generated samples: {add_quantity * 9}")


        
        # 调用 SVM 评估函数
        accuracy = samlpe_generation_feed_svm(
            add_quantity=add_quantity,
            test_x=self.test_X,
            test_y=self.test_Y,
            generator=self.gan.g_AB,
            domain_A_train_x=self.domain_A_train_X,
            domain_A_train_y=self.domain_A_train_Y,
            domain_B_train_x=self.domain_B_train_X,
            domain_B_train_y=self.domain_B_train_Y,
            c = 0.2,
            g = 0.001
        )
        
        print(f"\n{'='*80}")
        print(f"SVM Classification Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'='*80}\n")
        
        return accuracy
    
    def visualize_generation(self, original, generated, num_samples=5, 
                            title="Sample Generation", save_path=None):
        """
        可视化原始样本和生成样本
        
        Args:
            original: 原始样本
            generated: 生成样本
            num_samples: 要显示的样本数量
            title: 图表标题
            save_path: 保存路径（可选）
        """
        print(f"\nVisualizing {num_samples} samples...")
        
        num_samples = min(num_samples, len(original), len(generated))
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # 原始样本
            axes[i, 0].plot(original[i], linewidth=0.8, color='blue')
            axes[i, 0].set_title(f'Original Sample {i+1}')
            axes[i, 0].set_xlabel('Time Step')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 生成样本
            axes[i, 1].plot(generated[i], linewidth=0.8, color='red')
            axes[i, 1].set_title(f'Generated Sample {i+1}')
            axes[i, 1].set_xlabel('Time Step')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def batch_generate_all_faults(self, num_samples_per_class=100, save_dir='./generated_samples'):
        """
        批量生成所有9种故障类型的样本
        
        Args:
            num_samples_per_class: 每个故障类别生成的样本数量
            save_dir: 保存目录
        
        Returns:
            all_generated: 所有生成的样本
            all_labels: 对应的标签
        """
        print(f"\n{'='*80}")
        print(f"Batch Generating All Fault Types")
        print(f"{'='*80}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        all_generated = []
        all_labels = []
        
        for fault_label in range(9):
            print(f"\nGenerating fault class {fault_label}...")
            
            # 选择健康样本
            indices = np.random.choice(
                len(self.domain_A_train_X), 
                num_samples_per_class, 
                replace=False
            )
            original = self.domain_A_train_X[indices]
            
            # 生成故障样本
            target_labels = np.full((num_samples_per_class,), fault_label, dtype=np.int32)
            generated = self.gan.generate_samples(original, target_labels, generator='g_AB')
            
            all_generated.append(generated)
            all_labels.append(target_labels)
            
            # 保存每个类别
            save_path = os.path.join(save_dir, f'fault_class_{fault_label}.npz')
            np.savez(save_path, 
                    original=original, 
                    generated=generated, 
                    labels=target_labels)
            
            print(f"  ✓ Generated {num_samples_per_class} samples for class {fault_label}")
            print(f"  ✓ Saved to: {save_path}")
        
        # 合并所有样本
        all_generated = np.concatenate(all_generated, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 保存合并的文件
        combined_path = os.path.join(save_dir, 'all_fault_samples.npz')
        np.savez(combined_path, generated=all_generated, labels=all_labels)
        
        print(f"\n✓ Total generated samples: {len(all_generated)}")
        print(f"✓ Combined file saved to: {combined_path}")
        print(f"{'='*80}\n")
        
        return all_generated, all_labels


def main():
    """主函数 - 使用示例"""
    
    print("="*80)
    print(" "*25 + "CycleGAN Model Testing")
    print("="*80)
    
    # ==================== 配置 ====================
    
    # 模型和数据路径
    MODEL_PATH = './saved_models/best_cyclegan_model.pth'  # 修改为你的模型路径
    DATA_PATH = './dataset/dataset_fft_for_cyclegan_case1_512.npz'  # 修改为你的数据路径
    
    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH with your actual model file location.")
        return
    
    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Data file not found: {DATA_PATH}")
        print("Please update DATA_PATH with your actual data file location.")
        return
    
    # ==================== 初始化测试器 ====================
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = ModelTester(MODEL_PATH, DATA_PATH, device=device)
    
    # ==================== 功能演示 ====================
    
    print("\n" + "="*80)
    print("Available Functions:")
    print("="*80)
    print("1. Generate fault samples (g_AB)")
    print("2. Generate healthy samples (g_BA)")
    print("3. Evaluate with SVM")
    print("4. Batch generate all fault types")
    print("5. Exit")
    print("="*80)
    
    while True:
        choice = input("\nSelect function (1-5): ").strip()
        
        if choice == '1':
            # 生成故障样本
            print("\n" + "-"*80)
            num_samples = int(input("Number of samples to generate [10]: ") or "10")
            fault_label = int(input("Target fault label (0-8) [1]: ") or "1")
            visualize = input("Visualize results? (y/n) [y]: ").strip().lower() or 'y'
            
            original, generated = tester.generate_fault_samples(
                num_samples=num_samples,
                fault_label=fault_label,
                save_path=f'./fault_samples_label_{fault_label}.npz'
            )
            
            if visualize == 'y':
                tester.visualize_generation(
                    original, generated,
                    num_samples=min(5, num_samples),
                    title=f"g_AB: Healthy → Fault (Label {fault_label})",
                    save_path=f'./visualization_g_AB_label_{fault_label}.png'
                )
        
        elif choice == '2':
            # 生成健康样本
            print("\n" + "-"*80)
            num_samples = int(input("Number of samples to generate [10]: ") or "10")
            fault_label = int(input("Source fault label (0-8) [1]: ") or "1")
            visualize = input("Visualize results? (y/n) [y]: ").strip().lower() or 'y'
            
            original, generated = tester.generate_healthy_samples(
                num_samples=num_samples,
                fault_label=fault_label,
                save_path=f'./healthy_samples_from_fault_{fault_label}.npz'
            )
            
            if visualize == 'y':
                tester.visualize_generation(
                    original, generated,
                    num_samples=min(5, num_samples),
                    title=f"g_BA: Fault (Label {fault_label}) → Healthy",
                    save_path=f'./visualization_g_BA_from_{fault_label}.png'
                )
        
        elif choice == '3':
            # SVM 评估
            print("\n" + "-"*80)
            add_quantity = int(input("Samples per fault class: ") or "995")
            
            accuracy = tester.evaluate_with_svm(add_quantity=add_quantity)
            
            print(f"\n✓ Evaluation completed!")
            print(f"  Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        elif choice == '4':
            # 批量生成所有故障类型
            print("\n" + "-"*80)
            num_per_class = int(input("Samples per class [100]: ") or "100")
            save_dir = input("Save directory [./generated_samples]: ").strip() or './generated_samples'
            
            all_generated, all_labels = tester.batch_generate_all_faults(
                num_samples_per_class=num_per_class,
                save_dir=save_dir
            )
            
            print(f"\n✓ Batch generation completed!")
            print(f"  Total samples: {len(all_generated)}")
            print(f"  Saved to: {save_dir}")
        
        elif choice == '5':
            print("\nExiting...")
            break
        
        else:
            print("\n[ERROR] Invalid choice. Please select 1-5.")
    
    print("\n" + "="*80)
    print("Testing completed!")
    print("="*80)


# ==================== 快速使用示例 ====================

def quick_example():
    """
    快速使用示例 - 不需要交互
    """
    print("\n" + "="*80)
    print("Quick Example - Non-interactive Mode")
    print("="*80)
    
    # 配置路径
    MODEL_PATH = './saved_models/best_model.pth'
    DATA_PATH = './dataset_example.npz'
    
    # 初始化
    tester = ModelTester(MODEL_PATH, DATA_PATH)
    
    # 示例1: 生成故障样本
    print("\n[Example 1] Generating fault samples...")
    original, generated = tester.generate_fault_samples(
        num_samples=10,
        fault_label=1,
        save_path='./example_fault_samples.npz'
    )
    
    # 示例2: 生成健康样本
    print("\n[Example 2] Generating healthy samples...")
    original, generated = tester.generate_healthy_samples(
        num_samples=10,
        fault_label=2,
        save_path='./example_healthy_samples.npz'
    )
    
    # 示例3: SVM 评估
    print("\n[Example 3] Evaluating with SVM...")
    accuracy = tester.evaluate_with_svm(add_quantity=995)
    
    print(f"\n{'='*80}")
    print("Quick example completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    # 交互式模式
    main()
    
    # 或者使用快速示例（非交互）
    # quick_example()
