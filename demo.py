import torch
from models import OptimizedSwinMoE 
import timm

def run_demo():
    print("Initializing Demo for CervixMT-MoE...")
    # 1. 构建模型
    backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
    model = OptimizedSwinMoE(backbone, in_features=backbone.num_features, hidden_dim=backbone.num_features*4, num_experts=3)
    
    # 2. 生成随机输入 (Batch Size=2, Channels=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # 3. 前向传播
    print("Running forward pass...")
    logits, pos_logits = model(dummy_input, class3_label=None)
    
    print(f"Main classification output shape: {logits.shape}")
    print(f"Position classification output shape: {pos_logits.shape}")
    print("✅ Demo ran successfully! The architecture code is complete and functional.")

if __name__ == "__main__":
    run_demo()
