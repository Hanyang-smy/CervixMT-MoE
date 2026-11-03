class PositionModulatedExpert(nn.Module):
    def __init__(self, in_features, hidden_dim, pos_dim=32):
        super().__init__()
        self.pos_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, pos_dim),
            nn.GELU()
        )
        self.gamma_net = nn.Linear(pos_dim, in_features)
        self.beta_net = nn.Linear(pos_dim, in_features)
        self.feature_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_features)
        )

    def forward(self, x, pos_onehot):
        pos_feat = self.pos_net(pos_onehot)
        gamma = 1 + torch.tanh(self.gamma_net(pos_feat))
        beta = torch.tanh(self.beta_net(pos_feat))
        modulated_x = x * gamma + beta
        return self.feature_net(modulated_x)

class PositionAwareMoE(nn.Module):
    def __init__(self, in_features, hidden_dim, num_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([
            PositionModulatedExpert(in_features, hidden_dim)
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.BatchNorm1d(in_features),      
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.73))

    def forward(self, x, pos_onehot):
        weights = torch.softmax(self.gate(x), dim=-1)
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x, pos_onehot)
            expert_outputs.append(expert_out)
        expert_stack = torch.stack(expert_outputs, dim=1)
        moe_out = torch.sum(weights.unsqueeze(-1) * expert_stack, dim=1)
        return x + self.res_scale * moe_out

class OptimizedSwinMoE(nn.Module):
    def __init__(self, backbone, in_features, hidden_dim, num_experts=3):
        super().__init__()
        self.backbone = backbone
        self.backbone.head = nn.Identity()
        
        self.moe = PositionAwareMoE(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_experts=num_experts
        )
        self.pos_embedder = nn.Embedding(3, 256)
        self.pos_feature = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(in_features + 256, 1024),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.position_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 3)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 5)
        )

    def forward(self, x, class3_label=None):
        # 1. backbone 特征
        feat = self.backbone(x)  # 现在应该是四维张量 [B, H, W, C]
        
        # 确保是四维张量
        if feat.dim() == 4:
            B, H, W, C = feat.shape
            feat = feat.reshape(B, H*W, C).mean(1)  # [B, C]
        else:
            # 如果已经是二维张量，直接使用
            feat = feat
        
        # 2. 三分类预测
        pos_logits = self.position_head(feat)        # [B, 3]
        
        # 3. 硬路由用的 label
        if class3_label is None:
            # 测试：用预测结果 argmax
            used_label = pos_logits.argmax(1)
        else:
            # 训练：用真实标签
            used_label = class3_label
        
        # 4. MoE + 位置调制
        pos_onehot    = F.one_hot(used_label, 3).float().to(x.device)
        enhanced_feat = self.moe(feat, pos_onehot)   # [B, in_features]
        
        # 5. 位置语义特征
        pos_emb  = self.pos_embedder(used_label)     # [B,256] 
        pos_feat = self.pos_feature(pos_emb)         # [B,256]
        

        
        # 6. 融合、分类
        fused = torch.cat([enhanced_feat, pos_feat], dim=1)  # [B, in+256]
        final = self.fusion(fused)                             # [B,1024]
        main_logits = self.classifier(final)                   # [B,5]
        
        return main_logits, pos_logits
