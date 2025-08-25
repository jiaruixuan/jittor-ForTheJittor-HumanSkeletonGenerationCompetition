import jittor as jt
from jittor import nn
from jittor import concat, sqrt
import math

DEBUG_POINT_TRANSFORMER = False

def ensure_bn3(pos):
    """
    确保 pos 为 [B,N,3] 形状。
    支持输入 [B,3,N] 或 [B,N,3]；否则报错。
    """
    assert pos.ndim == 3, f"pos ndim 应为3, got {pos.shape}"
    B, A, B_ = pos.shape
    if pos.shape[-1] == 3:
        return pos  # [B,N,3]
    if pos.shape[1] == 3 and pos.shape[-1] != 3:
        # 认为是 [B,3,N]
        return pos.transpose(1, 2)
    raise ValueError(f"无法解析的 pos 形状: {pos.shape}, 期望 [B,N,3] 或 [B,3,N]")

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.main_path = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.shortcut = nn.Identity()
        self.act = nn.GELU()
        
    def execute(self, x):
        return self.act(self.main_path(x) + self.shortcut(x))

class InceptionBlock(nn.Module):
    """Inception风格模块"""
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock, self).__init__()
        mid_dim = out_dim // 4
        
        # 多分支结构
        self.branch1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU()
        )
        
        self.branch4 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU()
        )
        
        # 合并层
        self.combine = nn.Sequential(
            nn.Linear(mid_dim * 4, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        
    def execute(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # 拼接各分支输出
        concat = jt.concat([b1, b2, b3, b4], dim=1)
        return self.combine(concat)

class InceptionResidualBlock(nn.Module):
    """结合Inception和残差连接的模块"""
    def __init__(self, in_dim, out_dim):
        super(InceptionResidualBlock, self).__init__()
        self.inception = InceptionBlock(in_dim, out_dim)
        self.shortcut = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        ) if in_dim != out_dim else nn.Identity()
        self.act = nn.GELU()
        
    def execute(self, x):
        return self.act(self.inception(x) + self.shortcut(x))

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super(MLP, self).__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class ImprovedAttention(nn.Module):
    """修复的多头自注意力模块"""
    def __init__(self, dim=96, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super(ImprovedAttention, self).__init__()
        
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 分别定义Q、K、V线性层，更灵活
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def execute(self, x):
        B, N, C = x.shape
        
        # 分别计算Q、K、V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 数值稳定性处理
        attn = jt.clamp(attn, min=-50, max=50)
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力并重组
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class DropPath(nn.Module):
    """DropPath (随机深度)"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = float(drop_prob)
    
    def execute(self, x):
        if self.drop_prob == 0.0 or not self.is_training():
            return x
            
        keep_prob = 1.0 - self.drop_prob
        batch_size = x.shape[0]
        
        random_tensor = jt.rand((batch_size, 1, 1)) < keep_prob
        random_tensor = random_tensor.astype(x.dtype)
        
        output = x * random_tensor / keep_prob
        
        return output

class SimpleTransformerBlock(nn.Module):
    """简化版Transformer块"""
    def __init__(self, dim=96, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super(SimpleTransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        # 使用改进的注意力模块
        self.attn = ImprovedAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # 位置编码
        self.pos_embed = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def execute(self, x, points):
        # 计算位置编码
        pos = self.pos_embed(points)
        
        # 自注意力层
        x = x + self.drop_path(self.attn(self.norm1(x + pos)))
        
        # MLP层
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

# KNN和Point Transformer相关函数
def knn(x, k):
    """
    计算KNN邻域
    Args:
        x: [B, N, 3] 点坐标
        k: 邻居数量
    Returns:
        idx: [B, N, k] 邻居索引
    """
    inner = -2 * jt.matmul(x.transpose(1, 2), x)
    xx = jt.sum(x**2, dim=1, keepdims=True)
    pairwise_distance = -xx - inner - xx.transpose(1, 2)
    
    _, idx = jt.topk(pairwise_distance, k=k, dim=-1)
    return idx

def safe_knn(pos, k):
    pos = ensure_bn3(pos)
    B, N, _ = pos.shape
    K = min(k, N)
    inner = -2 * jt.matmul(pos.transpose(1,2), pos)          # [B,3,N] @ [B,N,3] -> [B,3,3]? (修正)
    # 修正：应使用 pairwise 距离公式；重写更安全
    # 重新计算距离矩阵 (B,N,N)
    xx = jt.sum(pos**2, dim=2, keepdims=True)                # [B,N,1]
    pairwise = xx + xx.transpose(1,2) - 2*jt.matmul(pos, pos.transpose(1,2))  # [B,N,N]
    # 取最邻近（用负号调 jt.topk）
    _, idx = jt.topk(-pairwise, K, dim=-1)                   # [B,N,K]
    return idx.stop_grad()

def index_points(points, idx):
    """
    points: [B,N,C]
    idx: [B,S,K] 或 [B,S]
    返回:
      [B,S,K,C] 或 [B,S,C]
    """
    points = points  # [B,N,C]
    assert points.ndim == 3, f"points shape 错误 {points.shape}"
    B, N, C = points.shape
    if idx.ndim == 2:
        B2, S = idx.shape
        assert B2 == B
        flat = points.reshape(B*N, C)
        offset = (jt.arange(B) * N).reshape(B,1)
        gather_idx = (idx + offset).reshape(-1)
        out = flat[gather_idx].reshape(B, S, C)
        return out
    elif idx.ndim == 3:
        B2, S, K = idx.shape
        assert B2 == B
        flat = points.reshape(B*N, C)
        offset = (jt.arange(B) * N).reshape(B,1,1)
        gather_idx = (idx + offset).reshape(-1)
        out = flat[gather_idx].reshape(B, S, K, C)
        return out
    else:
        raise ValueError(f"idx 维度不支持: {idx.shape}")

def fps(points, num_samples):
    """
    最远点采样 (Farthest Point Sampling)
    """
    B, N, _ = points.shape
    
    indices = jt.zeros((B, num_samples), dtype='int32')
    distances = jt.ones((B, N)) * 1e10
    
    farthest = jt.randint(0, N, (B,))
    
    for i in range(num_samples):
        indices[:, i] = farthest
        centroid = points[jt.arange(B), farthest, :].unsqueeze(1)
        
        dist = jt.sum((points - centroid) ** 2, dim=2)
        
        mask = dist < distances
        distances = jt.where(mask, dist, distances)
        
        farthest, _ = jt.argmax(distances, dim=1)
    
    sampled_points = points[jt.arange(B).unsqueeze(1), indices]
    return sampled_points, indices


def knn_interpolate(target_pos, source_pos, source_feat, k=3):
    """
    target_pos: [B,M,3]
    source_pos: [B,N,3]
    source_feat:[B,N,C]
    返回: [B,M,C]
    """
    B,M,_ = target_pos.shape
    B2,N,_ = source_pos.shape
    assert B==B2
    K = min(k, N)
    # 计算距离
    # (B,M,N)
    diff = target_pos.unsqueeze(2) - source_pos.unsqueeze(1)
    dist2 = (diff*diff).sum(dim=3)
    # 取最小 K
    neg_dist2, nn_idx = jt.topk(-dist2, K, dim=2)  # nn_idx [B,M,K]
    nn_idx = nn_idx.stop_grad()
    gathered_feat = index_points(source_feat, nn_idx)  # [B,M,K,C]
    w = jt.clamp(-neg_dist2, 1e-8, 1e10)              # 还原正距离平方
    inv = 1.0 / (w + 1e-8)                            # [B,M,K]
    inv_sum = inv.sum(dim=2, keepdims=True)
    weight = inv / inv_sum
    weight = weight.unsqueeze(-1)                     # [B,M,K,1]
    out = (gathered_feat * weight).sum(dim=2)         # [B,M,C]
    return out

class PointTransformerAttention(nn.Module):
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.dim = dim
        self.q_conv = nn.Linear(dim, dim)
        self.k_conv = nn.Linear(dim, dim)
        self.v_conv = nn.Linear(dim, dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, 1)
        )
        self.gamma = jt.zeros(1)

    def execute(self, x, pos):
        # 1. 形状校正
        pos = ensure_bn3(pos)                 # [B,N,3]
        B, N, C = x.shape
        assert pos.shape[0] == B and pos.shape[1] == N, f"x:{x.shape} pos:{pos.shape}"
        # 2. KNN
        idx = safe_knn(pos, self.k)           # [B,N,K]
        K = idx.shape[-1]
        # 3. 邻域 gather
        neighbor_x = index_points(x, idx)     # [B,N,K,C]
        neighbor_pos = index_points(pos, idx) # [B,N,K,3]
        if DEBUG_POINT_TRANSFORMER and jt.rank==0:
            print(f"[DEBUG] x:{x.shape} pos:{pos.shape} idx:{idx.shape} neighbor_x:{neighbor_x.shape} neighbor_pos:{neighbor_pos.shape}")
        # 4. 线性投影
        q = self.q_conv(x).unsqueeze(2)       # [B,N,1,C]
        k = self.k_conv(neighbor_x)           # [B,N,K,C]
        v = self.v_conv(neighbor_x)           # [B,N,K,C]
        # 5. 相对位置
        rel_pos = neighbor_pos - pos.unsqueeze(2)  # [B,N,K,3]
        scale = jt.clamp(jt.norm(rel_pos, dim=3, keepdims=True).mean(), 1e-4, 1e4)
        rel_pos_norm = rel_pos / (scale + 1e-8)
        pos_enc = self.pos_mlp(rel_pos_norm)  # [B,N,K,C]
        # 6. 注意力
        attn_logits = self.attn_mlp(q - k + pos_enc).squeeze(-1)  # [B,N,K]
        attn = nn.softmax(attn_logits, dim=-1).unsqueeze(-1)      # [B,N,K,1]
        out = jt.sum(attn * (v + pos_enc), dim=2)                 # [B,N,C]
        return x + self.gamma * out
    

class PointTransformerBlock(nn.Module):
    """Point Transformer块"""
    def __init__(self, dim, k=16, mlp_ratio=4):
        super().__init__()
        self.attn = PointTransformerAttention(dim, k)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def execute(self, x, pos):
        # 注意力 + 残差
        x = x + self.attn(self.norm1(x), pos)
        
        # MLP + 残差
        x = x + self.mlp(self.norm2(x))
        
        return x

class MultiScalePointTransformer(nn.Module):
    """多尺度Point Transformer（动态适配 + 缓存 + 融合门控）"""
    def __init__(self, in_channels=3, embed_dim=128, num_stages=3,
                 stage_points=(2048,1024,512), k=16, share_knn=False):
        super().__init__()
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.share_knn = share_knn
        self.base_k = k

        self.stem = nn.Sequential(
            nn.Linear(in_channels, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # 维度配置（递增 *2）
        dims = [embed_dim, embed_dim*2, embed_dim*4][:num_stages]
        self.stage_points_cfg = stage_points[:num_stages]

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, dim in enumerate(dims):
            self.encoders.append(PointTransformerBlock(dim, k=self.base_k))
            if i > 0:
                self.downsamples.append(nn.Linear(dims[i-1], dim))

        # 解码
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.fuse_gates = nn.ModuleList()  # 门控融合
        for i in range(len(dims)-2, -1, -1):
            self.upsamples.append(nn.Linear(dims[i+1]+dims[i], dims[i]))
            self.decoders.append(PointTransformerBlock(dims[i], k=self.base_k))
            self.fuse_gates.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i]),
                    nn.Sigmoid()
                )
            )

    def _safe_fps(self, points, target_num):
        """若 target 超过现有点数则直接返回全部"""
        B, N, _ = points.shape
        target = min(N, target_num)
        if target == N:
            idx = jt.arange(N).unsqueeze(0).repeat(B,1)
            return points, idx
        return fps(points, target)

    def execute(self, points):
        B, N, _ = points.shape
        features = self.stem(points)

        encoder_features = []
        encoder_points = []
        current_points = points
        current_features = features

        knn_cache = None

        # 编码
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                sampled_points, indices = self._safe_fps(current_points, self.stage_points_cfg[i])
                batch_indices = jt.arange(B).unsqueeze(1).repeat(1, sampled_points.shape[1])
                sampled_features = current_features[batch_indices, indices]
                sampled_features = self.downsamples[i-1](sampled_features)
                current_points = sampled_points
                current_features = sampled_features

            if self.share_knn:
                # 可选：共享 knn 索引
                if knn_cache is None:
                    knn_cache = knn(current_points, self.base_k)
                current_features = encoder.attn(self.encoders[i].norm1(current_features), current_points) + current_features
                current_features = current_features + self.encoders[i].mlp(self.encoders[i].norm2(current_features))
            else:
                current_features = encoder(current_features, current_points)

            encoder_features.append(current_features)
            encoder_points.append(current_points)

        # 解码
        multi_scale_features = [encoder_features[-1]]
        current_features = encoder_features[-1]
        current_points = encoder_points[-1]

        for i, (decoder, upsample, gate) in enumerate(zip(self.decoders, self.upsamples, self.fuse_gates)):
            stage_idx = len(encoder_features) - 2 - i
            target_points = encoder_points[stage_idx]
            target_features = encoder_features[stage_idx]

            upsampled = knn_interpolate(target_points, current_points, current_features, k=3)
            fused = jt.concat([target_features, upsampled], dim=2)
            fused = upsample(fused)

            # 门控（Residual Gate）
            g = gate(target_features)  # [B, M, C]
            fused = target_features + g * (fused - target_features)

            current_features = decoder(fused, target_points)
            current_points = target_points
            multi_scale_features.append(current_features)

        return current_features, multi_scale_features

class EnhancedMultiScaleBonePredictor(nn.Module):
    """增强的多尺度骨骼预测器"""
    def __init__(self, feat_dims=[512, 256, 128], num_joints=52):
        super().__init__()
        self.feat_dims = feat_dims
        self.num_joints = num_joints
        
        # 关节分组
        self.main_joints_count = 22
        self.left_hand_joints_count = 15
        self.right_hand_joints_count = 15
        
        # 多尺度特征融合
        self.fusion = nn.Sequential(
            nn.Linear(sum(feat_dims), feat_dims[-1]),
            nn.ReLU(),
            nn.Linear(feat_dims[-1], feat_dims[-1])
        )
        
        # 预测头
        self.main_bone_head = self._build_main_bone_head(feat_dims[-1])
        self.left_hand_head = self._build_hand_head(feat_dims[-1])
        self.right_hand_head = self._build_hand_head(feat_dims[-1])
    
    def _build_main_bone_head(self, feat_dim):
        """构建主要骨骼预测头"""
        return nn.Sequential(
            InceptionResidualBlock(feat_dim, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            
            InceptionResidualBlock(512, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            nn.Linear(256, self.main_joints_count * 3)
        )
    
    def _build_hand_head(self, feat_dim):
        """构建单侧手部预测头"""
        return nn.Sequential(
            nn.Linear(feat_dim + 3, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            
            InceptionResidualBlock(256, 256),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 15 * 3)
        )
    
    def execute(self, multi_scale_features):
        # 全局池化所有尺度特征
        pooled_features = []
        for features in multi_scale_features:
            pooled = features.mean(dim=1)
            pooled_features.append(pooled)
        
        # 融合多尺度特征
        fused_features = jt.concat(pooled_features, dim=1)
        global_features = self.fusion(fused_features)
        
        # 骨骼预测
        main_joints = self.main_bone_head(global_features)
        main_joints_3d = main_joints.reshape(-1, 22, 3)
        
        left_wrist = main_joints_3d[:, 9, :]
        right_wrist = main_joints_3d[:, 13, :]
        
        left_hand_input = jt.concat([global_features, left_wrist], dim=1)
        left_hand_joints = self.left_hand_head(left_hand_input)
        
        right_hand_input = jt.concat([global_features, right_wrist], dim=1)
        right_hand_joints = self.right_hand_head(right_hand_input)
        
        full_joints = jt.concat([main_joints, left_hand_joints, right_hand_joints], dim=1)
        hand_joints = jt.concat([left_hand_joints, right_hand_joints], dim=1)
        
        return full_joints, main_joints, hand_joints

class SkinningDecoder(nn.Module):
    def __init__(self, embed_dim=128, num_joints=52, topk=8, group_norm=True):
        super().__init__()
        self.num_joints = num_joints
        self.topk = topk
        self.group_norm = group_norm
        # 身体(22) 左手(15) 右手(15) 分组
        self.groups = [(0,22),(22,37),(37,52)] if num_joints==52 else [(0,num_joints)]

        self.vertex_proj = nn.Linear(embed_dim, embed_dim)
        self.joint_proj  = nn.Linear(embed_dim, embed_dim)
        self.dist_embed  = nn.Sequential(
            nn.Linear(topk, embed_dim//2),
            nn.GELU(),
            nn.Linear(embed_dim//2, embed_dim//2)
        )
        self.dir_embed = nn.Sequential(
            nn.Linear(3*topk, embed_dim//2),
            nn.GELU(),
            nn.Linear(embed_dim//2, embed_dim//2)
        )
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim*2 + embed_dim//2 + embed_dim//2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_joints)
        )

    def execute(self, vertex_features, joint_features, vertices, joints):
        B,N,_ = vertex_features.shape
        J = joints.shape[1]

        q = self.vertex_proj(vertex_features)
        k = self.joint_proj(joint_features)
        attn = jt.matmul(q, k.transpose(1,2)) / math.sqrt(q.shape[-1])
        attn = nn.softmax(attn, dim=-1)
        joint_attn_feat = jt.matmul(attn, joint_features)

        # 距离与方向
        rel = joints.unsqueeze(1) - vertices.unsqueeze(2)   # [B,N,J,3]
        dist = jt.norm(rel, dim=3)                          # [B,N,J]
        topk = min(self.topk, J)
        nd, idx = jt.topk(-dist, k=topk, dim=2)
        nd = -nd                                            # [B,N,k]
        gather_rel = rel[jt.arange(B).unsqueeze(1).unsqueeze(2), jt.arange(N).unsqueeze(0).unsqueeze(2), idx]  # [B,N,k,3]
        # 方向单位化
        dir_unit = gather_rel / (jt.norm(gather_rel, dim=3, keepdims=True)+1e-8)
        dir_feat = dir_unit.reshape(B,N,topk*3)
        dir_feat = self.dir_embed(dir_feat)
        dist_feat = self.dist_embed(nd)

        x = jt.concat([vertex_features, joint_attn_feat, dist_feat, dir_feat], dim=2)
        weights = self.fuse(x)
        weights = nn.softmax(weights, dim=-1)

        if self.group_norm and len(self.groups)>1:
            # 分组归一化，再拼回（避免手部被身体关节权重淹没）
            ws = []
            for (s,e) in self.groups:
                g_w = weights[:, :, s:e]
                g_w = g_w / (g_w.sum(dim=2, keepdims=True)+1e-8)
                ws.append(g_w)
            weights = jt.concat(ws, dim=2)
        return weights

class JointEncoder(nn.Module):
    """关节特征编码器"""
    def __init__(self, in_channels=3, embed_dim=128):
        super(JointEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def execute(self, joints):
        return self.encoder(joints)

class MultiScaleCombinedModel(nn.Module):
    """整合的多尺度联合模型"""
    def __init__(self, feat_dim=128, num_joints=52, use_point_transformer=True):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.use_point_transformer = use_point_transformer
        
        # 选择编码器类型
        if use_point_transformer:
            self.point_encoder = MultiScalePointTransformer(
                in_channels=3,
                embed_dim=feat_dim,
                num_stages=3
            )
            # 使用多尺度骨骼预测器
            self.skeleton_predictor = EnhancedMultiScaleBonePredictor(
                feat_dims=[feat_dim * 4, feat_dim * 2, feat_dim],  # 对应多尺度特征维度
                num_joints=num_joints
            )
        else:
            # 原始编码器
            self.point_encoder = PointEncoder(
                in_channels=3,
                embed_dim=feat_dim,
                depth=4,
                drop_rate=0.1,
                attn_drop_rate=0.0,
                drop_path_rate=0.1
            )
            # 原始骨骼预测器
            self.skeleton_predictor = MultiScaleBonePredictor(
                feat_dim=feat_dim, 
                num_joints=num_joints
            )
        
        # 其他组件
        self.skin_decoder = SkinningDecoder(
            embed_dim=feat_dim,
            num_joints=num_joints
        )
        
        self.joint_encoder = JointEncoder(
            in_channels=3,
            embed_dim=feat_dim
        )
        
        self.global_attn = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )
    
    def execute(self, vertices):
        """
        执行联合预测
        """
        B, N, _ = vertices.shape
        
        # 1. 点云特征编码
        if self.use_point_transformer:
            vertex_features, multi_scale_features = self.point_encoder(vertices)
            # 多尺度骨骼预测
            full_joints_flat, main_joints_flat, hand_joints_flat = self.skeleton_predictor(multi_scale_features)
        else:
            vertex_features = self.point_encoder(vertices)
            # 全局特征提取
            weights = self.global_attn(vertex_features)
            global_features = (vertex_features * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)
            # 单尺度骨骼预测
            full_joints_flat, main_joints_flat, hand_joints_flat = self.skeleton_predictor(global_features)
        
        # 2. 重塑为3D坐标
        predicted_joints = full_joints_flat.reshape(B, self.num_joints, 3)
        
        # 3. 关节特征编码（用于蒙皮预测）
        joint_features = self.joint_encoder(predicted_joints)
        
        # 4. 预测蒙皮权重
        predicted_skin = self.skin_decoder(
            vertex_features, 
            joint_features, 
            vertices, 
            predicted_joints
        )
        
        return full_joints_flat, predicted_skin, main_joints_flat, hand_joints_flat

# 保留原有的类作为向后兼容
class MultiScaleBonePredictor(nn.Module):
    """原始的多尺度骨骼预测器（向后兼容）"""
    def __init__(self, feat_dim=128, num_joints=52):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        # 根据format.py定义关节分组
        self.main_joints_count = 22  # 关节 0-21
        self.left_hand_joints_count = 15  # 关节 22-36
        self.right_hand_joints_count = 15  # 关节 37-51
        
        # 主要骨骼预测头
        self.main_bone_head = self._build_main_bone_head()
        
        # 左手预测头
        self.left_hand_head = self._build_hand_head()
        
        # 右手预测头  
        self.right_hand_head = self._build_hand_head()
        
    def _build_main_bone_head(self):
        """构建主要骨骼预测头"""
        return nn.Sequential(
            InceptionResidualBlock(self.feat_dim, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            
            InceptionResidualBlock(512, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            nn.Linear(256, self.main_joints_count * 3)  # 22个主要关节
        )
    
    def _build_hand_head(self):
        """构建单侧手部预测头"""
        return nn.Sequential(
            # 输入：全局特征 + 对应手腕关节位置
            nn.Linear(self.feat_dim + 3, 256),  # feat_dim + 手腕位置(3D)
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            
            InceptionResidualBlock(256, 256),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 15 * 3)  # 15个手部关节
        )
    
    def execute(self, global_features):
        """
        Args:
            global_features: [B, feat_dim] 全局特征
        Returns:
            full_joints: [B, 52*3] 完整关节 (正确顺序)
            main_joints: [B, 22*3] 主要关节
            hand_joints: [B, 30*3] 手部关节 (左手15 + 右手15)
        """
        # 1. 主要骨骼预测 (关节 0-21)
        main_joints = self.main_bone_head(global_features)  # [B, 22*3]
        main_joints_3d = main_joints.reshape(-1, 22, 3)
        
        # 2. 提取手腕关节位置
        left_wrist = main_joints_3d[:, 9, :]   # 左手腕 (关节9)
        right_wrist = main_joints_3d[:, 13, :] # 右手腕 (关节13)
        
        # 3. 左手预测 (关节 22-36)
        left_hand_input = jt.concat([global_features, left_wrist], dim=1)
        left_hand_joints = self.left_hand_head(left_hand_input)  # [B, 15*3]
        
        # 4. 右手预测 (关节 37-51)
        right_hand_input = jt.concat([global_features, right_wrist], dim=1)
        right_hand_joints = self.right_hand_head(right_hand_input)  # [B, 15*3]
        
        # 5. 按正确顺序组合完整骨骼
        full_joints = jt.concat([
            main_joints,        # 关节 0-21  (22*3)
            left_hand_joints,   # 关节 22-36 (15*3)
            right_hand_joints   # 关节 37-51 (15*3)
        ], dim=1)  # [B, 52*3]
        
        # 6. 组合手部关节用于损失计算
        hand_joints = jt.concat([left_hand_joints, right_hand_joints], dim=1)  # [B, 30*3]
        
        return full_joints, main_joints, hand_joints

class PointEncoder(nn.Module):
    """原始顶点特征编码器（向后兼容）"""
    def __init__(self, in_channels=3, embed_dim=128, depth=4, drop_rate=0.1, 
                 attn_drop_rate=0.0, drop_path_rate=0.1):
        super(PointEncoder, self).__init__()
        
        # 初始特征提取
        self.stem = nn.Sequential(
            nn.Linear(in_channels, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 渐进式增加Dropout率
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]
        
        # 使用可变数量的头
        num_heads = [4, 8, 8, 8] if depth == 4 else [4, 8, 8, 8, 16, 16]
        
        # 使用简化版Transformer块，避免Z-order问题
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SimpleTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads[i] if i < len(num_heads) else 8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i]
                )
            )
            
    def execute(self, points):
        # 初始特征提取
        features = self.stem(points)
        
        # 应用Transformer块
        for block in self.blocks:
            features = block(features, points)
            
        return features

class ScaleFusionModule(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(feat_dim, num_heads=8)
        self.norm = nn.LayerNorm(feat_dim)
        
    def execute(self, global_feat, local_feat):
        # 使用注意力机制融合全局和局部特征
        fused_feat, _ = self.attention(local_feat, global_feat, global_feat)
        return self.norm(fused_feat + local_feat)