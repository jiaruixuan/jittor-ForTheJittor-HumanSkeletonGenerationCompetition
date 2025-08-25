import jittor as jt
import numpy as np
import os
import argparse
import time
import random
from jittor import nn, optim
import math

# 从您提供的模型文件中导入
from models.combinedv5 import MultiScaleCombinedModel

# 从您的项目中导入其他必要的模块
from dataset.dataset2 import get_dataloader, transform
from dataset.sampler import SamplerMix
from models.metrics import J2J

def batch_j2j(pred, gt):
    """
    向量化 J2J 距离：pred/gt [B, J, 3]
    返回 [B] 每个样本的均值关节欧氏距离
    """
    diff = pred - gt                              # [B,J,3]
    dist = jt.sqrt((diff * diff).sum(dim=2))      # [B,J]
    return dist.mean(dim=1)                       # [B]

def criterion_j2j(joints_a, joints_b):
    """
    修复的批次 J2J 损失函数
    
    Args:
        joints_a: [B, N*3] 预测关节
        joints_b: [B, N*3] 真实关节
    """
    # 确保输入形状正确
    if joints_a.ndim == 2 and joints_a.shape[1] % 3 == 0:
        B = joints_a.shape[0]
        N = joints_a.shape[1] // 3
        joints_a = joints_a.reshape(B, N, 3)
        joints_b = joints_b.reshape(B, N, 3)
    
    return batch_j2j(joints_a, joints_b).mean()

# 骨长正则
BONE_PAIRS_52 = [
    # 根据骨架拓扑填写 (示例，需按真实拓扑调整)
    (0,1),(1,2),(2,3),(3,4),(4,5),
    (3,6),(6,7),(7,8),(8,9),
    (3,10),(10,11),(11,12),(12,13),
    (0,14),(14,15),(15,16),(16,17),
    (0,18),(18,19),(19,20),(20,21),
    (9,22),(22,23),(23,24),
    (9,25),(25,26),(26,27),
    (9,28),(28,29),(29,30),
    (9,31),(31,32),(32,33),
    (9,34),(34,35),(35,36),
    (13,37),(37,38),(38,39),
    (13,40),(40,41),(41,42),
    (13,43),(43,44),(44,45),
    (13,46),(46,47),(47,48),
    (13,49),(49,50),(50,51),
]

def bone_length_regularization(pred_flat, gt_flat, bone_pairs=BONE_PAIRS_52, weight=1.0):
    """
    关节骨长 L1 正则（自动适配关节数量不足的情况）
    pred_flat/gt_flat: [B, J*3] 或 [B,J,3]
    """
    if pred_flat.ndim == 2:
        B = pred_flat.shape[0]
        J = pred_flat.shape[1] // 3
        pred = pred_flat.reshape(B, J, 3)
        gt = gt_flat.reshape(B, J, 3)
    else:
        pred = pred_flat
        gt = gt_flat
        J = pred.shape[1]
    # 过滤超出当前 J 的骨对
    valid_pairs = [(i,j) for (i,j) in bone_pairs if i < J and j < J]
    if not valid_pairs or weight <= 0:
        return jt.zeros(1)
    loss = 0.0
    for (i,j) in valid_pairs:
        pred_len = jt.norm(pred[:, i] - pred[:, j], dim=1)
        gt_len   = jt.norm(gt[:, i] - gt[:, j], dim=1)
        loss += jt.abs(pred_len - gt_len).mean()
    loss = loss / len(valid_pairs)
    return weight * loss

def multi_scale_skeleton_loss(pred_joints_flat, main_joints_flat, hand_joints_flat, gt_joints_flat, epoch):
    """
    修正的多尺度骨骼损失函数
    
    Args:
        pred_joints_flat: [B, 52*3] 完整的预测关节
        main_joints_flat: [B, 22*3] 主体关节预测
        hand_joints_flat: [B, 30*3] 手部关节预测
        gt_joints_flat: [B, 52*3] 完整的真实关节
        epoch: 当前训练轮次
    
    Returns:
        loss_skel_mse: 整体MSE损失
        loss_skel_j2j: 整体J2J损失
        loss_details: 包含详细损失组件的字典
    """
    
    # 重塑真值关节，提取对应部分
    gt_joints_3d = gt_joints_flat.reshape(-1, 52, 3)  # [B, 52, 3]
    
    # 提取对应的真值部分
    gt_main = gt_joints_3d[:, :22, :].reshape(-1, 22*3)  # 主要关节 (0-21)
    gt_left_hand = gt_joints_3d[:, 22:37, :].reshape(-1, 15*3)  # 左手 (22-36)
    gt_right_hand = gt_joints_3d[:, 37:52, :].reshape(-1, 15*3)  # 右手 (37-51)
    gt_hand = jt.concat([gt_left_hand, gt_right_hand], dim=1)  # [B, 30*3] 合并手部
    
    # 计算各部分的MSE损失
    loss_main_mse = nn.MSELoss()(main_joints_flat, gt_main)
    loss_hand_mse = nn.MSELoss()(hand_joints_flat, gt_hand)
    loss_full_mse = nn.MSELoss()(pred_joints_flat, gt_joints_flat)
    
    # 计算各部分的J2J损失
    loss_main_j2j = criterion_j2j(main_joints_flat, gt_main)
    loss_hand_j2j = criterion_j2j(hand_joints_flat, gt_hand)
    loss_full_j2j = criterion_j2j(pred_joints_flat, gt_joints_flat)
    
    # # 动态权重调整
    # if epoch < 50:  # 前期重点训练主骨骼
    #     main_weight = 2.0
    #     hand_weight = 0.5
    #     full_weight = 1.0
    # elif epoch < 100:  # 中期平衡
    #     main_weight = 1.5
    #     hand_weight = 1.0
    #     full_weight = 1.5
    # else:  # 后期精细化
    #     main_weight = 1.0
    #     hand_weight = 1.5
    #     full_weight = 2.0
    
    main_weight = 0.8
    hand_weight = 2.0
    full_weight = 2.0
    
    # 组合损失 - 使用加权平均
    total_weight = main_weight + hand_weight + full_weight
    loss_skel_mse = (main_weight * loss_main_mse + 
                     hand_weight * loss_hand_mse + 
                     full_weight * loss_full_mse) / total_weight
    
    loss_skel_j2j = (main_weight * loss_main_j2j + 
                     hand_weight * loss_hand_j2j + 
                     full_weight * loss_full_j2j) / total_weight
    
    # 返回详细的损失组件用于日志记录
    loss_details = {
        'main_mse': loss_main_mse.item(),
        'hand_mse': loss_hand_mse.item(),
        'full_mse': loss_full_mse.item(),
        'main_j2j': loss_main_j2j.item(),
        'hand_j2j': loss_hand_j2j.item(),
        'full_j2j': loss_full_j2j.item(),
        'weights': f"main:{main_weight:.1f}, hand:{hand_weight:.1f}, full:{full_weight:.1f}"
    }
    
    return loss_skel_mse, loss_skel_j2j, loss_details

# 设置 Jittor flags
jt.flags.use_cuda = 1

def train(args):
    """联合训练主函数"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    def log_message(message):
        if jt.rank!=0:
            return
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    log_message(f"开始联合训练，参数: {args}")
    
    # 导入新模型
    model = MultiScaleCombinedModel(feat_dim=args.feat_dim, num_joints=args.num_joints)
    if args.pretrained_model:
        log_message(f"加载预训练模型: {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # 创建优化器
    if args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 创建损失函数
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    
    # 创建数据加载器
    sampler = SamplerMix(num_samples=args.num_samples, vertex_samples=args.vertex_samples)
    train_loader = get_dataloader(
        data_root=args.data_root, data_list=args.train_data_list, train=True,
        batch_size=args.batch_size, shuffle=True, sampler=sampler, transform=transform,
        random_pose=args.random_pose,
        use_track_poses=args.use_track_poses,      # 新增
        track_data_root=args.track_data_root, 
    )
    val_loader = None
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root, data_list=args.val_data_list, train=False,
            batch_size=args.batch_size, shuffle=False, sampler=sampler, transform=transform,
            random_pose=True,
            use_track_poses=args.use_track_poses,      # 新增
            track_data_root=args.track_data_root, 
        )
    
    # 训练循环
    best_val_metric = float('inf')
    best_competition_score = -float('inf')
      # 课程学习参数
    warmup_epochs = 30  # 预热期更长
    curriculum_epochs = 50  # 课程学习期
    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = time.time()
        
        skel_alpha = args.skel_alpha
        skel_beta = args.skel_beta
        skeleton_weight = args.skeleton_loss_weight
        skin_weight = args.skin_loss_weight
        for batch_idx, data in enumerate(train_loader):
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']
            
            # 前向传播 - 处理4个返回值
            model_output = model(vertices)
            if len(model_output) == 4:
                # 新模型：多尺度输出
                pred_joints_flat, pred_skin, main_joints_flat, hand_joints_flat = model_output
                use_multiscale = True
            else:
                # 旧模型：兼容性处理
                pred_joints_flat, pred_skin = model_output
                main_joints_flat = hand_joints_flat = None
                use_multiscale = False
            
            # 准备真值
            gt_joints_flat = joints.reshape(pred_joints_flat.shape)
            
            # 选择损失函数
            if use_multiscale:
                loss_skel_mse, loss_skel_j2j, loss_details = multi_scale_skeleton_loss(
                    pred_joints_flat, main_joints_flat, hand_joints_flat, gt_joints_flat, epoch
                )
                bone_reg = bone_length_regularization(
                    pred_joints_flat, gt_joints_flat, weight=args.bone_len_weight
                )
                total_skel_loss = skel_alpha * loss_skel_mse + skel_beta * loss_skel_j2j + bone_reg
                if skin_weight > 0:
                    loss_skin_mse = criterion_mse(pred_skin, skin)
                    loss_skin_l1 = criterion_l1(pred_skin, skin)
                    total_skin_loss = loss_skin_mse + loss_skin_l1
                else:
                    total_skin_loss = 0

                # 总损失
                loss = skeleton_weight * total_skel_loss + skin_weight * total_skin_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                
                # 多尺度详细日志
                if (batch_idx + 1) % args.print_freq == 0 and jt.rank == 0: #只允许第零个进程记录日志
                    log_message(
                        f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | "
                        f"Loss:{loss.item():.4f} | Main_MSE:{loss_details['main_mse']:.4f} "
                        f"| Hand_MSE:{loss_details['hand_mse']:.4f} | Full_MSE:{loss_details['full_mse']:.4f} "
                        f"| Main_J2J:{loss_details['main_j2j']:.4f} | Hand_J2J:{loss_details['hand_j2j']:.4f} "
                        f"| BoneReg:{bone_reg.item():.4f} | A:{skel_alpha:.2f} B:{skel_beta:.2f}"
                    )
            else:
                # 使用原始损失
                loss_skel_mse = criterion_mse(pred_joints_flat, gt_joints_flat)
                loss_skel_j2j = criterion_j2j(pred_joints_flat, gt_joints_flat)
                bone_reg = bone_length_regularization(
                    pred_joints_flat, gt_joints_flat, weight=args.bone_len_weight
                )
                total_skel_loss = skel_alpha * loss_skel_mse + skel_beta * loss_skel_j2j + bone_reg
                if skin_weight > 0:
                    loss_skin_mse = criterion_mse(pred_skin, skin)
                    loss_skin_l1 = criterion_l1(pred_skin, skin)
                    total_skin_loss = loss_skin_mse + loss_skin_l1
                else:
                    total_skin_loss = 0

                # 总损失
                loss = skeleton_weight * total_skel_loss + skin_weight * total_skin_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                
                # 原始日志
                if (batch_idx + 1) % args.print_freq == 0:
                    log_message(
                        f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | "
                        f"Loss:{loss.item():.4f} | Skel_MSE:{loss_skel_mse.item():.4f} "
                        f"| Skel_J2J:{loss_skel_j2j.item():.4f} | BoneReg:{bone_reg.item():.4f} "
                        f"| A:{skel_alpha:.2f} B:{skel_beta:.2f}"
                    )
        if jt.rank ==0:
            log_message(f"Epoch [{epoch+1}/{args.epochs}] 训练结束 | "
                    f"耗时: {time.time() - epoch_start_time:.2f}s | 学习率: {optimizer.lr:.6f}")

        # 修改第256行附近的验证部分：
        if val_loader and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_j2j_metric = 0.0
            val_skin_l1_metric = 0.0
            total_samples = 0
            
            with jt.no_grad():
                for data in val_loader:
                    vertices, joints, skin = data['vertices'], data['joints'], data['skin']
                    
                    # 处理模型输出
                    model_output = model(vertices)
                    if len(model_output) == 4:
                        pred_joints_flat, pred_skin, _, _ = model_output
                    else:
                        pred_joints_flat, pred_skin = model_output
                    
                    pred_joints = pred_joints_flat.reshape(-1, args.num_joints, 3)
                    batch_size = pred_joints.shape[0]
                    
                    # 计算 J2J 损失
                    batch_j2j = 0.0
                    for i in range(batch_size):
                        batch_j2j += J2J(pred_joints[i], joints[i]).item()
                    
                    # 累加损失和样本数
                    val_j2j_metric += batch_j2j
                    val_skin_l1_metric += criterion_l1(pred_skin, skin).item() * batch_size
                    total_samples += batch_size

            # 转换为 Jittor Var 以便进行 MPI 聚合
            val_j2j_var = jt.array([val_j2j_metric])
            val_skin_var = jt.array([val_skin_l1_metric])
            total_samples_var = jt.array([total_samples])
            
            # MPI 聚合（如果使用多卡）
            if jt.in_mpi:
                val_j2j_var = val_j2j_var.mpi_all_reduce()
                val_skin_var = val_skin_var.mpi_all_reduce()
                total_samples_var = total_samples_var.mpi_all_reduce()
            
            # 计算平均值
            val_j2j_metric = val_j2j_var.data[0] / total_samples_var.data[0]
            val_skin_l1_metric = val_skin_var.data[0] / total_samples_var.data[0]
            
            # 计算综合指标
            current_val_metric = val_j2j_metric + val_skin_l1_metric
            competition_score = 10/(math.sqrt(val_j2j_metric))*(0.5*(1-20*min(val_skin_l1_metric, 0.05)))
            log_message(f" 验证 | J2J: {val_j2j_metric:.4f} | Skin L1: {val_skin_l1_metric:.4f} | Combined: {current_val_metric:.4f} | Score: {competition_score:.4f}")

            if current_val_metric < best_val_metric and jt.rank == 0 :
                best_val_metric = current_val_metric
                model_path = os.path.join(args.output_dir, f'best_model.pkl')
                model.save(model_path)
                log_message(f"保存最佳模型，验证指标 {best_val_metric:.4f} -> {model_path}")
                
            if competition_score > best_competition_score and competition_score > 0 and  jt.rank == 0:
                best_competition_score = competition_score
                comp_model_path = os.path.join(args.output_dir, f'best_competition_model.pkl')
                model.save(comp_model_path)
                log_message(f"保存竞赛最佳模型（评分: {competition_score:.4f}） -> {comp_model_path}")
        
        if (epoch + 1) % args.save_freq == 0 and jt.rank == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"保存检查点 -> {checkpoint_path}")

    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"训练完成。最终模型保存在 {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description='联合训练骨骼和蒙皮模型')
    
    # --- 数据集参数 ---
    parser.add_argument('--train_data_list', type=str, required=True, help='训练数据列表文件路径')
    parser.add_argument('--val_data_list', type=str, default='', help='验证数据列表文件路径')
    parser.add_argument('--data_root', type=str, default='data', help='数据根目录')
    
    # --- 模型参数 ---
    parser.add_argument('--feat_dim', type=int, default=128, help='全局特征向量的维度')
    parser.add_argument('--num_joints', type=int, default=22, help='骨骼中的关节点数量')
    parser.add_argument('--pretrained_model', type=str, default='', help='预训练模型路径')
    
    # --- 训练参数 ---
    parser.add_argument('--batch_size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='使用的优化器')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减 (L2惩罚)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD优化器的动量')

    # --- 损失权重 ---
    parser.add_argument('--skeleton_loss_weight', type=float, default=1.0, help='骨骼总损失的权重')
    parser.add_argument('--skin_loss_weight', type=float, default=1.0, help='蒙皮总损失的权重')
    parser.add_argument('--skel_alpha', type=float, default=0.1, help='骨骼MSE损失的权重 (alpha)')
    parser.add_argument('--skel_beta', type=float, default=1.0, help='骨骼J2J损失的权重 (beta)')
    parser.add_argument('--bone_len_weight', type=float, default=0.1, help='骨长正则权重')

    # --- 采样参数 ---
    parser.add_argument('--num_samples', type=int, default=1024, help='每个点云的采样点数')
    parser.add_argument('--vertex_samples', type=int, default=512, help='顶点采样数')

    # --- 输出参数 ---
    parser.add_argument('--output_dir', type=str, default='output/combined', help='输出文件保存目录')
    parser.add_argument('--print_freq', type=int, default=20, help='打印频率')
    parser.add_argument('--save_freq', type=int, default=10, help='保存频率')
    parser.add_argument('--val_freq', type=int, default=1, help='验证频率')
    
    parser.add_argument('--random_pose', type=int, default=0,
                        help='Apply random pose to asset')
    
     # 新增track动作相关参数
    parser.add_argument('--use_track_poses', action='store_true',
                        help='使用track文件夹中的真实动作序列进行数据增强')
    parser.add_argument('--track_data_root', type=str, default='data/track',
                        help='track动作数据的根目录')
    
    args = parser.parse_args()
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()