import jittor as jt
import numpy as np
import os
import argparse
import random
from tqdm import tqdm
from scipy.spatial import cKDTree

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from models.combinedv5 import MultiScaleCombinedModel  # 使用最新的联合模型

# 设置Jittor flags
jt.flags.use_cuda = 1

def predict(args):
    # 创建联合模型
    model = MultiScaleCombinedModel(
        feat_dim=args.feat_dim,
        num_joints=args.num_joints
    )
    
    # 设置采样器
    sampler = SamplerMix(num_samples=args.num_samples, vertex_samples=args.vertex_samples)
    
    # 加载预训练模型
    if args.pretrained_model:
        print(f"加载预训练模型: {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # 创建数据加载器
    predict_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.predict_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=True,  # 需要原始顶点
    )
    
    # 创建输出目录
    predict_output_dir = args.predict_output_dir
    os.makedirs(predict_output_dir, exist_ok=True)
    
    print("开始联合预测...")
    model.eval()
    exporter = Exporter()
    
    with jt.no_grad():
        for batch_idx, data in tqdm(enumerate(predict_loader)):
            vertices, cls, id, origin_vertices, N = data['vertices'], data['cls'], data['id'], data['origin_vertices'], data['N']
            
            # 修改：处理模型的多个返回值
            model_output = model(vertices)
            if len(model_output) == 4:
                # 新的多尺度模型：使用完整的预测结果
                pred_joints_flat, pred_skin, main_joints_flat, hand_joints_flat = model_output
                use_multiscale = True
            else:
                # 旧模型的兼容性处理
                pred_joints_flat, pred_skin = model_output
                main_joints_flat = hand_joints_flat = None
                use_multiscale = False
            
            # 重塑关节预测
            B = vertices.shape[0]
            pred_joints = pred_joints_flat.reshape(B, args.num_joints, 3)
            
            # 如果是多尺度模型，也可以保存分层预测结果（可选）
            if use_multiscale:
                main_joints = main_joints_flat.reshape(B, 22, 3)
                hand_joints = hand_joints_flat.reshape(B, 30, 3)
            
            # 处理每个批次中的样本
            for i in range(B):
                # 为当前样本创建输出目录
                sample_output_dir = os.path.join(predict_output_dir, cls[i], str(id[i].item()))
                os.makedirs(sample_output_dir, exist_ok=True)
                
                # 1. 保存完整的预测骨架
                np.save(os.path.join(sample_output_dir, "predict_skeleton"), pred_joints[i].numpy())
                
                # # 可选：如果是多尺度模型，保存分层预测结果
                # if use_multiscale:
                #     np.save(os.path.join(sample_output_dir, "predict_main_skeleton"), main_joints[i].numpy())
                #     np.save(os.path.join(sample_output_dir, "predict_hand_skeleton"), hand_joints[i].numpy())
                
                # 2. 保存预测的蒙皮权重 - 使用和predict_skin.py相同的逻辑
                skin = pred_skin[i].numpy()
                o_vertices = origin_vertices[i, :N[i]].numpy()
                
                # 使用cKDTree找到最近的顶点
                tree = cKDTree(vertices[i].numpy())
                distances, indices = tree.query(o_vertices, k=3)
                
                # 计算基于距离的权重
                weights = 1 / (distances + 1e-6)
                weights /= weights.sum(axis=1, keepdims=True)  # 归一化
                
                # 计算最近3个点的蒙皮权重加权平均
                skin_resampled = np.zeros((o_vertices.shape[0], skin.shape[1]))
                for v in range(o_vertices.shape[0]):
                    skin_resampled[v] = weights[v] @ skin[indices[v]]
                
                # 保存结果
                np.save(os.path.join(sample_output_dir, "predict_skin"), skin_resampled)
                np.save(os.path.join(sample_output_dir, "transformed_vertices"), o_vertices)
                
                # 打印进度信息
                if batch_idx == 0 and i == 0:
                    print(f"样本 {cls[i]}/{id[i].item()}:")
                    print(f"  完整骨架形状: {pred_joints[i].shape}")
                    if use_multiscale:
                        print(f"  主骨架形状: {main_joints[i].shape}")
                        print(f"  手部骨架形状: {hand_joints[i].shape}")
                    print(f"  蒙皮权重形状: {skin_resampled.shape}")
                    print(f"  原始顶点数: {N[i].item()}")
    
    print("预测完成!")

def main():
    parser = argparse.ArgumentParser(description='联合预测骨骼和蒙皮权重')
    
    # 数据集参数
    parser.add_argument('--data_root', type=str, default='data',
                        help='数据文件的根目录')
    parser.add_argument('--predict_data_list', type=str, required=True,
                        help='预测数据列表文件路径')
    
    # 模型参数
    parser.add_argument('--feat_dim', type=int, default=128,
                        help='特征维度')
    parser.add_argument('--num_joints', type=int, default=52,
                        help='骨骼关节数量')
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='预训练模型路径')
    
    # 预测参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='预测批次大小')
    parser.add_argument('--num_samples', type=int, default=1024,
                        help='点云采样点数')
    parser.add_argument('--vertex_samples', type=int, default=512,
                        help='顶点采样数')
    parser.add_argument('--predict_output_dir', type=str, required=True,
                        help='预测结果保存目录')
    
    # 新增：多尺度预测选项
    parser.add_argument('--save_multiscale', action='store_true',
                        help='是否保存多尺度预测结果（主骨架和手部骨架）')
    
    args = parser.parse_args()
    predict(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()