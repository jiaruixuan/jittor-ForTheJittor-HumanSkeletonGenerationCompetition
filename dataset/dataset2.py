import jittor as jt
import numpy as np
import os
from jittor.dataset import Dataset

import os
from typing import List, Dict, Callable, Union

from .asset import Asset
from .sampler import Sampler

"""
写给符家祺看的
在大部分情况下你应该使用dataset2.py文件而非dataset文件，这里对data/track中的数据进行了应用并加入随机动作，而不是像官方那样仅使用了随机动作
"""


def transform(asset: Asset):
    """
    Transform the asset data into [-1, 1]^3.
    """
    # Find min and max values for each dimension of points
    min_vals = np.min(asset.vertices, axis=0)
    max_vals = np.max(asset.vertices, axis=0)
    
    # Calculate the center of the bounding box
    center = (min_vals + max_vals) / 2
    
    # Calculate the scale factor to normalize to [-1, 1]
    # We take the maximum range across all dimensions to preserve aspect ratio
    scale = np.max(max_vals - min_vals) / 2
    
    # Normalize points to [-1, 1]^3
    normalized_vertices = (asset.vertices - center) / scale
    
    # Apply the same transformation to joints
    if asset.joints is not None:
        normalized_joints = (asset.joints - center) / scale
    else:
        normalized_joints = None
    
    asset.vertices  = normalized_vertices
    asset.joints    = normalized_joints
    # remember to change matrix_local !
    asset.matrix_local[:, :3, 3] = normalized_joints

class RigDataset(Dataset):
    '''
    A simple dataset class.
    '''
    def __init__(
        self,
        data_root: str,
        paths: List[str],
        train: bool,
        batch_size: int,
        shuffle: bool,
        sampler: Sampler,
        transform: Union[Callable, None] = None,
        return_origin_vertices: bool = False,
        random_pose: bool = False,
        use_track_poses: bool = False,  # 新增参数
        track_data_root: str = None,    # track数据路径
    ):
        super().__init__()
        self.data_root  = data_root
        self.paths      = paths.copy()
        self.batch_size = batch_size
        self.train      = train
        self.shuffle    = shuffle
        self._sampler   = sampler # do not use `sampler` to avoid name conflict
        self.transform  = transform
        
        self.random_pose = random_pose
        
        self.return_origin_vertices = return_origin_vertices
        self.use_track_poses = use_track_poses
        self.track_poses = []
        
        # 加载track中的所有动作序列
        if self.use_track_poses and track_data_root:
            self._load_track_poses(track_data_root)
        
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.paths),
            shuffle=self.shuffle,
        )
    
    def _load_track_poses(self, track_data_root):
        """加载track文件夹中的所有动作序列"""
        track_files = [f for f in os.listdir(track_data_root) if f.endswith('.npz')]
        
        for track_file in track_files:
            track_path = os.path.join(track_data_root, track_file)
            track_data = np.load(track_path)
            
            if 'matrix_basis' in track_data:
                matrix_basis_sequence = track_data['matrix_basis']  # (frame, J, 4, 4)
                
                # 将所有帧的动作添加到pose池中
                for frame_idx in range(matrix_basis_sequence.shape[0]):
                    self.track_poses.append(matrix_basis_sequence[frame_idx])
        
        print(f"加载了 {len(self.track_poses)} 个track动作姿态")
    
    
    # 问题在于数据的比例，目前最好成绩的比例为4 3 3
    def _get_augmented_pose(self):
        """获取增强的姿态矩阵"""
        if self.use_track_poses and len(self.track_poses) > 0:
            # 50%概率使用track动作，50%概率使用随机动作
            if np.random.rand() < 1:
                # 使用track中的真实动作
                random_idx = np.random.randint(len(self.track_poses))
                return self.track_poses[random_idx].copy()
            else:
                # 使用随机生成的动作
                return None  # 将在后面生成随机动作
        else:
            # 只使用随机动作
            return None
    
    def __getitem__(self, index) -> Dict:
        """
        Get a sample from the dataset
        
        Args:
            index (int): Index of the sample
            
        Returns:
            data (Dict): Dictionary containing the following keys:
                - vertices: jt.Var, (B, N, 3) point cloud data
                - normals: jt.Var, (B, N, 3) point cloud normals
                - joints: jt.Var, (B, J, 3) joint positions
                - skin: jt.Var, (B, J, J) skinning weights
        """
        
        path = self.paths[index]
        asset = Asset.load(os.path.join(self.data_root, path))
        
        # 应用姿态增强
        if self.random_pose and np.random.rand() < 0.6:
            track_pose = self._get_augmented_pose()
            
            if track_pose is not None:
                # 使用track中的真实动作
                asset.apply_matrix_basis(track_pose)
            else:
                # 使用随机生成的动作
                asset.apply_matrix_basis(asset.get_random_matrix_basis(30.0))
        
        if self.transform is not None:
            self.transform(asset)
        origin_vertices = jt.array(asset.vertices.copy()).float32()
        
        sampled_asset = asset.sample(sampler=self._sampler)

        vertices    = jt.array(sampled_asset.vertices).float32()
        normals     = jt.array(sampled_asset.normals).float32()

        if sampled_asset.joints is not None:
            joints      = jt.array(sampled_asset.joints).float32()
        else:
            joints      = None

        if sampled_asset.skin is not None:
            skin        = jt.array(sampled_asset.skin).float32()
        else:
            skin        = None

        res = {
            'vertices': vertices,
            'normals': normals,
            'cls': asset.cls,
            'id': asset.id,
        }
        if joints is not None:
            res['joints'] = joints
        if skin is not None:
            res['skin'] = skin
        if self.return_origin_vertices:
            res['origin_vertices'] = origin_vertices
        return res
    
    def collate_batch(self, batch):
        if self.return_origin_vertices:
            max_N = 0
            for b in batch:
                max_N = max(max_N, b['origin_vertices'].shape[0])
            for b in batch:
                N = b['origin_vertices'].shape[0]
                b['origin_vertices'] = np.pad(b['origin_vertices'], ((0, max_N-N), (0, 0)), 'constant', constant_values=0.)
                b['N'] = N
        return super().collate_batch(batch)

# Example usage of the dataset
def get_dataloader(
    data_root: str,
    data_list: str,
    train: bool,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler,
    transform: Union[Callable, None] = None,
    return_origin_vertices: bool = False,
    random_pose: bool = False,
    use_track_poses: bool = False,      # 新增参数
    track_data_root: str = None,        # 新增参数
):
    """
    Create a dataloader for point cloud data
    
    Args:
        data_root (str): Root directory for the data files
        data_list (str): Path to the file containing list of data files
        train (bool): Whether the dataset is for training
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        sampler (Sampler): Sampler to use for point cloud sampling
        transform (callable, optional): Optional post-transform to be applied on a sample
        return_origin_vertices (bool): Whether to return original vertices
        
    Returns:
        dataset (RigDataset): The dataset
    """
    with open(data_list, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    dataset = RigDataset(
        data_root=data_root,
        paths=paths,
        train=train,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=return_origin_vertices,
        random_pose=random_pose,
        use_track_poses=use_track_poses,    # 新增
        track_data_root=track_data_root,    # 新增
    )
    
    return dataset
