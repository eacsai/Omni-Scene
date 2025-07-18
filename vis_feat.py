import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import open3d as o3d

def reshape_normalize(x):
    '''
    Args:
        x: [B, C, H, W]

    Returns:

    '''
    B, C, H, W = x.shape
    x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator==0, 1, denominator)
    return x / denominator

def normalize(x):
    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator == 0, 1, denominator)
    return x / denominator

def single_features_to_RGB(sat_features, idx=0, img_name='test_img.png'):
    sat_feat = sat_features[idx:idx+1,:,:,:].data.cpu().numpy()
    # 1. 重塑特征图形状为 [256, 64*64]
    B, C, H, W = sat_feat.shape
    flatten = np.concatenate([sat_feat], axis=0)
    # 2. 进行 PCA 降维到 3 维
    pca = PCA(n_components=3)
    pca.fit(reshape_normalize(flatten))
    
    # 3. 归一化到 [0, 1] 范围
    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, H, W, 3)

    sat = Image.fromarray((sat_feat_new[0] * 255).astype(np.uint8))
    # sat = sat.resize((512, 512))
    sat.save(img_name)

def reduce_gaussian_features_to_rgb(features):
    """
    使用PCA将高维特征降维至3维，并归一化为RGB值。

    参数:
    features (np.ndarray): 输入的特征数组，形状为 [B, N, C]。
                           例如：[6, 102400, 128]。

    返回:
    np.ndarray: 降维并归一化后的RGB特征，形状为 [B, N, 3]，数值范围在 [0, 1]。
    """
    # 1. 记录原始形状
    features = features.detach().cpu()
    B, N, C = features.shape
    print(f"原始特征形状: {features.shape}")

    # 2. 将数据重塑为2D数组 (B*N, C)，以便PCA处理
    #    PCA是按样本进行分析的，这里我们将B*N个点都看作样本
    features_reshaped = features.reshape(-1, C)
    print(f"重塑后用于PCA的形状: {features_reshaped.shape}")

    # 3. 初始化并执行PCA
    #    n_components=3 表示我们希望将特征降到3维
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_reshaped)
    print(f"PCA降维后的形状: {features_pca.shape}")

    # 4. 将降维后的数据重塑回原始的批次和空间维度
    features_pca_reshaped = features_pca.reshape(B, N, 3)
    print(f"恢复批次和空间维度后的形状: {features_pca_reshaped.shape}")

    # 5. (关键步骤) 归一化到[0, 1]范围，方便渲染为RGB颜色
    #    PCA的输出范围不是固定的，直接可视化效果会很差
    #    我们对每个通道（主成分）独立进行min-max归一化
    min_vals = features_pca_reshaped.min(axis=(0, 1), keepdims=True)
    max_vals = features_pca_reshaped.max(axis=(0, 1), keepdims=True)
    
    # 防止除以零
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    features_rgb = (features_pca_reshaped - min_vals) / range_vals
    print(f"最终RGB特征形状: {features_rgb.shape}")
    print(f"RGB特征的最小值: {features_rgb.min()}, 最大值: {features_rgb.max()}")

    return features_rgb

def point_features_to_rgb_colormap(point_features, cmap_name='viridis', zero_threshold=1e-6):
    """
    将点云的高维特征通过PCA降维并应用Colormap，生成RGB颜色。
    原始特征值都接近于零的点将被设置为黑色。

    参数:
    point_features (torch.Tensor or np.ndarray): 输入的特征张量，形状为 [B, N, C]。
    cmap_name (str): 要使用的matplotlib colormap的名称。
    zero_threshold (float): 用于判断特征值是否接近于零的阈值。

    返回:
    np.ndarray: 降维并应用颜色图后的RGB特征，形状为 [B, N, 3]，数值范围在 [0, 1]。
    """
    # --- 1. 确保数据为NumPy数组并获取形状 ---
    if hasattr(point_features, 'detach'): # 检查是否为PyTorch Tensor
        features = point_features.detach().cpu().numpy()
    elif isinstance(point_features, np.ndarray):
        features = point_features
    else:
        raise TypeError("输入必须是PyTorch张量或NumPy数组")
        
    B, N, C = features.shape
    print(f"原始特征形状: {features.shape}")

    # --- 2. 为全局PCA和Masking重塑数据 ---
    # 将所有批次的所有点合并，以便进行统一的PCA拟合
    features_reshaped = features.reshape(-1, C) # 形状变为 [B*N, C]
    
    # --- 3. 识别“零特征”点 ---
    # 在所有点中，找到那些所有特征通道都接近于零的点
    is_zero_mask_flat = np.all(np.abs(features_reshaped) < zero_threshold, axis=-1) # 形状为 [B*N]
    
    # --- 4. 执行PCA ---
    # 在所有点上拟合PCA，以确保颜色映射的全局一致性
    print("正在对所有点执行PCA...")
    pca = PCA(n_components=1)
    # 使用 fit_transform 学习并转换数据
    pc1_flat = pca.fit_transform(features_reshaped) # 形状为 [B*N, 1]

    # --- 5. 归一化第一主成分到 [0, 1] ---
    # 关键：为了获得更好的对比度，我们仅根据“非零特征”点的范围来确定归一化尺度
    pc1_non_zero = pc1_flat[~is_zero_mask_flat]
    
    if pc1_non_zero.size == 0:
        # 如果所有点都是零特征点，则所有点的颜色值设为0.5（灰色）
        normalized_pc1_flat = np.full_like(pc1_flat, 0.5)
    else:
        min_val = pc1_non_zero.min()
        max_val = pc1_non_zero.max()
        
        if max_val == min_val:
            # 如果所有非零点的值都一样，也设为0.5
            normalized_pc1_flat = np.full_like(pc1_flat, 0.5)
        else:
            # 使用非零点的范围进行归一化
            normalized_pc1_flat = (pc1_flat - min_val) / (max_val - min_val)
    
    # 将超出[0,1]范围的值裁剪掉，这可能发生在零特征点上
    normalized_pc1_flat = np.clip(normalized_pc1_flat, 0.0, 1.0)

    # --- 6. 应用Colormap ---
    print(f"正在应用 '{cmap_name}' 颜色图...")
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"警告: Colormap '{cmap_name}' 不存在，将使用 'viridis'。")
        cmap = plt.get_cmap('viridis')

    # cmap应用于一个1D数组会返回一个 [B*N, 4] 的RGBA数组
    colored_points_flat = cmap(normalized_pc1_flat.flatten())[:, :3] # 我们只取RGB，丢弃Alpha通道

    # --- 7. 应用零值掩码 ---
    # 将原始特征为零的点的颜色设置为黑色 (0, 0, 0)
    colored_points_flat[is_zero_mask_flat] = 0.0
    
    # --- 8. 恢复原始的批次形状 ---
    colored_points_rgb = colored_points_flat.reshape(B, N, 3)
    print(f"处理完成，返回的RGB颜色形状为: {colored_points_rgb.shape}")

    return colored_points_rgb


def save_point_cloud(points_xyz, points_rgb, filename="point_cloud.ply"):
    """
    将NumPy格式的坐标和颜色数据保存为PLY点云文件。

    参数:
    points_xyz (np.ndarray): 点的XYZ坐标，形状为 [N, 3]。
    points_rgb (np.ndarray): 点的RGB颜色，形状为 [N, 3]，数值范围应在 [0, 1] 之间。
    filename (str): 要保存的文件名。
    """
    # 1. 创建一个open3d的点云对象
    pcd = o3d.geometry.PointCloud()

    # 2. 将NumPy数组赋值给点云对象的points属性
    #    open3d需要的数据类型是Vector3dVector
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    print(f"成功加载 {len(pcd.points)} 个点。")

    # 3. 将NumPy数组赋值给点云对象的colors属性
    #    颜色值的范围必须在 [0, 1] 之间
    pcd.colors = o3d.utility.Vector3dVector(points_rgb)
    print(f"成功加载 {len(pcd.colors)} 个点的颜色。")

    # 4. 将点云对象写入文件
    #    write_ascii=True可以生成人类可读的文本文件，方便调试
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    print(f"点云已成功保存到当前目录下的 '{filename}' 文件中。")
    print("您可以使用MeshLab, CloudCompare或Blender等软件打开查看。")