import torch
import torch.distributions as dist

# --- 1. 定义 p(r) 函数 (修改为可处理 Tensor 输入) ---
def concentration_parameter_tensor(r_batch, threshold=10.0):
    """计算 Beta 分布的参数 p，使其随半径 r 增长。支持 Tensor 输入。"""
    # 使用 torch.max 确保 p >= 1.0
    # 确保 r_batch 是浮点类型
    r_batch = r_batch - threshold
    # p = torch.max(torch.tensor(1.0, device=r_batch.device), r_batch.float())
    # 可以选择其他增长更快的函数，例如:
    # p = r_batch.float()**2 + 1.0
    # p = torch.exp(r_batch.float() * 0.5) + 1.0 # 加 1 保证大于0
    # 慢很多
    p = torch.log(torch.max(torch.tensor(1.0, device=r_batch.device), r_batch.float())) + 1.0
    return p

def sample_concentrating_sphere(r_batch, n_samples, threshold = 10.0, device='cpu'):
    """
    使用 PyTorch 在一组半径 r_batch 的球面上确定性地采样，
    每个半径采样 n_samples 个点。点随着 r 增大越来越靠近赤道。

    Args:
        r_batch (torch.Tensor): 形状为 [N_radii] 的半径张量。
        n_samples (int): *每个*半径要采样的点的数量。
        device (str): 计算设备 ('cpu' or 'cuda').

    Returns:
        tuple: (theta, phi, r_tensor)
               theta: 形状为 [N_radii, n_samples] 的极角张量。
               phi: 形状为 [N_radii, n_samples] 的方位角张量。
               r_tensor: 形状为 [N_radii, n_samples] 的半径张量（扩展r_batch）。
    """
    # 检查 r_batch 是否有效 (例如，是否都大于 0)
    if torch.any(r_batch <= 0):
        # 对于无效半径，可以返回 NaN 或根据需要处理
        # 这里我们简单地继续，但后续计算可能对非正半径产生 NaN
        print("Warning: r_batch contains non-positive values.")
        # 或者可以抛出错误: raise ValueError("Radii must be positive.")

    N_radii = r_batch.shape[0]
    r_batch = r_batch.to(device) # 确保半径在正确的设备上

    # 计算 Beta 分布参数 p (形状 [N_radii])
    p_batch = concentration_parameter_tensor(r_batch, threshold)

    # --- 采样极角 theta ---
    # 1. 创建 Beta(p, p) 分布对象 (支持批处理参数)
    # p_batch shape [N_radii], beta_dist represents a batch of N_radii distributions
    beta_dist = dist.Beta(p_batch, p_batch)

    # 2. 从 Beta 分布采样 x in [0, 1]
    # sample([n_samples]) will return shape [n_samples, N_radii]
    # 我们需要对每个分布采样 n_samples 次
    # 使用 .expand() 和 .sample() 可能更直观，或者直接sample
    # sample shape needs to be [N_radii, n_samples]
    # Let's sample N_radii * n_samples independent values first
    # Resample might be needed if using .expand() is tricky
    # Alternative: loop (inefficient) or sample and reshape
    # Sample shape [n_samples, N_radii] is standard output shape
    x_beta = beta_dist.sample((n_samples,)) # shape [n_samples, N_radii]

    # 3. 将 x 转换回 z = cos(theta) in [-1, 1]
    z = 2 * x_beta - 1
    z = torch.clamp(z, -1.0, 1.0) # shape [n_samples, N_radii]

    # 4. 计算 theta = acos(z) in [0, pi]
    phi = 2 * torch.acos(z) / torch.pi - 1 # shape [n_samples, N_radii]

    # --- 采样方位角 phi ---
    # 在 [0, 2*pi) 上均匀采样, 需要形状 [N_radii, n_samples]
    theta = 2 * torch.rand(N_radii, n_samples, device=device) - 1 # shape [N_radii, n_samples]

    # --- 准备半径张量 ---
    # 将 r_batch [N_radii] 扩展为 [N_radii, n_samples]
    r_max = r_batch.max()
    r_tensor = 2 * r_batch.unsqueeze(1).expand(-1, n_samples) / r_max - 1 # shape [N_radii, n_samples]

    # --- 调整 theta 的形状 ---
    # 将 theta 从 [n_samples, N_radii] 转置为 [N_radii, n_samples]
    phi = phi.T # shape [N_radii, n_samples]

    return torch.stack([theta, phi, r_tensor], dim=-1).view(-1,3) # [N_radii * n_samples, 3]

def project_onto_planes(coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    planes = torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32, device=coordinates.device)
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).repeat(1, n_planes, 1, 1)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).repeat(N, 1, 1, 1)
    projections = coordinates @ inv_planes
    return projections[..., :2] # [N, n_planes, M, 3]
