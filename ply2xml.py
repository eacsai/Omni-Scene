import numpy as np

def standardize_bbox(pcl, points_per_object, percentile_to_keep=99.0):
    """
    先对整个点云剔除离群点，然后进行采样和归一化。
    
    该函数会：
    1. 对整个输入点云进行分析，保留最靠近中心的 `percentile_to_keep`% 的点。
    2. 从这些筛选出的“核心点”中，再随机采样出 `points_per_object` 个点。
    3. 对最后采样出的点云进行归一化，使其位于[-0.5, 0.5]的立方体内。

    参数:
    pcl (np.ndarray): 输入的原始点云，形状为 [N, 3]。
    points_per_object (int): 希望最终采样并返回的点的数量。
    percentile_to_keep (float): 在采样前，希望从原始点云中保留的核心点的百分比。

    返回:
    np.ndarray: 处理后并归一化的点云，形状为 [points_per_object, 3]。
    """
    print(f"原始点云数量: {pcl.shape[0]}")

    # --- 第一步：对整个点云进行离群点剔除 ---

    # 1. 计算整个点云的中心点
    center = np.mean(pcl, axis=0)

    # 2. 计算所有点到中心点的欧氏距离
    distances = np.linalg.norm(pcl - center, axis=1)

    # 3. 找到距离的百分位数阈值
    distance_threshold = np.percentile(distances, percentile_to_keep)
    print(f"保留靠近中心的 {percentile_to_keep}% 的点，距离阈值为: {distance_threshold:.4f}")

    # 4. 根据阈值筛选出核心点云
    core_points_mask = distances <= distance_threshold
    filtered_pcl = pcl[core_points_mask]
    
    print(f"筛选后剩余的核心点云数量: {filtered_pcl.shape[0]}")

    # --- 第二步：从核心点云中进行采样 ---
    
    # 5. 检查核心点的数量是否足够采样
    if filtered_pcl.shape[0] < points_per_object:
        print(f"警告: 筛选后的点数({filtered_pcl.shape[0]})少于期望的采样点数({points_per_object})。")
        # 在这种情况下，我们可以选择直接返回所有筛选出的点，或者报错。
        # 这里我们选择继续，但最终结果的点数会少于期望值。
        # 如果希望数量必须一致，可以设置 replace=True 进行有放回采样。
        replace_flag = True
    else:
        replace_flag = False

    pt_indices = np.random.choice(filtered_pcl.shape[0], points_per_object, replace=replace_flag)
    np.random.shuffle(pt_indices)
    sampled_pcl = filtered_pcl[pt_indices]

    print(f"从核心点中采样后的点云数量: {sampled_pcl.shape[0]}")

    # --- 第三步：对最后采样出的点云进行归一化 ---
    
    # 6. 计算边界框和缩放因子
    mins = np.amin(sampled_pcl, axis=0)
    maxs = np.amax(sampled_pcl, axis=0)
    new_center = (mins + maxs) / 2.
    new_scale = np.amax(maxs - mins)

    if new_scale == 0:
        new_scale = 1.0
        
    print("最终采样点的中心点: {}, 缩放因子: {}".format(new_center, new_scale))

    # 7. 归一化
    result = ((sampled_pcl - new_center) / new_scale).astype(np.float32)
    
    return result


# def standardize_bbox(pcl, points_per_object):
#     pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
#     np.random.shuffle(pt_indices)
#     pcl = pcl[pt_indices] # n by 3
#     mins = np.amin(pcl, axis=0)
#     maxs = np.amax(pcl, axis=0)
#     center = ( mins + maxs ) / 2.
#     scale = np.amax(maxs-mins)
#     print("Center: {}, Scale: {}".format(center, scale))
#     result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
#     return result

scale = 2
xml_head = \
f"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="{1600 * scale}"/>
            <integer name="height" value="{1200 * scale}"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.025"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]
xml_segments = [xml_head]

# 读取npy文件
# pcl = np.load('chair_pcl.npy')

# 读取PLY文件
from plyfile import PlyData
import pandas as pd
file_dir = 'point_cloud.ply'  #文件的路径
plydata = PlyData.read(file_dir)  # 读取文件
data = plydata.elements[0].data  # 读取数据
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
pcl = np.zeros(data_pd.shape, dtype=np.float64)  # 初始化储存数据的array
property_names = data[0].dtype.names  # 读取property的名字
for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    pcl[:, i] = data_pd[name]


pcl = standardize_bbox(pcl[:,:3], 2048)
pcl = pcl[:,[2,0,1]]
pcl[:,0] *= -1
pcl[:,2] += 0.0125

for i in range(pcl.shape[0]):
    color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
    xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
xml_segments.append(xml_tail)

xml_content = str.join('', xml_segments)

with open('mitsuba_scene.xml', 'w') as f:
    f.write(xml_content)
