import os
import numpy as np
import random
from PIL import Image  # 新增：用于验证图像完整性
from typing import Tuple, List  # 新增：类型提示，提升代码可读性

# 新增：全局数据集配置（严格按论文设定，避免硬编码）
DATASET_CONFIG = {
    "sysu": {
        "ir_cameras": ["cam3", "cam6"],  # 论文SYSU-MM01红外相机（🔶1-204）
        "vis_cameras_all": ["cam1", "cam2", "cam4", "cam5"],  # All-Search可见光相机
        "vis_cameras_indoor": ["cam1", "cam2"],  # Indoor-Search可见光相机（🔶1-204）
        "test_id_path": "exp/test_id.txt",  # 测试ID文件路径
        "pid_slice": slice(-13, -9),  # PID提取位置（路径后13-9位）
        "camid_idx": -15  # 相机ID提取位置（路径后15位）
    },
    "regdb": {
        "vis_test_prefix": "idx/test_visible_{trial}.txt",  # 可见光测试列表
        "thermal_test_prefix": "idx/test_thermal_{trial}.txt",  # 红外测试列表
        "img_path_prefix": ""  # 图像路径前缀（根据数据集实际结构调整）
    }
}

# query_img, query_id, query_cam = process_query_sysu("/SYSU-MM01", mode="all")
# print(query_img[:3])
# # ['/SYSU-MM01/cam3/0001/0001_0001.jpg', '/SYSU-MM01/cam3/0001/0001_0002.jpg', '/SYSU-MM01/cam6/0001/0001_0001.jpg']
# print(query_id[:3])
# # [1, 1, 1]
# print(query_cam[:3])
# # [3, 3, 6]

def process_query_sysu(
    data_path: str, 
    mode: str = "all"
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    按论文要求处理SYSU-MM01测试查询集（IR查询）（🔶1-204）
    Args:
        data_path: SYSU-MM01数据集根路径
        mode: 测试模式（"all"或"indoor"，仅影响画廊集，查询集IR相机固定为cam3、cam6）
    Returns:
        query_img: 有效查询图像路径列表
        query_id: 查询图像对应的行人ID（np.ndarray）
        query_cam: 查询图像对应的相机ID（np.ndarray）
    """
    # 1. 加载测试ID（论文test_id.txt格式：逗号分隔的ID，如"1,2,5,10"）（🔶1-204）
    test_id_path = os.path.join(data_path, DATASET_CONFIG["sysu"]["test_id_path"])
    if not os.path.exists(test_id_path):
        raise FileNotFoundError(f"SYSU-MM01测试ID文件不存在：{test_id_path}")
    
    with open(test_id_path, 'r') as f:
        ids_line = f.read().splitlines()[0].strip()  # 读取第一行（ID列表）
        ids = [int(id_str) for id_str in ids_line.split(',') if id_str.strip()]  # 过滤空字符串
        ids = ["%04d" % pid for pid in ids]  # 格式化为4位字符串（如1→"0001"）
    
    # 2. 加载IR查询图像（论文固定用cam3、cam6作为IR查询相机）（🔶1-204）
    ir_cameras = DATASET_CONFIG["sysu"]["ir_cameras"]
    raw_query_files = []
    for pid in sorted(ids):  # 按PID排序，确保一致性
        for cam in ir_cameras:
            cam_pid_dir = os.path.join(data_path, cam, pid)
            if not os.path.isdir(cam_pid_dir):
                print(f"警告：SYSU-MM01 IR路径不存在，跳过：{cam_pid_dir}")
                continue
            # 读取该目录下所有图像，按文件名排序（论文要求）
            img_names = sorted([name for name in os.listdir(cam_pid_dir) if name.endswith(('.jpg', '.png'))])
            img_paths = [os.path.join(cam_pid_dir, name) for name in img_names]
            raw_query_files.extend(img_paths)
    
    # 3. 过滤异常图像（损坏、尺寸不匹配），提取PID和相机ID（按论文路径格式）
    query_img = []
    query_id = []
    query_cam = []
    pid_slice = DATASET_CONFIG["sysu"]["pid_slice"]
    camid_idx = DATASET_CONFIG["sysu"]["camid_idx"]
    
    for img_path in raw_query_files:
        # 验证图像完整性
        try:
            with Image.open(img_path) as img:
                img.verify()  # 检查图像是否损坏
                # （可选）验证图像尺寸（按论文训练时的resize尺寸，避免测试尺寸不匹配）
                # if img.size != (args.img_w, args.img_h):
                #     print(f"警告：图像尺寸不匹配，跳过：{img_path}")
                #     continue
        except (IOError, SyntaxError) as e:
            print(f"警告：图像损坏，跳过：{img_path}，错误：{e}")
            continue
        
        # 提取PID和相机ID（严格按论文路径格式解析）
        try:
            pid = int(img_path[pid_slice])  # 路径后13-9位为PID（如"0001"）
            camid = int(img_path[camid_idx])  # 路径后15位为相机ID（如"cam3"→"3"）
        except (ValueError, IndexError) as e:
            print(f"警告：路径解析失败，跳过：{img_path}，错误：{e}")
            continue
        
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    
    # 4. 转换为numpy数组（便于后续评估代码处理）
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(
    data_path: str, 
    mode: str = "all", 
    trial: int = 0, 
    seed: int = 42  # 新增：全局种子，确保实验可复现
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    按论文要求处理SYSU-MM01测试画廊集（VIS画廊）（🔶1-204）
    Args:
        data_path: SYSU-MM01数据集根路径
        mode: 测试模式（"all"→所有VIS相机；"indoor"→仅cam1、cam2）
        trial: 第几次随机划分（1-10，论文要求10次平均）（🔶1-204）
        seed: 全局随机种子，确保不同trial的采样可复现
    Returns:
        gall_img: 有效画廊图像路径列表
        gall_id: 画廊图像对应的行人ID（np.ndarray）
        gall_cam: 画廊图像对应的相机ID（np.ndarray）
    """
    # 1. 固定随机种子（论文要求10次随机采样，需确保每次trial的采样可复现）
    random.seed(seed + trial)  # 结合trial和seed，避免不同trial采样重复
    np.random.seed(seed + trial)
    
    # 2. 选择对应模式的可见光相机（严格按论文设定）（🔶1-204）
    if mode == "all":
        vis_cameras = DATASET_CONFIG["sysu"]["vis_cameras_all"]
    elif mode == "indoor":
        vis_cameras = DATASET_CONFIG["sysu"]["vis_cameras_indoor"]
    else:
        raise ValueError(f"SYSU-MM01测试模式错误：{mode}，仅支持'all'或'indoor'")
    
    # 3. 加载测试ID（与查询集一致，确保身份匹配）
    test_id_path = os.path.join(data_path, DATASET_CONFIG["sysu"]["test_id_path"])
    if not os.path.exists(test_id_path):
        raise FileNotFoundError(f"SYSU-MM01测试ID文件不存在：{test_id_path}")
    
    with open(test_id_path, 'r') as f:
        ids_line = f.read().splitlines()[0].strip()
        ids = [int(id_str) for id_str in ids_line.split(',') if id_str.strip()]
        ids = ["%04d" % pid for pid in ids]
    
    # 4. 加载VIS画廊图像（论文要求：每个身份-相机对随机选1张）（🔶1-204）
    raw_gall_files = []
    for pid in sorted(ids):
        for cam in vis_cameras:
            cam_pid_dir = os.path.join(data_path, cam, pid)
            if not os.path.isdir(cam_pid_dir):
                print(f"警告：SYSU-MM01 VIS路径不存在，跳过：{cam_pid_dir}")
                continue
            # 读取该目录下所有有效图像（过滤非图像文件）
            img_names = [name for name in os.listdir(cam_pid_dir) if name.endswith(('.jpg', '.png'))]
            if not img_names:
                print(f"警告：SYSU-MM01 VIS目录无图像，跳过：{cam_pid_dir}")
                continue
            # 按论文要求：随机选1张图像（每次trial采样不同）
            selected_img = random.choice(img_names)
            raw_gall_files.append(os.path.join(cam_pid_dir, selected_img))
    
    # 5. 过滤异常图像，提取PID和相机ID（逻辑与查询集一致）
    gall_img = []
    gall_id = []
    gall_cam = []
    pid_slice = DATASET_CONFIG["sysu"]["pid_slice"]
    camid_idx = DATASET_CONFIG["sysu"]["camid_idx"]
    
    for img_path in raw_gall_files:
        # 验证图像完整性
        try:
            with Image.open(img_path) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            print(f"警告：图像损坏，跳过：{img_path}，错误：{e}")
            continue
        
        # 提取PID和相机ID
        try:
            pid = int(img_path[pid_slice])
            camid = int(img_path[camid_idx])
        except (ValueError, IndexError) as e:
            print(f"警告：路径解析失败，跳过：{img_path}，错误：{e}")
            continue
        
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(
    img_dir: str, 
    trial: int = 1, 
    modality: str = "visible"
) -> Tuple[List[str], np.ndarray]:
    """
    按论文要求处理RegDB测试集（支持可见光/红外模态）（🔶1-205）
    Args:
        img_dir: RegDB数据集根路径
        trial: 第几次随机划分（1-10，论文要求10次平均）（🔶1-205）
        modality: 测试模态（"visible"→可见光；"thermal"→红外）
    Returns:
        file_image: 有效测试图像路径列表
        file_label: 测试图像对应的行人ID（np.ndarray）
    """
    # 1. 校验参数合法性（避免模态错误或trial范围错误）
    if modality not in ["visible", "thermal"]:
        raise ValueError(f"RegDB测试模态错误：{modality}，仅支持'visible'或'thermal'")
    if not (1 <= trial <= 10):
        raise ValueError(f"RegDB trial范围错误：{trial}，需在1-10之间（论文要求）（🔶1-205）")
    
    # 2. 加载测试列表文件（严格按论文路径格式）（🔶1-205）
    if modality == "visible":
        test_list_name = DATASET_CONFIG["regdb"]["vis_test_prefix"].format(trial=trial)
    else:
        test_list_name = DATASET_CONFIG["regdb"]["thermal_test_prefix"].format(trial=trial)
    
    test_list_path = os.path.join(img_dir, test_list_name)
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"RegDB测试列表文件不存在：{test_list_path}")
    
    # 3. 解析测试列表（格式："img_path label"）
    raw_file_image = []
    raw_file_label = []
    with open(test_list_path, 'rt') as f:
        for line_idx, line in enumerate(f.read().splitlines(), 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            parts = line.split(' ')
            if len(parts) != 2:
                print(f"警告：RegDB测试列表行格式错误，跳过第{line_idx}行：{line}")
                continue
            img_rel_path, label_str = parts
            # 转换标签为整数
            try:
                label = int(label_str)
            except ValueError:
                print(f"警告：RegDB标签格式错误，跳过第{line_idx}行：{line}")
                continue
            # 拼接完整图像路径（避免冗余前缀）
            img_abs_path = os.path.join(img_dir, img_rel_path)
            raw_file_image.append(img_abs_path)
            raw_file_label.append(label)
    
    # 4. 过滤异常图像（确保测试时无IO错误）
    file_image = []
    file_label = []
    for img_path, label in zip(raw_file_image, raw_file_label):
        try:
            with Image.open(img_path) as img:
                img.verify()
                # RegDB红外图为单通道，此处无需转换（后续数据加载时统一处理）
        except (IOError, SyntaxError) as e:
            print(f"警告：图像损坏，跳过：{img_path}，错误：{e}")
            continue
        file_image.append(img_path)
        file_label.append(label)
    
    # 5. 转换为numpy数组（便于后续评估）
    return file_image, np.array(file_label)