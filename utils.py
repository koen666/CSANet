import os
import sys
import errno
import random
import copy
import numpy as np
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from sklearn.metrics import adjusted_rand_score
from PIL import Image

# -------------------------- CSANet全局配置（严格遵循论文设定） --------------------------
CSANET_CONFIG = {
    "curriculum": {
        "step_ii_total": 40,  # Step-II总epoch数（论文Step-I 20epoch，Step-II 40epoch）🔶1-209
        "plain_ratio": 1/3,    # Step-II前1/3为简单课程（🔶1-210 Algorithm 1）
        "moderate_ratio": 2/3  # Step-II中间1/3为中等课程，后1/3为复杂课程
    },
    "memory": {
        "momentum": 0.9,       # 记忆库动量更新系数（借鉴ClusterContrast）🔶1-134
        "tau": 0.05            # 温度系数（所有对比损失、概率响应统一设为0.05）🔶1-133、🔶1-182
    },
    "dbscan": {
        "sysu_eps": 0.6,       # SYSU-MM01的DBSCAN最大距离🔶1-209
        "regdb_eps": 0.3,      # RegDB的DBSCAN最大距离🔶1-209
        "min_samples": 2       # DBSCAN最小聚类样本数（避免单点聚类）🔶1-131
    },
    "eval": {
        "metrics": ["rank1", "mAP", "mINP"]  # 论文核心评估指标🔶1-206
    }
}

# -------------------------- 基础数据加载与索引生成工具 --------------------------
def load_data(input_data_path: str) -> tuple[list[str], list[int]]:
    """
    加载数据集列表文件（格式："img_path label"），适配RegDB/SYSU-MM01测试集加载🔶1-204、🔶1-205
    Args:
        input_data_path: 列表文件路径
    Returns:
        file_image: 图像路径列表
        file_label: 图像对应标签列表
    """
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"数据列表文件不存在：{input_data_path}")
    
    with open(input_data_path, 'rt') as f:
        data_file_list = f.read().splitlines()
    
    file_image = []
    file_label = []
    for line in data_file_list:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ')
        if len(parts) != 2:
            print(f"警告：无效行格式，跳过：{line}")
            continue
        img_path, label_str = parts
        try:
            label = int(label_str)
        except ValueError:
            print(f"警告：标签格式错误，跳过：{line}")
            continue
        file_image.append(img_path)
        file_label.append(label)
    
    return file_image, file_label


def GenIdx(train_color_label: np.ndarray, train_thermal_label: np.ndarray) -> tuple[list[list[int]], list[list[int]]]:
    """
    生成双模态身份-样本索引映射，适配有监督/伪监督场景下的样本定位🔶1-88
    Args:
        train_color_label: 可见光模态标签（真实/伪标签）
        train_thermal_label: 红外模态标签（真实/伪标签）
    Returns:
        color_pos: 可见光身份-索引列表（如[[0,2], [1,4]]表示ID0对应索引0/2）
        thermal_pos: 红外身份-索引列表
    """
    # 生成可见光索引映射
    color_pos = []
    unique_color_labels = np.unique(train_color_label)
    for label in unique_color_labels:
        idx = [k for k, v in enumerate(train_color_label) if v == label]
        color_pos.append(idx)
    
    # 生成红外索引映射
    thermal_pos = []
    unique_thermal_labels = np.unique(train_thermal_label)
    for label in unique_thermal_labels:
        idx = [k for k, v in enumerate(train_thermal_label) if v == label]
        thermal_pos.append(idx)
    
    return color_pos, thermal_pos


def GenIdx_single(label: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    """
    生成单模态身份-样本索引映射与身份占比，适配Step-I单模态聚类🔶1-132
    Args:
        label: 单模态标签（真实/伪标签）
    Returns:
        pos: 身份-索引列表
        prob: 各身份样本占比（用于采样权重）
    """
    pos = []
    num = []
    max_label = np.max(label) if len(label) > 0 else 0
    unique_labels = np.unique(label)
    
    for i in range(max_label + 1):
        if i in unique_labels:
            idx = [k for k, v in enumerate(label) if v == i]
            pos.append(idx)
            num.append(len(idx))
        else:
            pos.append([])
            num.append(0)
    
    # 计算各身份占比（避免除以零）
    total = np.sum(num)
    prob = np.array(num) / total if total > 0 else np.zeros_like(num)
    return pos, prob


def GenCamIdx(gall_img: list[str], gall_label: np.ndarray, mode: str = "all") -> list[list[int]]:
    """
    生成SYSU-MM01画廊集的“身份-相机”索引映射，适配测试集相机过滤🔶1-204
    Args:
        gall_img: 画廊集图像路径列表
        gall_label: 画廊集标签
        mode: 测试模式（"all"→4个可见光相机；"indoor"→2个室内相机）
    Returns:
        sample_pos: （身份-相机）对应的样本索引列表
    """
    # 论文SYSU-MM01相机设定：all模式用cam1/2/4/5，indoor模式用cam1/2🔶1-204
    cam_idx_map = {"all": [1,2,4,5], "indoor": [1,2]}
    if mode not in cam_idx_map:
        raise ValueError(f"无效模式：{mode}，仅支持'all'/'indoor'")
    target_cams = cam_idx_map[mode]
    
    # 提取画廊集图像的相机ID（路径后10位为相机标识，如"cam1/..."→1）
    gall_cam = []
    for img_path in gall_img:
        try:
            cam_id = int(img_path[-10])  # 适配路径格式：".../camX/.../xxx.jpg"
            gall_cam.append(cam_id)
        except (IndexError, ValueError):
            print(f"警告：相机ID解析失败，跳过：{img_path}")
            gall_cam.append(-1)  # 标记为无效相机
    
    # 生成“身份-相机”索引映射
    sample_pos = []
    unique_labels = np.unique(gall_label)
    for label in unique_labels:
        for cam in target_cams:
            # 筛选该身份+该相机的样本索引
            idx = [k for k, (v, c) in enumerate(zip(gall_label, gall_cam)) if v == label and c == cam]
            if idx:
                sample_pos.append(idx)
    
    return sample_pos

# -------------------------- CSANet核心采样器（适配课程学习） --------------------------
class CSANetCurriculumSampler(Sampler):
    """
    CSANet无监督课程采样器：按“简单→中等→复杂”阶段采样双模态样本🔶1-99、🔶1-210
    支持Step-II不同阶段仅加载对应课程样本，且双模态样本按CAP关联匹配🔶1-166
    """
    def __init__(
        self,
        train_color_pseudo_label: np.ndarray,  # 可见光伪标签（DBSCAN生成）
        train_thermal_pseudo_label: np.ndarray, # 红外伪标签
        color_curriculum_mask: np.ndarray,     # 可见光课程掩码（0=plain,1=moderate,2=intricate）
        thermal_curriculum_mask: np.ndarray,   # 红外课程掩码
        cap_mapping: dict,                     # CAP传递的跨模态关联（vis_pid→ir_pid）🔶1-174
        num_pos: int = 4,                      # 每个身份采样样本数
        batch_size: int = 8,                   # 每个batch的身份数（论文设定）🔶1-208
        current_stage: str = "plain",          # 当前课程阶段
        dataset_name: str = "sysu"
    ):
        # 1. 筛选当前课程的样本全局索引
        stage2level = {"plain": 0, "moderate": 1, "intricate": 2}
        if current_stage not in stage2level:
            raise ValueError(f"无效阶段：{current_stage}，仅支持'plain'/'moderate'/'intricate'")
        target_level = stage2level[current_stage]
        
        # 筛选可见光/红外当前课程的全局索引
        self.color_global_idx = np.where(color_curriculum_mask == target_level)[0]
        self.thermal_global_idx = np.where(thermal_curriculum_mask == target_level)[0]
        
        if len(self.color_global_idx) == 0 or len(self.thermal_global_idx) == 0:
            raise RuntimeError(f"当前阶段{current_stage}无有效样本，检查课程掩码")
        
        # 2. 构建当前课程的“身份-局部索引”映射（局部索引对应筛选后的样本）
        # 可见光映射
        color_pid_in_stage = train_color_pseudo_label[self.color_global_idx]
        self.color_pid2local = defaultdict(list)
        for local_idx, global_idx in enumerate(self.color_global_idx):
            pid = train_color_pseudo_label[global_idx]
            self.color_pid2local[pid].append(local_idx)
        
        # 红外映射
        thermal_pid_in_stage = train_thermal_pseudo_label[self.thermal_global_idx]
        self.thermal_pid2local = defaultdict(list)
        for local_idx, global_idx in enumerate(self.thermal_global_idx):
            pid = train_thermal_pseudo_label[global_idx]
            self.thermal_pid2local[pid].append(local_idx)
        
        # 3. 筛选有CAP跨模态关联的有效身份
        self.valid_vis_pids = [pid for pid in self.color_pid2local.keys() if pid in cap_mapping]
        if len(self.valid_vis_pids) == 0:
            raise RuntimeError(f"当前阶段{current_stage}无CAP关联身份，检查关联字典")
        
        # 4. 采样参数初始化
        self.num_pos = num_pos
        self.batch_size = batch_size
        self.cap_mapping = cap_mapping  # vis_pid → ir_pid
        # 采样总长度：覆盖2轮所有样本（避免训练中断）
        self.total_samples = max(len(self.color_global_idx), len(self.thermal_global_idx)) * 2
        self.N = self.total_samples // self.num_pos  # 迭代次数（每个样本对算1次）

    def __iter__(self):
        batch_num = self.total_samples // (self.batch_size * self.num_pos)
        for _ in range(batch_num):
            # 随机选择当前batch的可见光身份
            batch_vis_pids = np.random.choice(self.valid_vis_pids, self.batch_size, replace=False)
            for vis_pid in batch_vis_pids:
                # 1. 采样可见光样本（当前课程内）
                vis_local_idx = np.random.choice(
                    self.color_pid2local[vis_pid], 
                    self.num_pos, 
                    replace=len(self.color_pid2local[vis_pid]) < self.num_pos
                )
                vis_global_idx = self.color_global_idx[vis_local_idx]  # 转为全局索引
                
                # 2. 采样对应红外样本（按CAP关联找红外身份）
                ir_pid = self.cap_mapping[vis_pid]
                if ir_pid not in self.thermal_pid2local:
                    print(f"警告：红外无身份{ir_pid}，随机采样红外样本")
                    ir_local_idx = np.random.choice(
                        range(len(self.thermal_global_idx)), 
                        self.num_pos, 
                        replace=False
                    )
                else:
                    ir_local_idx = np.random.choice(
                        self.thermal_pid2local[ir_pid], 
                        self.num_pos, 
                        replace=len(self.thermal_pid2local[ir_pid]) < self.num_pos
                    )
                ir_global_idx = self.thermal_global_idx[ir_local_idx]  # 转为全局索引
                
                # 3. 返回（可见光全局索引，红外全局索引）对
                for c_idx, t_idx in zip(vis_global_idx, ir_global_idx):
                    yield (c_idx, t_idx)

    def __len__(self):
        return self.N


class CSANetSingleModalitySampler(Sampler):
    """
    CSANet单模态采样器：用于Step-I单模态对比聚类（仅采样单个模态样本）🔶1-132
    """
    def __init__(
        self,
        modal_pseudo_label: np.ndarray,  # 单模态伪标签
        num_pos: int = 4,
        batch_size: int = 8,
        dataset_len: int = None
    ):
        # 构建“身份-样本索引”映射
        self.pid2idx = defaultdict(list)
        for idx, pid in enumerate(modal_pseudo_label):
            self.pid2idx[pid].append(idx)
        
        # 筛选有效身份（样本数≥num_pos）
        self.valid_pids = [pid for pid in self.pid2idx.keys() if len(self.pid2idx[pid]) >= num_pos]
        if len(self.valid_pids) == 0:
            raise RuntimeError("无足够样本的身份，检查DBSCAN聚类结果")
        
        # 采样参数初始化
        self.num_pos = num_pos
        self.batch_size = batch_size
        self.dataset_len = dataset_len if dataset_len is not None else len(modal_pseudo_label)
        self.total_samples = self.dataset_len * 2  # 覆盖2轮采样
        self.N = self.total_samples // self.num_pos  # 迭代次数

    def __iter__(self):
        batch_num = self.total_samples // (self.batch_size * self.num_pos)
        for _ in range(batch_num):
            # 随机选择当前batch的身份
            batch_pids = np.random.choice(self.valid_pids, self.batch_size, replace=False)
            for pid in batch_pids:
                # 采样该身份的num_pos个样本
                sample_idx = np.random.choice(self.pid2idx[pid], self.num_pos, replace=False)
                for idx in sample_idx:
                    yield idx

    def __len__(self):
        return self.N


class IdentitySampler(Sampler):
    """
    有监督双模态采样器（基线方法用）：基于真实标签采样，适配对比实验🔶1-212
    """
    def __init__(
        self,
        train_color_label: np.ndarray,
        train_thermal_label: np.ndarray,
        color_pos: list[list[int]],
        thermal_pos: list[list[int]],
        num_pos: int = 4,
        batchSize: int = 8,
        dataset_num_size: int = 2
    ):
        self.uni_label = np.unique(train_color_label)
        self.n_classes = len(self.uni_label)
        self.num_pos = num_pos
        self.batchSize = batchSize
        
        # 计算采样总长度
        max_len = np.maximum(len(train_color_label), len(train_thermal_label))
        self.N = dataset_num_size * max_len

        # 预生成采样索引
        self.index1 = []  # 可见光索引
        self.index2 = []  # 红外索引
        batch_num = int(self.N / (batchSize * num_pos)) + 1
        for _ in range(batch_num):
            # 随机选择batch身份
            batch_idx = np.random.choice(self.uni_label, batchSize, replace=False)
            for pid in batch_idx:
                # 采样可见光样本
                color_sample = np.random.choice(color_pos[pid], num_pos)
                # 采样红外样本
                thermal_sample = np.random.choice(thermal_pos[pid], num_pos)
                # 拼接索引
                self.index1.extend(color_sample)
                self.index2.extend(thermal_sample)
        
        # 截断到指定长度
        self.index1 = self.index1[:self.N]
        self.index2 = self.index2[:self.N]

    def __iter__(self):
        return iter(zip(self.index1, self.index2))

    def __len__(self):
        return self.N


class SemiIdentitySampler_pseudoIR(Sampler):
    """
    半监督伪标签采样器（基线方法用）：适配红外伪标签场景，用于对比实验🔶1-212
    """
    def __init__(
        self,
        train_color_label: np.ndarray,
        train_thermal_label: np.ndarray,
        color_pos: list[list[int]],
        num_pos: int = 4,
        batchSize: int = 8,
        dataset_num_size: int = 2
    ):
        self.uni_label_thermal = np.unique(train_thermal_label)
        self.n_classes_thermal = len(self.uni_label_thermal)
        self.num_pos = num_pos
        self.batchSize = batchSize
        
        # 生成红外身份-索引映射
        self.thermal_pos, _ = GenIdx_single(train_thermal_label)
        
        # 计算采样总长度
        max_len = np.maximum(len(train_color_label), len(train_thermal_label))
        self.N = dataset_num_size * max_len

        # 预生成batch身份列表
        batch_idx_list = []
        uni_label_temp = copy.deepcopy(self.uni_label_thermal)
        batch_num = int(self.N / (batchSize * num_pos)) + 1
        for _ in range(batch_num):
            batch_idx = []
            for _ in range(batchSize):
                if len(uni_label_temp) == 0:
                    uni_label_temp = copy.deepcopy(self.uni_label_thermal)
                idx = random.randint(0, len(uni_label_temp)-1)
                batch_idx.append(uni_label_temp[idx])
                uni_label_temp = np.delete(uni_label_temp, idx)
            batch_idx_list.append(np.array(batch_idx))

        # 预生成采样索引
        self.index1 = []  # 可见光索引
        self.index2 = []  # 红外索引
        color_pos_temp = copy.deepcopy(color_pos)
        thermal_pos_temp = copy.deepcopy(self.thermal_pos)
        for batch_idx in batch_idx_list:
            for pid in batch_idx:
                # 采样可见光样本（循环复用）
                if len(color_pos_temp[pid]) == 0:
                    color_pos_temp[pid] = copy.deepcopy(color_pos[pid])
                # 采样红外样本（循环复用）
                if len(thermal_pos_temp[pid]) == 0:
                    thermal_pos_temp[pid] = copy.deepcopy(self.thermal_pos[pid])
                
                sample_color = []
                sample_thermal = []
                for _ in range(num_pos):
                    # 可见光采样
                    c_idx = random.randint(0, len(color_pos_temp[pid])-1)
                    sample_color.append(color_pos_temp[pid][c_idx])
                    color_pos_temp[pid].pop(c_idx)
                    # 红外采样
                    t_idx = random.randint(0, len(thermal_pos_temp[pid])-1)
                    sample_thermal.append(thermal_pos_temp[pid][t_idx])
                    thermal_pos_temp[pid].pop(t_idx)
                
                self.index1.extend(sample_color)
                self.index2.extend(sample_thermal)
        
        # 截断到指定长度
        self.index1 = self.index1[:self.N]
        self.index2 = self.index2[:self.N]

    def __iter__(self):
        return iter(zip(self.index1, self.index2))

    def __len__(self):
        return self.N

# -------------------------- 课程学习与记忆库辅助工具 --------------------------
def generate_curriculum_mask(
    modal_pseudo_label: np.ndarray,
    tbgm_curriculum: list[tuple[int, int]],  # TBGM输出：(pid, level)，level=0/1/2
    dataset_name: str = "sysu"
) -> np.ndarray:
    """
    生成单模态课程掩码：为每个样本分配课程级别（0=plain,1=moderate,2=intricate）🔶1-99、🔶1-101
    Args:
        modal_pseudo_label: 单模态伪标签
        tbgm_curriculum: TBGM模块的课程划分结果
    Returns:
        curriculum_mask: 课程掩码（长度=N，值为0/1/2）
    """
    # 构建“身份-课程级别”映射
    pid2level = dict(tbgm_curriculum)
    # 初始化掩码（-1表示未分配）
    curriculum_mask = np.ones_like(modal_pseudo_label, dtype=int) * -1
    
    # 为每个样本分配课程级别
    for idx, pid in enumerate(modal_pseudo_label):
        if pid in pid2level:
            curriculum_mask[idx] = pid2level[pid]
        else:
            # 未划分的样本归为复杂课程（论文默认处理）
            curriculum_mask[idx] = 2
    
    # 检查未分配样本
    unassigned_num = np.sum(curriculum_mask == -1)
    if unassigned_num > 0:
        print(f"警告：{unassigned_num}个样本未分配课程，已归为复杂课程")
    
    return curriculum_mask


def get_current_curriculum(epoch: int, step_ii_total: int = CSANET_CONFIG["curriculum"]["step_ii_total"]) -> str:
    """
    根据Step-II当前epoch判断课程阶段（遵循论文Algorithm 1）🔶1-210
    Args:
        epoch: Step-II的当前epoch（1-based）
        step_ii_total: Step-II总epoch数
    Returns:
        current_stage: 课程阶段（"plain"/"moderate"/"intricate"）
    """
    if not (1 <= epoch <= step_ii_total):
        raise ValueError(f"Step-II epoch需在1~{step_ii_total}之间，当前为{epoch}")
    
    stage1_end = int(step_ii_total * CSANET_CONFIG["curriculum"]["plain_ratio"])
    stage2_end = int(step_ii_total * CSANET_CONFIG["curriculum"]["moderate_ratio"])
    
    if epoch <= stage1_end:
        return "plain"
    elif epoch <= stage2_end:
        return "moderate"
    else:
        return "intricate"


def count_curriculum_anchors(
    curriculum_mask: np.ndarray,
    modal_pseudo_label: np.ndarray
) -> dict:
    """
    统计当前课程各阶段的锚点数量（身份数+样本数），适配论文Fig.6(a)分析🔶1-265
    Args:
        curriculum_mask: 课程掩码
        modal_pseudo_label: 单模态伪标签
    Returns:
        anchor_stats: 统计结果（如{"plain_pid_num": 50, "plain_sample_num": 500}）
    """
    anchor_stats = defaultdict(int)
    level2name = {0: "plain", 1: "moderate", 2: "intricate"}
    
    for level, name in level2name.items():
        # 筛选该课程级别的样本
        level_mask = (curriculum_mask == level)
        level_pids = modal_pseudo_label[level_mask]
        # 统计身份数和样本数
        anchor_stats[f"{name}_pid_num"] = len(np.unique(level_pids))
        anchor_stats[f"{name}_sample_num"] = np.sum(level_mask)
    
    return anchor_stats


def build_modal_memory(
    modal_feats: np.ndarray,
    modal_pseudo_label: np.ndarray,
    momentum: float = CSANET_CONFIG["memory"]["momentum"],
    old_memory: dict = None  # 旧记忆库：{pid: centroid}
) -> tuple[np.ndarray, dict]:
    """
    按论文公式3构建/更新单模态记忆库（聚类中心），支持动量更新🔶1-132、🔶1-134
    Args:
        modal_feats: 单模态特征（shape=[N, d]）
        modal_pseudo_label: 单模态伪标签（shape=[N]）
        old_memory: 旧记忆库（用于动量更新，None则初始化）
    Returns:
        memory_array: 记忆库数组（shape=[C, d]，C为身份数）
        pid2idx: 身份到记忆库索引的映射（dict）
    """
    # 1. 计算当前聚类中心（公式3）
    pid2centroid = defaultdict(np.ndarray)
    pid_counter = Counter(modal_pseudo_label)
    
    for pid in pid_counter.keys():
        feats_pid = modal_feats[modal_pseudo_label == pid]
        centroid = np.mean(feats_pid, axis=0)  # 聚类中心=特征平均值
        pid2centroid[pid] = centroid
    
    # 2. 动量更新（若有旧记忆库）
    if old_memory is not None:
        for pid in pid2centroid.keys():
            if pid in old_memory:
                # 公式：new = momentum * old + (1 - momentum) * current
                pid2centroid[pid] = momentum * old_memory[pid] + (1 - momentum) * pid2centroid[pid]
        # 保留旧记忆库中未出现的身份（避免身份丢失）
        for pid in old_memory.keys():
            if pid not in pid2centroid:
                pid2centroid[pid] = old_memory[pid]
    
    # 3. 格式转换（排序后转为数组）
    valid_pids = sorted(pid2centroid.keys())
    memory_array = np.array([pid2centroid[pid] for pid in valid_pids])
    pid2idx = {pid: idx for idx, pid in enumerate(valid_pids)}
    
    return memory_array, pid2idx


def compute_prob_response(
    feats: torch.Tensor,
    memory: torch.Tensor,
    tau: float = CSANET_CONFIG["memory"]["tau"]
) -> torch.Tensor:
    """
    计算概率响应（论文公式14、16、17、18），用于IPCC模块🔶1-187、🔶1-193
    Args:
        feats: 输入特征（shape=[B, d]，B为batch size）
        memory: 参考记忆库（shape=[C, d]，C为身份数）
        tau: 温度系数
    Returns:
        prob: 概率响应（shape=[B, C]，每行和为1）
    """
    # 计算余弦相似度（feats与memory的 pairwise 相似度）
    sim = F.cosine_similarity(feats.unsqueeze(1), memory.unsqueeze(0), dim=-1)  # [B, C]
    # 温度缩放+softmax归一化
    prob = F.softmax(sim / tau, dim=1)
    return prob


def kl_div_consistency_loss(
    instance_prob: torch.Tensor,
    prototype_prob: torch.Tensor
) -> torch.Tensor:
    """
    计算IPCC模块的KL散度损失（论文公式19），强制实例与原型概率响应一致🔶1-194
    Args:
        instance_prob: 复杂实例的概率响应（shape=[B, C]）
        prototype_prob: 对应原型的概率响应（shape=[B, C]）
    Returns:
        kl_loss: KL散度损失（标量）
    """
    # 加eps避免log(0)
    eps = 1e-10
    kl_loss = F.kl_div(
        instance_prob.log() + eps,
        prototype_prob + eps,
        reduction="batchmean"  # 按batch平均
    )
    return kl_loss

# -------------------------- 实验评估与日志工具 --------------------------
class AverageMeter(object):
    """
    平均指标计算器：用于跟踪训练/测试过程中的损失、准确率等指标🔶1-212、🔶1-223
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0    # 当前值
        self.avg = 0.0    # 平均值
        self.sum = 0.0    # 总和
        self.count = 0    # 计数

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def compute_ari_score(
    true_label: np.ndarray,  # 修正：强制要求输入真实标签（论文🔶1-256需基于真实标签评估）
    pseudo_label: np.ndarray,
    modal: str = "vis"
) -> float:
    """
    计算Adjusted Rand Index（ARI），评估伪标签与真实标签的聚类一致性（论文表VIII）🔶1-256
    论文中用于衡量可见光（VIS）、红外（IR）模态伪标签的可靠性，值越高聚类质量越好。
    
    Args:
        true_label: 模态真实标签（np.ndarray，shape=[N]），需与数据集真实身份对应（如SYSU-MM01测试集96个身份）🔶1-204
        pseudo_label: 模态伪标签（np.ndarray，shape=[N]），由DBSCAN聚类生成（🔶1-131）
        modal: 模态标识（仅支持"vis"=可见光、"ir"=红外），匹配论文双模态设定🔶1-88
    
    Returns:
        ari: ARI分数（范围[-1,1]，1表示完全一致，0表示随机聚类）
    
    Raises:
        ValueError: 若参数不满足论文设定（如标签长度不匹配、模态标识错误）
    """
    # 1. 校验模态标识（仅支持论文中的双模态）🔶1-88
    if modal not in ["vis", "ir"]:
        raise ValueError(
            f"模态标识错误：{modal}，仅支持'vis'（可见光）或'ir'（红外）"
            "（参考论文🔶1-88，VI-ReID仅涉及可见光-红外双模态）"
        )
    
    # 2. 校验标签数组非空
    if len(true_label) == 0 or len(pseudo_label) == 0:
        raise ValueError(
            "真实标签/伪标签为空数组，无法计算ARI"
            "（参考论文🔶1-256，需输入有效样本的标签）"
        )
    
    # 3. 校验真实标签与伪标签长度一致（论文要求一一对应）🔶1-256
    if len(true_label) != len(pseudo_label):
        raise ValueError(
            f"真实标签与伪标签长度不匹配：{len(true_label)} vs {len(pseudo_label)}"
            "（参考论文🔶1-256，同一模态的真实标签与伪标签需覆盖相同样本）"
        )
    
    # 4. 计算ARI（严格遵循论文表VIII的评估逻辑）🔶1-256
    try:
        ari = adjusted_rand_score(true_label, pseudo_label)
    except Exception as e:
        raise RuntimeError(
            f"ARI计算失败：{str(e)}"
            "（检查标签格式：需为整数数组，如真实标签[0,0,1,1]、伪标签[1,1,0,0]）"
        )
    
    # 5. 打印与论文一致的模态标识（如"VIS-ARI" "IR-ARI"）🔶1-256
    modal_name = "VIS" if modal == "vis" else "IR"
    print(f"{modal_name}-ARI分数：{ari:.4f}（参考论文表VIII，值越高聚类质量越好）")
    
    return ari


class CSANetLogger(object):
    """
    CSANet专用日志工具：记录课程阶段、指标变化、锚点数量等关键信息🔶1-265、🔶1-267
    """
    def __init__(self, fpath: str = None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w', encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg: str):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

    def log_curriculum(self, epoch: int, step: str, stats: dict):
        """记录课程阶段统计信息（锚点数量等）"""
        msg = f"[{step}] Epoch {epoch:3d} | 课程统计："
        for k, v in stats.items():
            msg += f"{k}={v:4d} | "
        msg += "\n"
        self.write(msg)
        self.flush()

    def log_metrics(self, epoch: int, step: str, metrics: dict):
        """记录评估指标（Rank-1、mAP等）"""
        msg = f"[{step}] Epoch {epoch:3d} | 评估指标："
        for k, v in metrics.items():
            msg += f"{k}={v:.4f} | "
        msg += "\n"
        self.write(msg)
        self.flush()

    def log_loss(self, epoch: int, step: str, losses: dict):
        """记录损失变化"""
        msg = f"[{step}] Epoch {epoch:3d} | 损失："
        for k, v in losses.items():
            msg += f"{k}={v:.4f} | "
        msg += "\n"
        self.write(msg)
        self.flush()

# -------------------------- 通用工具函数 --------------------------
def mkdir_if_missing(directory: str):
    """创建目录（若不存在）"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def set_seed(seed: int, cuda: bool = True):
    """
    严格设置随机种子，确保实验可复现（覆盖numpy、torch、CUDA）🔶1-204、🔶1-205
    Args:
        seed: 随机种子
        cuda: 是否使用CUDA
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 禁用cuDNN自动优化（避免相同种子下结果差异）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子：{seed}（CUDA={cuda}）")


def sort_list_with_unique_index(initial_list: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    排序并获取每个唯一值的首尾索引与数量，适配样本分组统计🔶1-256
    Returns:
        s1: 每个唯一值的首索引
        s2: 每个唯一值的尾索引
        num: 每个唯一值的数量
        idx_: 排序后的唯一值
        s3: 每个唯一值的所有索引
    """
    a = np.asarray(initial_list)
    if len(a) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), defaultdict(list)
    
    # 获取唯一值及其首次出现索引
    a_u, idx = np.unique(a, return_index=True)
    idx_sorted = np.sort(idx)
    idx_ = a[idx_sorted]  # 排序后的唯一值
    
    # 初始化统计数组
    max_val = a_u[-1]
    s1 = np.ones(max_val + 1, dtype=int) * -1  # 首索引
    s2 = np.ones(max_val + 1, dtype=int) * -1  # 尾索引
    num = np.zeros(max_val + 1, dtype=int)     # 数量
    s3 = defaultdict(list)                     # 所有索引
    
    # 遍历统计
    for i, val in enumerate(a):
        if val not in a_u:
            continue
        if s1[val] == -1:
            s1[val] = i
            s2[val] = i
            num[val] = 1
        else:
            s2[val] = i
            num[val] += 1
        s3[val].append(i)
    
    # 筛选有效唯一值的统计结果
    s1 = s1[idx_]
    s2 = s2[idx_]
    num = num[idx_]
    
    return s1, s2, num, idx_, s3


def validate_image_path(img_paths: list[str]) -> list[str]:
    """
    验证图像路径有效性（存在且可打开），适配测试集异常图像过滤🔶1-204、🔶1-205
    Args:
        img_paths: 图像路径列表
    Returns:
        valid_paths: 有效路径列表
    """
    valid_paths = []
    for path in img_paths:
        if not os.path.exists(path):
            print(f"警告：图像不存在，跳过：{path}")
            continue
        try:
            with Image.open(path) as img:
                img.verify()  # 验证图像完整性
            valid_paths.append(path)
        except (IOError, SyntaxError) as e:
            print(f"警告：图像损坏，跳过：{path}，错误：{e}")
    return valid_paths