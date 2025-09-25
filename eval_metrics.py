import numpy as np
from typing import Tuple, List, Optional, Dict

# -------------------------- CSANet评估全局配置（严格遵循论文） --------------------------
EVAL_CONFIG = {
    "sysu": {
        "cameras": {
            "ir_cameras": [3, 6],          # 红外查询相机（SYSU-MM01，🔶1-204）
            "vis_cameras_all": [1, 2, 4, 5],# All-Search可见光画廊相机
            "vis_cameras_indoor": [1, 2]   # Indoor-Search可见光画廊相机
        },
        "max_rank": 20,                   # 论文报告Rank-1~Rank-20，默认取Rank-20（🔶1-206）
        "filter_same_cam": True           # 过滤同身份-同相机样本（论文要求）
    },
    "regdb": {
        "cameras": {
            "visible_cam": 1,              # 可见光相机ID（RegDB，🔶1-205）
            "thermal_cam": 2               # 红外相机ID
        },
        "max_rank": 20,
        "filter_same_cam": True
    },
    "metrics": {
        "report_ranks": [1, 5, 10, 20]    # 论文重点报告的Rank指标（🔶1-206）
    }
}

def format_metrics(
    cmc: np.ndarray, 
    mAP: float, 
    mINP: float, 
    dataset: str = "sysu",
    mode: str = "all"
) -> Dict[str, float]:
    """
    将评估结果格式化为论文报告格式（如Rank-1、mAP、mINP）（🔶1-206）
    Args:
        cmc: CMC曲线数组（shape=[max_rank]）
        mAP: 平均精度
        mINP: 平均逆序精度
        dataset: 数据集名称（"sysu"/"regdb"）
        mode: 测试模式（SYSU用"all"/"indoor"，RegDB用"vis2thermal"/"thermal2vis"）
    Returns:
        格式化的指标字典（便于日志记录与论文表格生成）
    """
    metrics = {}
    # 记录重点Rank指标（论文表I、表II报告Rank-1/5/10/20）
    for rank in EVAL_CONFIG["metrics"]["report_ranks"]:
        if rank <= len(cmc):
            metrics[f"Rank-{rank}"] = cmc[rank-1] * 100  # 转为百分比
        else:
            metrics[f"Rank-{rank}"] = 0.0
    # 记录mAP和mINP（保留4位小数，与论文格式一致）
    metrics["mAP"] = mAP * 100
    metrics["mINP"] = mINP * 100
    # 补充模式信息
    metrics["mode"] = mode
    metrics["dataset"] = dataset
    return metrics


def eval_sysu(
    distmat: np.ndarray, 
    q_pids: np.ndarray, 
    g_pids: np.ndarray, 
    q_camids: np.ndarray, 
    g_camids: np.ndarray, 
    mode: str = "all",  # 测试模式："all"（All-Search）/"indoor"（Indoor-Search）
    max_rank: int = EVAL_CONFIG["sysu"]["max_rank"],
    filter_same_cam: bool = EVAL_CONFIG["sysu"]["filter_same_cam"]
) -> Tuple[np.ndarray, float, float, Dict[str, float]]:
    """
    评估SYSU-MM01数据集（遵循论文测试协议，🔶1-204、🔶1-206）
    Args:
        mode: 测试模式，决定使用的可见光画廊相机
        filter_same_cam: 是否过滤同身份-同相机样本（论文要求为True）
    Returns:
        cmc: CMC曲线数组（shape=[max_rank]）
        mAP: 平均精度
        mINP: 平均逆序精度
        formatted_metrics: 格式化指标字典（含Rank-1/5/10/20、mAP、mINP）
    """
    # 1. 校验输入合法性
    num_q, num_g = distmat.shape
    if num_q == 0 or num_g == 0:
        raise ValueError("距离矩阵为空（查询数={}，画廊数={}），无法评估".format(num_q, num_g))
    if max_rank > num_g:
        max_rank = num_g
        print(f"警告：画廊样本数不足{max_rank}，自动调整为{num_g}")
    if len(q_pids) != num_q or len(g_pids) != num_g:
        raise ValueError("身份数组与距离矩阵维度不匹配")
    if len(q_camids) != num_q or len(g_camids) != num_g:
        raise ValueError("相机ID数组与距离矩阵维度不匹配")

    # 2. 确定当前模式的画廊相机（论文🔶1-204）
    if mode == "all":
        valid_g_camids = EVAL_CONFIG["sysu"]["cameras"]["vis_cameras_all"]
    elif mode == "indoor":
        valid_g_camids = EVAL_CONFIG["sysu"]["cameras"]["vis_cameras_indoor"]
    else:
        raise ValueError(f"SYSU测试模式错误：{mode}，仅支持'all'/'indoor'")
    
    # 过滤画廊中不属于目标相机的样本（仅保留valid_g_camids）
    g_valid_mask = np.isin(g_camids, valid_g_camids)
    g_pids_valid = g_pids[g_valid_mask]
    g_camids_valid = g_camids[g_valid_mask]
    distmat_valid = distmat[:, g_valid_mask]  # 更新距离矩阵（仅保留有效画廊样本）
    num_g_valid = len(g_pids_valid)
    if num_g_valid == 0:
        raise RuntimeError(f"当前模式{mode}下无有效画廊样本（目标相机：{valid_g_camids}）")

    # 3. 初始化评估变量
    indices = np.argsort(distmat_valid, axis=1)  # 按距离升序排序（距离越小越相似）
    matches = (g_pids_valid[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # 匹配标记（1=正确，0=错误）
    
    all_cmc = []          # 所有查询的CMC曲线
    all_AP = []           # 所有查询的AP
    all_INP = []          # 所有查询的INP
    num_valid_q = 0.0     # 有效查询数（至少有一个正确匹配的查询）

    # 4. 逐查询计算指标
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_dist = distmat_valid[q_idx]
        q_indices = indices[q_idx]  # 当前查询的画廊排序索引

        # 过滤条件1：仅保留目标模式的画廊相机（已在步骤2处理）
        # 过滤条件2：剔除同身份-同相机的样本（论文要求，避免虚假匹配）
        if filter_same_cam:
            # 逻辑：同一身份且查询相机ID == 画廊相机ID → 剔除
            same_pid_mask = (g_pids_valid[q_indices] == q_pid)
            same_cam_mask = (g_camids_valid[q_indices] == q_camid)
            remove_mask = same_pid_mask & same_cam_mask
            keep_mask = np.invert(remove_mask)
        else:
            keep_mask = np.ones_like(q_indices, dtype=bool)

        # 提取当前查询的有效匹配标记
        raw_cmc = matches[q_idx][keep_mask]
        # 跳过无正确匹配的查询（不参与指标计算）
        if not np.any(raw_cmc):
            continue

        # 4.1 计算CMC曲线（累计匹配曲线）
        cmc = raw_cmc.cumsum()
        # 修正CMC（超过1的值设为1，避免累计超过100%）
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])  # 仅保留前max_rank个点
        num_valid_q += 1.0

        # 4.2 计算mINP（平均逆序精度，论文表I、表II核心指标）
        # 逻辑：找到最后一个正确匹配的位置，计算该位置的累计匹配率除以位置+1（🔶1-206）
        pos_indices = np.where(raw_cmc == 1)[0]  # 所有正确匹配的索引
        pos_max_idx = np.max(pos_indices)        # 最后一个正确匹配的索引
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)  # INP = 累计匹配率 / 位置+1
        all_INP.append(inp)

        # 4.3 计算AP（平均精度，遵循信息检索标准）
        num_rel = raw_cmc.sum()  # 相关样本数（正确匹配数）
        # 逐位置计算精度并加权（仅对正确匹配位置计算）
        tmp_cmc = raw_cmc.cumsum()  # 累计匹配数
        # 精度 = 累计匹配数 / 当前位置+1（位置从0开始）
        precision = [tmp_cmc[i] / (i + 1.0) for i in range(len(tmp_cmc))]
        precision = np.asarray(precision) * raw_cmc  # 仅保留正确匹配位置的精度
        AP = precision.sum() / num_rel  # AP = 相关位置精度之和 / 相关样本数
        all_AP.append(AP)

    # 5. 校验有效查询数（避免所有查询均无效）
    if num_valid_q == 0:
        raise RuntimeError("所有查询在画廊中无正确匹配，无法计算指标")

    # 6. 计算全局指标（所有有效查询的平均值）
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    avg_cmc = all_cmc.sum(axis=0) / num_valid_q  # 平均CMC曲线
    mAP = np.mean(all_AP)                        # 平均AP
    mINP = np.mean(all_INP)                     # 平均INP

    # 7. 格式化指标（适配论文报告格式）
    formatted_metrics = format_metrics(avg_cmc, mAP, mINP, dataset="sysu", mode=mode)

    return avg_cmc, mAP, mINP, formatted_metrics

def eval_regdb(
    distmat: np.ndarray, 
    q_pids: np.ndarray, 
    g_pids: np.ndarray, 
    q_camids: np.ndarray, 
    g_camids: np.ndarray, 
    mode: str = "vis2thermal",  # 测试模式："vis2thermal"/"thermal2vis"
    max_rank: int = EVAL_CONFIG["regdb"]["max_rank"],
    filter_same_cam: bool = EVAL_CONFIG["regdb"]["filter_same_cam"]
) -> Tuple[np.ndarray, float, float, Dict[str, float]]:
    """
    评估RegDB数据集（遵循论文测试协议，🔶1-205、🔶1-206）
    Args:
        mode: 双向测试模式（"vis2thermal"=可见光查红外；"thermal2vis"=红外查可见光）
        filter_same_cam: 是否过滤同身份-同相机样本（论文要求为True）
    Returns:
        cmc: CMC曲线数组（shape=[max_rank]）
        mAP: 平均精度
        mINP: 平均逆序精度
        formatted_metrics: 格式化指标字典
    """
    # 1. 校验输入合法性
    num_q, num_g = distmat.shape
    if num_q == 0 or num_g == 0:
        raise ValueError(f"距离矩阵为空（查询数={num_q}，画廊数={num_g}），无法评估")
    if max_rank > num_g:
        max_rank = num_g
        print(f"警告：画廊样本数不足{max_rank}，自动调整为{num_g}")
    if len(q_pids) != num_q or len(g_pids) != num_g:
        raise ValueError("身份数组与距离矩阵维度不匹配")
    if len(q_camids) != num_q or len(g_camids) != num_g:
        raise ValueError("相机ID数组与距离矩阵维度不匹配")

    # 2. 确定当前模式的查询/画廊模态（论文🔶1-205）
    if mode == "vis2thermal":
        # 可见光查询（相机1）→ 红外画廊（相机2）
        valid_q_cam = EVAL_CONFIG["regdb"]["cameras"]["visible_cam"]
        valid_g_cam = EVAL_CONFIG["regdb"]["cameras"]["thermal_cam"]
    elif mode == "thermal2vis":
        # 红外查询（相机2）→ 可见光画廊（相机1）
        valid_q_cam = EVAL_CONFIG["regdb"]["cameras"]["thermal_cam"]
        valid_g_cam = EVAL_CONFIG["regdb"]["cameras"]["visible_cam"]
    else:
        raise ValueError(f"RegDB测试模式错误：{mode}，仅支持'vis2thermal'/'thermal2vis'")

    # 过滤查询中不属于目标相机的样本
    q_valid_mask = (q_camids == valid_q_cam)
    q_pids_valid = q_pids[q_valid_mask]
    q_camids_valid = q_camids[q_valid_mask]
    distmat_q_valid = distmat[q_valid_mask]  # 更新距离矩阵（仅保留有效查询）
    num_q_valid = len(q_pids_valid)
    if num_q_valid == 0:
        raise RuntimeError(f"当前模式{mode}下无有效查询样本（目标相机：{valid_q_cam}）")

    # 过滤画廊中不属于目标相机的样本
    g_valid_mask = (g_camids == valid_g_cam)
    g_pids_valid = g_pids[g_valid_mask]
    g_camids_valid = g_camids[g_valid_mask]
    distmat_valid = distmat_q_valid[:, g_valid_mask]  # 最终距离矩阵
    num_g_valid = len(g_pids_valid)
    if num_g_valid == 0:
        raise RuntimeError(f"当前模式{mode}下无有效画廊样本（目标相机：{valid_g_cam}）")

    # 3. 初始化评估变量
    indices = np.argsort(distmat_valid, axis=1)  # 按距离升序排序
    matches = (g_pids_valid[indices] == q_pids_valid[:, np.newaxis]).astype(np.int32)  # 匹配标记

    all_cmc = []          # 所有查询的CMC曲线
    all_AP = []           # 所有查询的AP
    all_INP = []          # 所有查询的INP
    num_valid_q = 0.0     # 有效查询数（至少有一个正确匹配）

    # 4. 逐查询计算指标
    for q_idx in range(num_q_valid):
        q_pid = q_pids_valid[q_idx]
        q_camid = q_camids_valid[q_idx]
        q_indices = indices[q_idx]  # 当前查询的画廊排序索引

        # 过滤同身份-同相机样本（论文要求，避免虚假匹配）
        if filter_same_cam:
            same_pid_mask = (g_pids_valid[q_indices] == q_pid)
            same_cam_mask = (g_camids_valid[q_indices] == q_camid)
            remove_mask = same_pid_mask & same_cam_mask
            keep_mask = np.invert(remove_mask)
        else:
            keep_mask = np.ones_like(q_indices, dtype=bool)

        # 提取有效匹配标记
        raw_cmc = matches[q_idx][keep_mask]
        # 跳过无正确匹配的查询
        if not np.any(raw_cmc):
            continue

        # 4.1 计算CMC曲线
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1  # 修正超过1的累计值
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # 4.2 计算mINP
        pos_indices = np.where(raw_cmc == 1)[0]
        pos_max_idx = np.max(pos_indices)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        # 4.3 计算AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision = [tmp_cmc[i] / (i + 1.0) for i in range(len(tmp_cmc))]
        precision = np.asarray(precision) * raw_cmc
        AP = precision.sum() / num_rel
        all_AP.append(AP)

    # 5. 校验有效查询数
    if num_valid_q == 0:
        raise RuntimeError("所有查询在画廊中无正确匹配，无法计算指标")

    # 6. 计算全局指标
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    avg_cmc = all_cmc.sum(axis=0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    # 7. 格式化指标
    formatted_metrics = format_metrics(avg_cmc, mAP, mINP, dataset="regdb", mode=mode)

    return avg_cmc, mAP, mINP, formatted_metrics

def stat_multirun_metrics(
    metrics_list: List[Dict[str, float]],
    dataset: str = "sysu",
    mode: Optional[str] = None
) -> Dict[str, Tuple[float, float]]:
    """
    统计多轮实验的指标（如10次实验），计算平均值与标准差（论文要求）
    Args:
        metrics_list: 多轮实验的格式化指标字典列表（每轮一个字典）
        dataset: 数据集名称（用于过滤无关结果）
        mode: 测试模式（如"all"，可选，用于过滤特定模式结果）
    Returns:
        统计后的指标字典（key=指标名，value=(平均值, 标准差)，单位%）
    """
    # 过滤目标数据集和模式的结果
    filtered_metrics = []
    for metrics in metrics_list:
        if metrics["dataset"] != dataset:
            continue
        if mode is not None and metrics["mode"] != mode:
            continue
        filtered_metrics.append(metrics)
    if len(filtered_metrics) == 0:
        raise ValueError(f"无符合条件的实验结果（数据集：{dataset}，模式：{mode}）")

    # 提取所有指标名（如Rank-1、mAP、mINP）
    metric_names = [k for k in filtered_metrics[0].keys() if k not in ["dataset", "mode"]]
    stat_results = {}

    # 逐指标计算平均值与标准差
    for metric in metric_names:
        values = [m[metric] for m in filtered_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        stat_results[metric] = (mean_val, std_val)

    return stat_results


def print_paper_style_results(
    stat_results: Dict[str, Tuple[float, float]],
    dataset: str = "sysu",
    mode: str = "all"
) -> None:
    """
    按论文表格格式打印多轮实验结果（如Rank-1: 64.58±0.32%）（🔶1-206）
    """
    print(f"\n{dataset.upper()}数据集 {mode} 模式 10次实验结果（参考论文表I/II格式）：")
    print("-" * 60)
    # 按论文报告顺序打印指标（Rank-1/5/10/20、mAP、mINP）
    report_order = ["Rank-1", "Rank-5", "Rank-10", "Rank-20", "mAP", "mINP"]
    for metric in report_order:
        if metric in stat_results:
            mean, std = stat_results[metric]
            print(f"{metric:8s}: {mean:5.2f}±{std:4.2f}%")
    print("-" * 60)