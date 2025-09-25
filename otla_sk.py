import time
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from utils import sort_list_with_unique_index, count_curriculum_anchors  # 复用utils.py的课程统计工具
from sklearn.metrics import adjusted_rand_score  # 确保头部已导入该函数（若未导入需补充）
from collections import defaultdict

# -------------------------- CSANet OTLA-SK全局配置（严格遵循论文） --------------------------
OTLA_SK_CONFIG = {
    "sinkhorn": {
        "error_threshold": 1e-3,  # 论文SK算法迭代终止误差（原1e-1修正，🔶1-256）
        "max_step": 1000,          # 最大迭代步数（避免无限循环）
        "lambda_sk": 2.0           # SK算法的温度系数（论文经验值）
    },
    "curriculum": {
        "optimize_courses": ["moderate", "intricate"]  # 仅优化中等/复杂课程伪标签（🔶1-99）
    },
    "confidence": {
        "min_confidence": 0.5      # 伪标签最小置信度（筛选高置信度样本，🔶1-171）
    }
}

def calculate_pseudo_confidence(
    pred_probs: np.ndarray,  # 模型输出的Softmax概率（shape=[N, C]）
    pseudo_labels: np.ndarray  # 伪标签（shape=[N]）
) -> np.ndarray:
    """
    计算伪标签置信度（论文要求量化置信度，🔶1-171）
    Args:
        pred_probs: 模型输出的类别概率（Softmax后）
        pseudo_labels: 伪标签（每个样本的预测类别）
    Returns:
        confidence: 每个伪标签的置信度（对应类别的概率值，shape=[N]）
    """
    N = len(pseudo_labels)
    confidence = np.zeros(N)
    for i in range(N):
        confidence[i] = pred_probs[i, pseudo_labels[i]]
    return confidence

def cpu_sk_ir_trainloader(
    args, 
    main_net: nn.Module, 
    trainloader: torch.utils.data.DataLoader, 
    tIndex: np.ndarray,  # 红外样本索引（对应trainloader中的红外样本）
    n_class: int,  # 类别数（伪标签最大类别）
    curriculum_mask: Optional[np.ndarray] = None,  # 红外课程掩码（0=plain,1=moderate,2=intricate）
    print_freq: int = 50
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    基于SK算法优化红外伪标签（适配CSANet分课程需求，🔶1-131、🔶1-256）
    Args:
        curriculum_mask: 红外课程掩码（用于筛选需优化的中等/复杂课程样本）
        其他参数与原函数一致
    Returns:
        ir_pseudo_label_op: SK优化后的伪标签（OTLA-SK输出）
        ir_pseudo_label_mp: 最大概率伪标签（原逻辑保留）
        ir_real_label: 真实标签（若有，用于评估；无则为占位符）
        selected_tIndex: 筛选后需优化的红外样本索引
        pseudo_confidence: 伪标签置信度（SK优化后）
        stats: 统计信息（如各课程优化样本数、置信度分布）
    """
    main_net.train()
    device = next(main_net.parameters()).device  # 自动获取模型设备（CPU/GPU）
    n_ir_total = len(tIndex)
    P = np.zeros((n_ir_total, n_class))  # 存储所有红外样本的Softmax概率
    ir_real_label = torch.tensor([]).to(device)
    stats = defaultdict(list)

    # 1. 第一步：提取红外样本的模型预测概率（Softmax后）
    print("=== 开始提取红外样本预测概率 ===")
    with torch.no_grad():
        for batch_idx, (input_rgb, input_ir, label_rgb, label_ir) in enumerate(trainloader):
            t_start = time.time()
            # 数据设备对齐
            input_ir = input_ir.to(device)
            label_ir = label_ir.to(device)

            # 模型前向传播（假设main_net输出为：特征、预测logits、其他输出）
            # 适配CSANet模型输出：modal=2表示红外模态（需与模型定义一致）
            _, p_logits, _ = main_net(input_ir, input_ir, modal=2, train_set=False)
            p_softmax = nn.Softmax(dim=1)(p_logits).cpu().numpy()  # 转为CPU numpy数组

            # 存储当前batch的概率（处理最后一个batch的维度对齐）
            batch_start = batch_idx * args.train_batch_size * args.num_pos
            batch_end = min((batch_idx + 1) * args.train_batch_size * args.num_pos, n_ir_total)
            P[batch_start:batch_end, :] = p_softmax[:batch_end - batch_start, :]

            # 累积真实标签（若有）
            if ir_real_label.numel() == 0:
                ir_real_label = label_ir
            else:
                ir_real_label = torch.cat((ir_real_label, label_ir), dim=0)

            # 打印进度
            if (batch_idx + 1) % print_freq == 0:
                batch_time = time.time() - t_start
                print(f"提取预测概率：[{batch_idx + 1}/{len(trainloader)}]\t"
                      f"单batch耗时：{batch_time:.3f}s\t"
                      f"累计处理样本：{batch_end}/{n_ir_total}")

    # 2. 第二步：按课程筛选需优化的样本（仅中等/复杂课程，🔶1-99）
    print("\n=== 按课程筛选需优化的红外样本 ===")
    # 课程掩码校验与映射（0=plain,1=moderate,2=intricate）
    if curriculum_mask is None:
        # 无课程掩码时，默认优化所有样本（兼容基线模式）
        selected_mask = np.ones(n_ir_total, dtype=bool)
        print("警告：未输入课程掩码，将优化所有红外样本（建议补充课程划分）")
    else:
        # 仅保留中等（1）和复杂（2）课程样本
        course2level = {"moderate": 1, "intricate": 2}
        target_levels = [course2level[course] for course in OTLA_SK_CONFIG["curriculum"]["optimize_courses"]]
        selected_mask = np.isin(curriculum_mask, target_levels)
    
    # 筛选样本索引与概率
    selected_idx = np.where(selected_mask)[0]
    if len(selected_idx) == 0:
        raise RuntimeError(f"无符合条件的样本（需优化课程：{OTLA_SK_CONFIG['curriculum']['optimize_courses']}）")
    
    # 基于tIndex筛选需优化的样本（确保与trainloader索引对应）
    selected_tIndex = tIndex[selected_idx]
    P_selected = P[selected_idx, :]  # 筛选后的概率矩阵（shape=[N_selected, C]）
    n_ir_selected = len(selected_tIndex)
    print(f"需优化的样本数：{n_ir_selected}/{n_ir_total}（课程：{OTLA_SK_CONFIG['curriculum']['optimize_courses']}）")

    # 3. 第三步：按红外样本身份分组（基于tIndex的唯一身份）
    print("\n=== 按身份分组样本 ===")
    # 复用utils.py的sort_list_with_unique_index，获取每个身份的样本索引
    _, unique_last_idx, unique_num, idx_order, unique_list = sort_list_with_unique_index(selected_tIndex)
    n_ir_unique = len(idx_order)  # 唯一身份数
    print(f"唯一身份数：{n_ir_unique}（每个身份平均样本数：{n_ir_selected / n_ir_unique:.1f}）")

    # 计算每个身份的平均概率（降低单样本噪声）
    P_avg = np.zeros((n_ir_unique, n_class))
    for i, idx in enumerate(idx_order):
        # 每个身份的所有样本概率平均
        P_avg[i] = P_selected[unique_list[idx]].mean(axis=0)
        # 统计各身份的样本数
        stats["identity_sample_count"].append(len(unique_list[idx]))

    # 4. 第四步：Sinkhorn-Knopp算法优化伪标签（OTLA核心逻辑）
    print("\n=== 运行Sinkhorn-Knopp算法 ===")
    # 处理数值稳定性（避免0概率导致的log异常）
    eps = 1e-12
    P_avg = np.clip(P_avg, eps, 1.0 - eps)  # 截断概率值
    # 论文公式：PS = (P_avg^T)^lambda_sk（温度系数调整）
    lambda_sk = OTLA_SK_CONFIG["sinkhorn"]["lambda_sk"]
    PS = (P_avg.T) ** lambda_sk  # shape=[C, N_unique]

    # 初始化对偶变量
    alpha = np.ones((n_class, 1)) / n_class  # 类别分布（初始均匀）
    beta = np.ones((n_ir_unique, 1)) / n_ir_unique  # 身份分布（初始均匀）
    inv_K = 1.0 / n_class  # 类别数倒数
    inv_N = 1.0 / n_ir_unique  # 唯一身份数倒数

    # 迭代优化
    err = float("inf")
    step = 0
    t_sk_start = time.time()
    max_step = OTLA_SK_CONFIG["sinkhorn"]["max_step"]
    error_threshold = OTLA_SK_CONFIG["sinkhorn"]["error_threshold"]

    while err > error_threshold and step < max_step:
        # 更新alpha（类别对偶变量）
        alpha = inv_K / (PS @ beta + eps)  # 加eps避免除以0
        # 更新beta（身份对偶变量）
        beta_new = inv_N / (alpha.T @ PS + eps).T
        # 计算误差（beta的相对变化）
        if step % 10 == 0:
            # 避免beta为0导致的nan
            valid_mask = (beta > eps) & (beta_new > eps)
            if valid_mask.sum() > 0:
                err = np.nanmean(np.abs(beta[valid_mask] / beta_new[valid_mask] - 1))
            else:
                err = error_threshold  # 无有效beta时提前终止
            # 记录误差变化
            stats["sinkhorn_error"].append(err)
        
        beta = beta_new
        step += 1

    # SK算法结果统计
    sk_time = time.time() - t_sk_start
    print(f"Sinkhorn-Knopp优化完成：")
    print(f"  迭代步数：{step}/{max_step}")
    print(f"  最终误差：{err:.6f}（阈值：{error_threshold}）")
    print(f"  总耗时：{sk_time:.3f}s")
    stats["sinkhorn_final_error"] = err
    stats["sinkhorn_steps"] = step
    stats["sinkhorn_time"] = sk_time

    # 5. 第五步：生成优化后的伪标签与置信度
    print("\n=== 生成优化后的伪标签 ===")
    # 计算优化后的联合概率矩阵
    PS_opt = PS * np.squeeze(beta)  # 按beta调整
    PS_opt = PS_opt.T * np.squeeze(alpha)  # 按alpha调整
    PS_opt = PS_opt.T  # 恢复shape=[C, N_unique]

    # 生成每个唯一身份的伪标签（取概率最大的类别）
    argmaxes_unique = np.nanargmax(PS_opt, axis=0)  # shape=[N_unique]
    # 映射到所有筛选后的样本（每个身份的所有样本共享同一伪标签）
    ir_pseudo_label_op = np.zeros(n_ir_selected, dtype=int)
    for i, idx in enumerate(idx_order):
        sample_idx = unique_list[idx]  # 当前身份的所有样本索引
        ir_pseudo_label_op[sample_idx] = argmaxes_unique[i]

    # 生成最大概率伪标签（原逻辑保留，用于对比）
    argmaxes_mp = np.nanargmax(P_avg, axis=1)  # 每个唯一身份的最大概率类别
    ir_pseudo_label_mp = np.zeros(n_ir_selected, dtype=int)
    for i, idx in enumerate(idx_order):
        sample_idx = unique_list[idx]
        ir_pseudo_label_mp[sample_idx] = argmaxes_mp[i]

    # 计算伪标签置信度（SK优化后）
    pseudo_confidence = calculate_pseudo_confidence(P_selected, ir_pseudo_label_op)
    # 筛选高置信度伪标签（按论文最小置信度阈值）
    high_conf_mask = pseudo_confidence >= OTLA_SK_CONFIG["confidence"]["min_confidence"]
    stats["high_confidence_ratio"] = high_conf_mask.sum() / n_ir_selected
    print(f"高置信度样本（≥{OTLA_SK_CONFIG['confidence']['min_confidence']}）占比：{stats['high_confidence_ratio']:.2%}")

    # 6. 第六步：整理输出与统计信息
    # 转换为PyTorch张量
    ir_pseudo_label_op = torch.LongTensor(ir_pseudo_label_op)
    ir_pseudo_label_mp = torch.LongTensor(ir_pseudo_label_mp)
    # 真实标签筛选（仅保留需优化的样本）
    ir_real_label = ir_real_label[selected_idx].cpu() if ir_real_label.numel() > 0 else torch.tensor([])

    # 课程统计（若有课程掩码）
    if curriculum_mask is not None:
        selected_course = curriculum_mask[selected_idx]
        for level in [1, 2]:
            course_name = "moderate" if level == 1 else "intricate"
            count = (selected_course == level).sum()
            stats[f"{course_name}_sample_count"] = count
            print(f"{course_name}课程优化样本数：{count}/{n_ir_selected}")

    # 置信度统计
    stats["confidence_mean"] = pseudo_confidence.mean()
    stats["confidence_std"] = pseudo_confidence.std()
    print(f"伪标签置信度：均值={stats['confidence_mean']:.4f}，标准差={stats['confidence_std']:.4f}")

    return (ir_pseudo_label_op, ir_pseudo_label_mp, ir_real_label, 
            selected_tIndex, pseudo_confidence, stats)

def evaluate_pseudo_label(
    pseudo_label: np.ndarray,  # SK优化后的伪标签
    true_label: np.ndarray,    # 真实标签（用于评估）
    confidence: np.ndarray,    # 伪标签置信度
    dataset: str = "sysu",     # 数据集名称（用于日志）
    course: str = "step_i_ir" # 课程/阶段标识（用于日志）
) -> Dict[str, float]:
    """
    评估伪标签质量（论文表VIII核心指标，🔶1-256）
    计算ARI（Adjusted Rand Index）、高置信度样本占比等关键指标
    Args:
        pseudo_label: 优化后的伪标签数组（shape=[N]）
        true_label: 真实标签数组（shape=[N]，需与伪标签长度一致）
        confidence: 伪标签置信度数组（shape=[N]）
        course: 当前课程/阶段（如"step_i_ir"表示Step-I红外）
    Returns:
        eval_stats: 评估统计字典（含ARI、置信度统计等）
    """
    # 1. 校验输入合法性
    if len(pseudo_label) != len(true_label) or len(pseudo_label) != len(confidence):
        raise ValueError(
            f"输入数组长度不匹配：伪标签={len(pseudo_label)}，真实标签={len(true_label)}，置信度={len(confidence)}"
        )
    if len(pseudo_label) == 0:
        raise ValueError("输入数组为空，无法评估伪标签质量")

    # 2. 计算核心指标：ARI（Adjusted Rand Index）
    # ARI范围[-1,1]，1表示完全一致，0表示随机聚类（论文表VIII关键指标）
    ari = adjusted_rand_score(true_label, pseudo_label)

    # 3. 置信度统计（高置信度样本占比、均值、标准差）
    min_confidence = OTLA_SK_CONFIG["confidence"]["min_confidence"]  # 从全局配置获取置信度阈值
    high_conf_mask = confidence >= min_confidence
    high_conf_ratio = high_conf_mask.sum() / len(confidence)  # 高置信度样本占比
    conf_mean = confidence.mean()  # 置信度均值
    conf_std = confidence.std()    # 置信度标准差

    # 4. 高置信度样本的ARI（额外评估高置信度伪标签质量）
    high_conf_ari = 0.0
    if high_conf_mask.sum() > 0:
        high_conf_ari = adjusted_rand_score(
            true_label[high_conf_mask],
            pseudo_label[high_conf_mask]
        )

    # 5. 整理评估结果
    eval_stats = defaultdict(float)
    eval_stats["ARI"] = ari
    eval_stats["high_confidence_ratio"] = high_conf_ratio
    eval_stats["confidence_mean"] = conf_mean
    eval_stats["confidence_std"] = conf_std
    eval_stats["high_conf_ARI"] = high_conf_ari

    # 6. 打印评估日志（适配论文格式）
    print(f"\n=== {dataset} {course} 伪标签质量评估 ===")
    print(f"ARI分数（全局）: {ari:.4f}（1=完全一致，0=随机）")
    print(f"高置信度样本（≥{min_confidence}）:")
    print(f"  占比: {high_conf_ratio:.2%}")
    print(f"  ARI分数: {high_conf_ari:.4f}")
    print(f"置信度统计: 均值={conf_mean:.4f}，标准差={conf_std:.4f}")
    print("="*60)

    return eval_stats