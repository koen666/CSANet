import torch
import torch.optim as optim
from typing import Optional, Dict, List

# -------------------------- CSANet优化器全局配置（严格遵循论文） --------------------------
OPTIM_CONFIG = {
    "train_stage": {
        "step_i_epoch": 20,    # Step-I（单模态聚类）总epoch数（🔶1-210）
        "step_ii_epoch": 40,   # Step-II（跨模态关联）总epoch数
        "total_epoch": 60      # 总训练epoch数（20+40）
    },
    "lr": {
        "base_lr": 3e-4,       # 骨干网络基础学习率（Transformer骨干，🔶1-208）
        "module_lr_ratio": 10, # 新增模块（TBGM/CAP/IPCC）lr为骨干的10倍（🔶1-209）
        "warmup_epoch": 5,     # Step-I前5 epoch线性升温（避免初始lr过低）
        "decay_steps": [40, 50], # Step-II阶梯衰减节点（40/50 epoch，对应总epoch 40=20+20、50=20+30）
        "decay_gamma": 0.1      # 衰减系数（每次衰减为之前的1/10，🔶1-209）
    },
    "optim": {
        "type": "adam",        # 论文使用Adam优化器（参考Transformer常用配置）
        "weight_decay": 5e-4,  # 权重衰减（避免过拟合，🔶1-209）
        "betas": (0.9, 0.999)  # Adam默认betas参数
    }
}

#动态调整学习率
def adjust_learning_rate(
    optimizer: optim.Optimizer,
    current_epoch: int,
    base_lr: float = OPTIM_CONFIG["lr"]["base_lr"],
    module_lr_ratio: int = OPTIM_CONFIG["lr"]["module_lr_ratio"]
) -> float:
    """
    按CSANet分阶段训练调整学习率（🔶1-209、🔶1-210 Algorithm 1）
    Args:
        optimizer: 已初始化的优化器
        current_epoch: 当前训练epoch（1-based，范围1~60）
        base_lr: 骨干网络基础学习率
        module_lr_ratio: 新增模块与骨干的lr比例
    Returns:
        current_base_lr: 当前骨干网络学习率
    """
    # 1. 判断当前训练阶段（Step-I: 1~20 epoch；Step-II: 21~60 epoch）
    step_i_end = OPTIM_CONFIG["train_stage"]["step_i_epoch"]
    if not (1 <= current_epoch <= OPTIM_CONFIG["train_stage"]["total_epoch"]):
        raise ValueError(f"epoch需在1~{OPTIM_CONFIG['train_stage']['total_epoch']}之间，当前为{current_epoch}")
    
    # 2. 计算当前基础学习率（骨干网络）
    if current_epoch <= step_i_end:
        # Step-I：前warmup_epoch线性升温，之后保持base_lr（🔶1-209）
        warmup_epoch = OPTIM_CONFIG["lr"]["warmup_epoch"]
        if current_epoch <= warmup_epoch:
            # 线性升温：lr = base_lr * (current_epoch / warmup_epoch)
            current_base_lr = base_lr * (current_epoch / warmup_epoch)
        else:
            # Step-I后期（6~20 epoch）保持base_lr
            current_base_lr = base_lr
    else:
        # Step-II：按decay_steps阶梯衰减（🔶1-209）
        current_base_lr = base_lr
        decay_steps = OPTIM_CONFIG["lr"]["decay_steps"]
        decay_gamma = OPTIM_CONFIG["lr"]["decay_gamma"]
        # 计算衰减次数（仅在Step-II内判断）
        decay_count = sum(1 for step in decay_steps if current_epoch > step)
        current_base_lr *= (decay_gamma ** decay_count)
    
    # 3. 更新优化器各参数组lr（骨干网络组 *1，新增模块组 *module_lr_ratio）
    # 约定：参数组0为骨干网络，1~N为新增模块（TBGM/CAP/IPCC）
    if len(optimizer.param_groups) == 0:
        raise RuntimeError("优化器未初始化参数组")
    
    # 更新骨干网络参数组（组0）
    optimizer.param_groups[0]["lr"] = current_base_lr
    # 更新新增模块参数组（组1及以后）
    module_lr = current_base_lr * module_lr_ratio
    for i in range(1, len(optimizer.param_groups)):
        optimizer.param_groups[i]["lr"] = module_lr
    
    # 打印当前lr信息（适配论文实验记录）
    print(f"Epoch {current_epoch:2d} | 阶段: {'Step-I' if current_epoch <= step_i_end else 'Step-II'} | "
          f"骨干lr: {current_base_lr:.6f} | 模块lr: {module_lr:.6f}")
    
    return current_base_lr

def select_optimizer(
    args,
    main_net: torch.nn.Module,
    base_lr: float = OPTIM_CONFIG["lr"]["base_lr"],
    module_lr_ratio: int = OPTIM_CONFIG["lr"]["module_lr_ratio"]
) -> optim.Optimizer:
    """
    为CSANet选择优化器，按模块分组配置差异化学习率（🔶1-104 Fig.1、🔶1-208、🔶1-209）
    Args:
        args: 命令行参数（需包含args.optim，默认"adam"）
        main_net: CSANet完整模型（需包含指定子模块）
        base_lr: 骨干网络基础学习率
        module_lr_ratio: 新增模块与骨干的lr比例
    Returns:
        optimizer: 配置完成的Adam优化器
    """
    # 1. 校验优化器类型（论文仅使用Adam）
    if args.optim != OPTIM_CONFIG["optim"]["type"]:
        print(f"警告：论文推荐使用{OPTIM_CONFIG['optim']['type']}优化器，当前为{args.optim}，将自动切换")
        args.optim = OPTIM_CONFIG["optim"]["type"]
    
    # 2. 按CSANet模块分组参数（排除非可训练参数如记忆库）
    # 模块1：Transformer骨干网络（特征提取，如main_net.backbone）
    # 需确保模型定义时骨干网络命名为"backbone"
    if not hasattr(main_net, "backbone"):
        raise AttributeError("CSANet模型需包含'backbone'子模块（Transformer骨干，参考TransReID）")
    backbone_params = list(main_net.backbone.parameters())
    
    # 模块2：TBGM模块（双二分图匹配，如main_net.tbgm）
    if not hasattr(main_net, "tbgm"):
        raise AttributeError("CSANet模型需包含'tbgm'子模块（🔶1-104 Fig.1）")
    tbgm_params = list(main_net.tbgm.parameters())
    
    # 模块3：CAP+IPCC模块（跨课程关联+一致性约束，如main_net.cap、main_net.ipcc）
    cap_ipcc_params = []
    if hasattr(main_net, "cap"):
        cap_ipcc_params.extend(list(main_net.cap.parameters()))
    if hasattr(main_net, "ipcc"):
        cap_ipcc_params.extend(list(main_net.ipcc.parameters()))
    if not cap_ipcc_params:
        raise AttributeError("CSANet模型需包含'cap'和/或'ipcc'子模块（🔶1-104 Fig.1）")
    
    # 3. 构建参数组（骨干组lr=base_lr，其他组lr=base_lr*module_lr_ratio）
    param_groups = [
        # 组0：骨干网络（低lr，避免预训练参数震荡）
        {"params": backbone_params, "lr": base_lr, "weight_decay": OPTIM_CONFIG["optim"]["weight_decay"]},
        # 组1：TBGM模块（高lr，快速收敛）
        {"params": tbgm_params, "lr": base_lr * module_lr_ratio, "weight_decay": OPTIM_CONFIG["optim"]["weight_decay"]},
        # 组2：CAP+IPCC模块（高lr，快速收敛）
        {"params": cap_ipcc_params, "lr": base_lr * module_lr_ratio, "weight_decay": OPTIM_CONFIG["optim"]["weight_decay"]}
    ]
    
    # 4. 初始化Adam优化器（按论文配置）
    optimizer = optim.Adam(
        param_groups,
        betas=OPTIM_CONFIG["optim"]["betas"],
        weight_decay=OPTIM_CONFIG["optim"]["weight_decay"]
    )
    
    # 打印参数分组信息（适配实验记录）
    print(f"优化器初始化完成（Adam）：")
    print(f"- 骨干网络参数数：{sum(p.numel() for p in backbone_params) / 1e6:.2f}M | lr: {base_lr:.6f}")
    print(f"- TBGM模块参数数：{sum(p.numel() for p in tbgm_params) / 1e3:.2f}K | lr: {base_lr * module_lr_ratio:.6f}")
    print(f"- CAP+IPCC模块参数数：{sum(p.numel() for p in cap_ipcc_params) / 1e3:.2f}K | lr: {base_lr * module_lr_ratio:.6f}")
    
    return optimizer

def freeze_modules(
    main_net: torch.nn.Module,
    freeze_backbone: bool = False,
    freeze_tbgm_cap_ipcc: bool = True
) -> None:
    """
    冻结CSANet指定模块（适配分阶段训练，🔶1-210 Algorithm 1）
    Args:
        main_net: CSANet完整模型
        freeze_backbone: 是否冻结骨干网络（Step-II可冻结预训练骨干，仅微调模块）
        freeze_tbgm_cap_ipcc: 是否冻结TBGM/CAP/IPCC（Step-I需冻结，仅训练骨干）
    """
    # 1. 处理骨干网络冻结
    if hasattr(main_net, "backbone"):
        for param in main_net.backbone.parameters():
            param.requires_grad = not freeze_backbone
        print(f"骨干网络状态：{'冻结' if freeze_backbone else '可训练'}")
    
    # 2. 处理TBGM/CAP/IPCC冻结
    modules_to_freeze = []
    if hasattr(main_net, "tbgm"):
        modules_to_freeze.append(main_net.tbgm)
    if hasattr(main_net, "cap"):
        modules_to_freeze.append(main_net.cap)
    if hasattr(main_net, "ipcc"):
        modules_to_freeze.append(main_net.ipcc)
    
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = not freeze_tbgm_cap_ipcc
    print(f"TBGM/CAP/IPCC状态：{'冻结' if freeze_tbgm_cap_ipcc else '可训练'}")


def get_current_training_stage(current_epoch: int) -> str:
    """
    判断当前训练阶段（辅助日志记录，🔶1-210）
    """
    step_i_end = OPTIM_CONFIG["train_stage"]["step_i_epoch"]
    if current_epoch <= step_i_end:
        return "Step-I (Intra-modality Clustering)"
    else:
        return "Step-II (Cross-modality Curriculum Association)"