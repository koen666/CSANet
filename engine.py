import time
import numpy as np
import torch
from torch.autograd import Variable
from typing import Tuple, Dict, Optional, List
from tensorboardX import SummaryWriter  # 新增：导入tensorboardX的SummaryWriter
from utils import AverageMeter, CSANetLogger, get_current_curriculum
from eval_metrics import eval_regdb, eval_sysu, format_metrics
from loss import CSANetTotalLoss
from otla_sk import calculate_pseudo_confidence

# -------------------------- CSANet训练/测试全局配置（严格遵循论文） --------------------------
ENGINE_CONFIG = {
    "train_stage": {
        "step_i_epoch": 20,    # Step-I总epoch数（🔶1-210）
        "step_ii_epoch": 40,   # Step-II总epoch数
        "total_epoch": 60      # 总epoch数
    },
    "loss_weights": {
        "step_i": {"lambda_nce": 1.0},  # Step-I仅单模态NCE损失
        "step_ii": {"lambda_nce": 1.0, "lambda_cc": 1.0, "lambda_ipcc": 0.5}  # Step-II三损失权重（公式21）
    },
    "feature": {
        "feat_dim": 2048       # 特征维度（与Transformer骨干输出一致，🔶1-208）
    },
    "test": {
        "sysu_modes": ["all", "indoor"],  # SYSU-MM01测试模式
        "regdb_modes": ["vis2thermal", "thermal2vis"]  # RegDB测试模式
    }
}

def get_training_stage(epoch: int) -> str:
    """判断当前训练阶段（Step-I/Step-II），适配论文Algorithm 1（🔶1-210）"""
    if epoch <= ENGINE_CONFIG["train_stage"]["step_i_epoch"]:
        return "step_i"
    elif epoch <= ENGINE_CONFIG["train_stage"]["total_epoch"]:
        return "step_ii"
    else:
        raise ValueError(f"epoch超出范围（1~{ENGINE_CONFIG['train_stage']['total_epoch']}），当前为{epoch}")

def trainer(
    args,
    epoch: int,
    main_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trainloader: torch.utils.data.DataLoader,
    total_loss_fn: CSANetTotalLoss,  # CSANet总损失计算器
    logger: Optional[CSANetLogger] = None,
    writer: Optional[SummaryWriter] = None,
    # Step-II专属输入
    curriculum_mask_vis: Optional[np.ndarray] = None,  # 可见光课程掩码
    curriculum_mask_ir: Optional[np.ndarray] = None,   # 红外课程掩码
    cap_mapping: Optional[Dict[str, Dict[int, int]]] = None,  # CAP关联字典（{"vis2ir":..., "ir2vis":...}）
    ref_memory: Optional[torch.Tensor] = None,  # IPCC参考记忆库（简单课程记忆库）
    print_freq: int = 50
) -> Dict[str, float]:
    """
    CSANet训练器：分Step-I/II训练，支持课程学习与核心模块协同（🔶1-210 Algorithm 1）
    Args:
        cap_mapping: Step-II专属，CAP传递的跨模态关联字典（vis2ir: vis_pid→ir_pid；ir2vis: ir_pid→vis_pid）
        ref_memory: Step-II专属，IPCC模块的参考记忆库（简单课程记忆库，shape=[C_ref, feat_dim]）
    Returns:
        train_stats: 训练统计信息（各损失、准确率）
    """
    # 1. 初始化阶段与学习率
    stage = get_training_stage(epoch)
    # 调用修改后的adjust_learning_rate（optimizer.py中定义）调整学习率
    current_lr = adjust_learning_rate(optimizer, current_epoch=epoch)

    # 2. 初始化指标计算器
    metrics = {
        "total_loss": AverageMeter(),
        "nce_loss": AverageMeter(),       # 单模态NCE损失
        "cc_loss": AverageMeter(),        # 跨模态对比损失（Step-II）
        "ipcc_loss": AverageMeter(),      # IPCC一致性损失（Step-II）
        "batch_time": AverageMeter(),
        "data_time": AverageMeter()
    }
    acc_metrics = {
        "nce_acc_vis": 0,  # 可见光NCE准确率
        "nce_acc_ir": 0,   # 红外NCE准确率
        "cc_acc_vis2ir": 0,  # 可见光→红外CC准确率
        "cc_acc_ir2vis": 0,  # 红外→可见光CC准确率
        "ipcc_prob_sim": 0   # IPCC实例-原型概率相似度
    }
    num_samples = {"vis": 0, "ir": 0}

    # 3. 模型切换至训练模式
    main_net.train()
    end = time.time()

    # 4. 迭代训练
    for batch_id, (input_vis, input_ir, label_vis, label_ir) in enumerate(trainloader):
        # 4.1 数据预处理（设备对齐、课程筛选）
        data_time = time.time() - end
        metrics["data_time"].update(data_time)

        # 数据设备对齐
        input_vis = input_vis.cuda()
        input_ir = input_ir.cuda()
        label_vis = label_vis.cuda()
        label_ir = label_ir.cuda()
        B_vis, B_ir = input_vis.size(0), input_ir.size(0)
        num_samples["vis"] += B_vis
        num_samples["ir"] += B_ir

        # Step-II课程筛选：仅保留当前课程的样本（plain/moderate/intricate）
        if stage == "step_ii" and curriculum_mask_vis is not None and curriculum_mask_ir is not None:
            # 获取当前课程阶段（plain/moderate/intricate）
            current_course = get_current_curriculum(epoch)
            course2level = {"plain": 0, "moderate": 1, "intricate": 2}
            target_level = course2level[current_course]

            # 筛选可见光当前课程样本
            vis_mask = (curriculum_mask_vis[num_samples["vis"]-B_vis : num_samples["vis"]] == target_level)
            input_vis = input_vis[vis_mask]
            label_vis = label_vis[vis_mask]
            B_vis = input_vis.size(0)

            # 筛选红外当前课程样本
            ir_mask = (curriculum_mask_ir[num_samples["ir"]-B_ir : num_samples["ir"]] == target_level)
            input_ir = input_ir[ir_mask]
            label_ir = label_ir[ir_mask]
            B_ir = input_ir.size(0)

            # 跳过空batch
            if B_vis == 0 or B_ir == 0:
                end = time.time()
                continue

        # 4.2 模型前向传播（提取特征与伪标签）
        # 适配CSANet模型输出：(vis_feat, ir_feat, vis_pseudo_prob, ir_pseudo_prob)
        # modal=0表示训练模式，输出双模态特征与伪标签概率
        vis_feat, ir_feat, vis_pseudo_prob, ir_pseudo_prob = main_net(
            input_vis, input_ir, modal=0, train_set=True
        )

        # 4.3 准备损失计算输入（按阶段区分）
        # 记忆库准备（Step-I/II均需，从模型获取动态更新的记忆库）
        vis_memory = main_net.vis_memory  # 可见光记忆库（shape=[C_vis, feat_dim]）
        ir_memory = main_net.ir_memory    # 红外记忆库（shape=[C_ir, feat_dim]）
        vis_pid2idx = main_net.vis_pid2idx  # 可见光pid→记忆库索引
        ir_pid2idx = main_net.ir_pid2idx    # 红外pid→记忆库索引

        if stage == "step_i":
            # Step-I：仅计算单模态NCE损失
            loss_inputs = {
                "stage": "step_i",
                "vis_feats": vis_feat, "vis_labels": label_vis,
                "vis_memory": vis_memory, "vis_pid2idx": vis_pid2idx,
                "ir_feats": ir_feat, "ir_labels": label_ir,
                "ir_memory": ir_memory, "ir_pid2idx": ir_pid2idx
            }
        else:
            # Step-II：计算NCE + CC + IPCC损失，需CAP关联字典与参考记忆库
            if cap_mapping is None or ref_memory is None:
                raise ValueError("Step-II训练需传入CAP关联字典（cap_mapping）与IPCC参考记忆库（ref_memory）")
            
            # 准备IPCC模块输入（仅复杂课程样本，此处简化为所有Step-II样本）
            # （实际需按课程掩码筛选复杂样本，此处为示例）
            ipcc_instance_feats = torch.cat([vis_feat, ir_feat], dim=0)  # 复杂实例特征
            # 原型特征：同模态原型（记忆库中对应聚类中心）+ 跨模态原型（CAP关联）
            vis_proto_feats = vis_memory[torch.tensor([vis_pid2idx[pid.item()] for pid in label_vis], device=vis_feat.device)]
            ir_proto_feats = ir_memory[torch.tensor([ir_pid2idx[pid.item()] for pid in label_ir], device=ir_feat.device)]
            ipcc_proto_feats = torch.cat([vis_proto_feats, ir_proto_feats], dim=0)

            loss_inputs = {
                "stage": "step_ii",
                # NCE损失输入（与Step-I一致）
                "vis_feats": vis_feat, "vis_labels": label_vis,
                "vis_memory": vis_memory, "vis_pid2idx": vis_pid2idx,
                "ir_feats": ir_feat, "ir_labels": label_ir,
                "ir_memory": ir_memory, "ir_pid2idx": ir_pid2idx,
                # CC损失输入（CAP关联字典）
                "cap_vis2ir": cap_mapping["vis2ir"], "cap_ir2vis": cap_mapping["ir2vis"],
                # IPCC损失输入
                "ipcc_instance_feats": ipcc_instance_feats,
                "ipcc_proto_feats": ipcc_proto_feats,
                "ref_memory": ref_memory
            }

        # 4.4 计算总损失与分项指标
        total_loss, loss_metrics = total_loss_fn(**loss_inputs)

        # 4.5 反向传播与优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 4.6 更新指标与统计
        # 更新损失指标
        metrics["total_loss"].update(total_loss.item(), B_vis + B_ir)
        metrics["nce_loss"].update(
            loss_metrics["vis_nce_loss"] + loss_metrics["ir_nce_loss"],
            B_vis + B_ir
        )
        if stage == "step_ii":
            metrics["cc_loss"].update(loss_metrics["cc_loss"], B_vis + B_ir)
            metrics["ipcc_loss"].update(loss_metrics["ipcc_loss"], B_vis + B_ir)
            # 更新CC准确率与IPCC概率相似度
            acc_metrics["cc_acc_vis2ir"] += loss_metrics["vis2ir_acc"] * B_vis
            acc_metrics["cc_acc_ir2vis"] += loss_metrics["ir2vis_acc"] * B_ir
            acc_metrics["ipcc_prob_sim"] += loss_metrics["ipcc_prob_sim"] * (B_vis + B_ir)

        # 更新NCE准确率
        acc_metrics["nce_acc_vis"] += loss_metrics["vis_nce_acc"] * B_vis
        acc_metrics["nce_acc_ir"] += loss_metrics["ir_nce_acc"] * B_ir

        # 更新时间指标
        metrics["batch_time"].update(time.time() - end)
        end = time.time()

        # 4.7 打印训练日志
        if batch_id % print_freq == 0:
            log_msg = f"Epoch: [{epoch}/{ENGINE_CONFIG['train_stage']['total_epoch']}] " \
                      f"Stage: {stage} " \
                      f"Batch: [{batch_id}/{len(trainloader)}] " \
                      f"LR: {current_lr:.6f} " \
                      f"BatchTime: {metrics['batch_time'].val:.3f}({metrics['batch_time'].avg:.3f}) " \
                      f"TotalLoss: {metrics['total_loss'].val:.4f}({metrics['total_loss'].avg:.4f}) " \
                      f"NCELoss: {metrics['nce_loss'].val:.4f}({metrics['nce_loss'].avg:.4f}) "
            if stage == "step_ii":
                log_msg += f"CCLoss: {metrics['cc_loss'].val:.4f}({metrics['cc_loss'].avg:.4f}) " \
                           f"IPCCLoss: {metrics['ipcc_loss'].val:.4f}({metrics['ipcc_loss'].avg:.4f}) "
            # 打印准确率
            log_msg += f"NCELossVisAcc: {acc_metrics['nce_acc_vis']/num_samples['vis']:.4f} " \
                       f"NCELossIrAcc: {acc_metrics['nce_acc_ir']/num_samples['ir']:.4f} "
            if stage == "step_ii":
                log_msg += f"CCVis2IrAcc: {acc_metrics['cc_acc_vis2ir']/num_samples['vis']:.4f} " \
                           f"CCIr2VisAcc: {acc_metrics['cc_acc_ir2vis']/num_samples['ir']:.4f} "
            print(log_msg)
            if logger is not None:
                logger.write(log_msg + "\n")
                logger.flush()

    # 5. 整理训练统计信息（计算平均准确率）
    train_stats = {
        "epoch": epoch,
        "stage": stage,
        "lr": current_lr,
        "total_loss": metrics["total_loss"].avg,
        "nce_loss": metrics["nce_loss"].avg,
        "nce_acc_vis": acc_metrics["nce_acc_vis"] / num_samples["vis"],
        "nce_acc_ir": acc_metrics["nce_acc_ir"] / num_samples["ir"]
    }
    if stage == "step_ii":
        train_stats.update({
            "cc_loss": metrics["cc_loss"].avg,
            "ipcc_loss": metrics["ipcc_loss"].avg,
            "cc_acc_vis2ir": acc_metrics["cc_acc_vis2ir"] / num_samples["vis"],
            "cc_acc_ir2vis": acc_metrics["cc_acc_ir2vis"] / num_samples["ir"],
            "ipcc_prob_sim": acc_metrics["ipcc_prob_sim"] / (num_samples["vis"] + num_samples["ir"])
        })

    # 6. 写入TensorBoard（若有）
    if writer is not None:
        writer.add_scalar(f"{stage}/LR", current_lr, epoch)
        writer.add_scalar(f"{stage}/TotalLoss", train_stats["total_loss"], epoch)
        writer.add_scalar(f"{stage}/NCELoss", train_stats["nce_loss"], epoch)
        writer.add_scalar(f"{stage}/NCEAccVis", train_stats["nce_acc_vis"], epoch)
        writer.add_scalar(f"{stage}/NCEAccIr", train_stats["nce_acc_ir"], epoch)
        if stage == "step_ii":
            writer.add_scalar(f"{stage}/CCLoss", train_stats["cc_loss"], epoch)
            writer.add_scalar(f"{stage}/IPCCLoss", train_stats["ipcc_loss"], epoch)
            writer.add_scalar(f"{stage}/CCAccVis2Ir", train_stats["cc_acc_vis2ir"], epoch)
            writer.add_scalar(f"{stage}/CCAccIr2Vis", train_stats["cc_acc_ir2vis"], epoch)

    return train_stats


def tester(
    args,
    epoch: int,
    main_net: torch.nn.Module,
    test_loader: Dict[str, torch.utils.data.DataLoader],  # 测试加载器字典（含query/gall）
    test_info: Dict[str, np.ndarray],  # 测试信息（query_pids/gall_pids/query_cams/gall_cams）
    dataset: str = "sysu",
    test_mode: str = "all",  # SYSU用"all"/"indoor"，RegDB用"vis2thermal"/"thermal2vis"
    feat_dim: int = ENGINE_CONFIG["feature"]["feat_dim"],
    logger: Optional[CSANetLogger] = None,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[np.ndarray, float, float, Dict[str, float]]:
    """
    CSANet测试器：适配SYSU-MM01/RegDB的论文测试协议（🔶1-204、🔶1-205）
    Args:
        test_loader: 测试加载器字典，含"query_loader"（查询加载器）和"gall_loader"（画廊加载器）
        test_info: 测试信息字典，含"query_pids"/"gall_pids"/"query_cams"/"gall_cams"
        test_mode: 测试模式（需与数据集匹配）
    Returns:
        cmc: CMC曲线数组
        mAP: 平均精度
        mINP: 平均逆序精度
        test_stats: 格式化测试统计（含Rank-1/5/10/20、mAP、mINP）
    """
    # 1. 校验输入合法性
    required_keys = ["query_loader", "gall_loader"]
    if not all(key in test_loader for key in required_keys):
        raise KeyError(f"test_loader需包含{required_keys}键")
    required_info = ["query_pids", "gall_pids", "query_cams", "gall_cams"]
    if not all(key in test_info for key in required_info):
        raise KeyError(f"test_info需包含{required_info}键")
    
    query_loader = test_loader["query_loader"]
    gall_loader = test_loader["gall_loader"]
    query_pids, gall_pids = test_info["query_pids"], test_info["gall_pids"]
    query_cams, gall_cams = test_info["query_cams"], test_info["gall_cams"]

    # 校验特征维度
    if feat_dim != ENGINE_CONFIG["feature"]["feat_dim"]:
        print(f"警告：特征维度{feat_dim}与论文默认{ENGINE_CONFIG['feature']['feat_dim']}不一致，可能影响评估")

    # 2. 模型切换至评估模式
    main_net.eval()
    print(f"\n=== 开始测试（数据集：{dataset}，模式：{test_mode}，Epoch：{epoch}）===")

    # 3. 提取画廊特征
    print("Step 1/3：提取画廊特征...")
    ngall = len(gall_pids)
    gall_feat = np.zeros((ngall, feat_dim), dtype=np.float32)
    ptr = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (input_gall, label_gall) in enumerate(gall_loader):
            batch_size = input_gall.size(0)
            input_gall = Variable(input_gall.cuda())
            
            # 确定画廊模态（SYSU：画廊为可见光；RegDB：按模式区分）
            if dataset == "sysu":
                # SYSU测试：画廊固定为可见光（modal=1）
                modal_gall = 1
            else:  # regdb
                # RegDB测试：vis2thermal→画廊为红外（modal=2）；thermal2vis→画廊为可见光（modal=1）
                modal_gall = 2 if test_mode == "vis2thermal" else 1
            
            # 模型提取特征（仅输出特征，不输出其他参数）
            feat_gall = main_net(input_gall, input_gall, modal=modal_gall, train_set=False)
            # 校验特征维度
            if feat_gall.size(1) != feat_dim:
                raise RuntimeError(f"画廊特征维度{feat_gall.size(1)}与预期{feat_dim}不匹配")
            
            # 存储特征
            gall_feat[ptr:ptr+batch_size, :] = feat_gall.detach().cpu().numpy()
            ptr += batch_size

    gall_extract_time = time.time() - t_start
    print(f"画廊特征提取完成：{ngall}个样本，耗时{gall_extract_time:.3f}s")

    # 4. 提取查询特征
    print("Step 2/3：提取查询特征...")
    nquery = len(query_pids)
    query_feat = np.zeros((nquery, feat_dim), dtype=np.float32)
    ptr = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (input_query, label_query) in enumerate(query_loader):
            batch_size = input_query.size(0)
            input_query = Variable(input_query.cuda())
            
            # 确定查询模态（SYSU：查询固定为红外（modal=2）；RegDB：按模式区分）
            if dataset == "sysu":
                modal_query = 2
            else:  # regdb
                modal_query = 1 if test_mode == "vis2thermal" else 2
            
            # 模型提取特征
            feat_query = main_net(input_query, input_query, modal=modal_query, train_set=False)
            if feat_query.size(1) != feat_dim:
                raise RuntimeError(f"查询特征维度{feat_query.size(1)}与预期{feat_dim}不匹配")
            
            # 存储特征
            query_feat[ptr:ptr+batch_size, :] = feat_query.detach().cpu().numpy()
            ptr += batch_size

    query_extract_time = time.time() - t_start
    print(f"查询特征提取完成：{nquery}个样本，耗时{query_extract_time:.3f}s")

    # 5. 计算距离矩阵与评估
    print("Step 3/3：计算距离矩阵与评估...")
    t_start = time.time()

    # 计算欧式距离矩阵（论文中距离计算统一用欧式距离，🔶1-131、🔶1-187）
    distmat = compute_euclidean_distance(query_feat, gall_feat)

    # 按数据集调用对应评估函数
    if dataset == "sysu":
        # SYSU-MM01评估：需传递相机ID，区分测试模式
        cmc, mAP, mINP = eval_sysu(
            distmat=distmat,
            q_pids=query_pids,
            g_pids=gall_pids,
            q_camids=query_cams,
            g_camids=gall_cams,
            mode=test_mode
        )
    elif dataset == "regdb":
        # RegDB评估：需传递相机ID（动态适配双向模式）
        cmc, mAP, mINP = eval_regdb(
            distmat=distmat,
            q_pids=query_pids,
            g_pids=gall_pids,
            q_camids=query_cams,
            g_camids=gall_cams,
            mode=test_mode
        )
    else:
        raise ValueError(f"不支持的数据集：{dataset}，仅支持'sysu'/'regdb'")

    eval_time = time.time() - t_start
    print(f"评估完成：耗时{eval_time:.3f}s")

    # 6. 格式化测试结果（适配论文报告格式）
    test_stats = format_metrics(
        cmc=cmc,
        mAP=mAP,
        mINP=mINP,
        dataset=dataset,
        mode=test_mode
    )

    # 7. 打印与记录测试日志
    log_msg = f"\n=== 测试结果（Epoch：{epoch}，数据集：{dataset}，模式：{test_mode}）==="
    log_msg += f"\nRank-1: {test_stats['Rank-1']:.2f}% | Rank-5: {test_stats['Rank-5']:.2f}% " \
               f"| Rank-10: {test_stats['Rank-10']:.2f}% | Rank-20: {test_stats['Rank-20']:.2f}%"
    log_msg += f"\nmAP: {test_stats['mAP']:.2f}% | mINP: {test_stats['mINP']:.2f}%"
    log_msg += f"\n特征提取总耗时：{gall_extract_time + query_extract_time:.3f}s | 评估耗时：{eval_time:.3f}s"
    print(log_msg)
    if logger is not None:
        logger.write(log_msg + "\n")
        logger.flush()

    # 8. 写入TensorBoard（若有）
    if writer is not None:
        writer.add_scalar(f"test_{dataset}_{test_mode}/Rank-1", test_stats["Rank-1"], epoch)
        writer.add_scalar(f"test_{dataset}_{test_mode}/mAP", test_stats["mAP"], epoch)
        writer.add_scalar(f"test_{dataset}_{test_mode}/mINP", test_stats["mINP"], epoch)

    return cmc, mAP, mINP, test_stats


def compute_euclidean_distance(query_feat: np.ndarray, gall_feat: np.ndarray) -> np.ndarray:
    """
    计算欧式距离矩阵（论文中所有距离计算均用欧式距离，🔶1-131、🔶1-187）
    优化实现：基于矩阵分解，避免广播导致的内存溢出
    """
    # 公式：||a - b||² = ||a||² + ||b||² - 2abᵀ，开根号后为欧式距离
    eps = 1e-12  # 避免数值不稳定
    query_sq = np.sum(query_feat ** 2, axis=1, keepdims=True)  # (nquery, 1)
    gall_sq = np.sum(gall_feat ** 2, axis=1, keepdims=True)    # (ngall, 1)
    dot_product = np.matmul(query_feat, gall_feat.T)           # (nquery, ngall)
    # 计算距离并确保非负（避免浮点误差导致的负值）
    dist_sq = query_sq + gall_sq.T - 2 * dot_product
    dist_sq = np.maximum(dist_sq, eps)  # 截断负值
    distmat = np.sqrt(dist_sq)
    return distmat

def csanet_train_pipeline(
    args,
    main_net: torch.nn.Module,
    train_loaders: Dict[str, torch.utils.data.DataLoader],  # 训练加载器（step_i/step_ii）
    test_loaders: Dict[str, Dict[str, torch.utils.data.DataLoader]],  # 测试加载器（按数据集-模式区分）
    test_infos: Dict[str, Dict[str, np.ndarray]],  # 测试信息（按数据集-模式区分）
    optimizer: torch.optim.Optimizer,
    total_loss_fn: CSANetTotalLoss,
    logger: CSANetLogger,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, List[Dict[str, float]]]:
    """
    CSANet完整训练流程（Step-I→Step-II），遵循论文Algorithm 1（🔶1-210）
    Returns:
        all_stats: 所有训练/测试统计信息（用于后续分析与论文绘图）
    """
    all_stats = {"train": [], "test": []}
    best_rank1 = {"sysu_all": 0.0, "sysu_indoor": 0.0, "regdb_vis2thermal": 0.0, "regdb_thermal2vis": 0.0}

    # 初始化Step-I/II专属组件（如CAP模块、IPCC模块、记忆库）
    # （此处简化，实际需初始化main_net中的vis_memory/ir_memory、cap模块、ipcc模块）
    main_net.init_memory(feat_dim=ENGINE_CONFIG["feature"]["feat_dim"])  # 初始化记忆库
    cap_mapping = {"vis2ir": {}, "ir2vis": {}}  # CAP关联字典（Step-II动态更新）
    ref_memory = None  # IPCC参考记忆库（Step-II从Step-I简单课程记忆库获取）

    # 迭代训练（总epoch：60）
    for epoch in range(1, ENGINE_CONFIG["train_stage"]["total_epoch"] + 1):
        stage = get_training_stage(epoch)
        print(f"\n===================== Epoch {epoch}/{ENGINE_CONFIG['train_stage']['total_epoch']}（{stage}）=====================")

        # 1. 训练阶段
        train_loader = train_loaders[stage]
        # Step-II需准备课程掩码与CAP关联字典（此处简化为动态获取）
        if stage == "step_ii":
            # 从模型获取课程掩码（TBGM模块输出）
            curriculum_mask_vis = main_net.tbgm.curriculum_mask_vis
            curriculum_mask_ir = main_net.tbgm.curriculum_mask_ir
            # 从CAP模块获取关联字典
            cap_mapping = main_net.cap.get_mapping()
            # 从Step-I记忆库获取参考记忆库（简单课程记忆库）
            ref_memory = main_net.vis_memory  # 简化为可见光简单课程记忆库
        else:
            curriculum_mask_vis = None
            curriculum_mask_ir = None

        # 调用训练器
        train_stats = trainer(
            args=args,
            epoch=epoch,
            main_net=main_net,
            optimizer=optimizer,
            trainloader=train_loader,
            total_loss_fn=total_loss_fn,
            logger=logger,
            writer=writer,
            curriculum_mask_vis=curriculum_mask_vis,
            curriculum_mask_ir=curriculum_mask_ir,
            cap_mapping=cap_mapping,
            ref_memory=ref_memory,
            print_freq=args.print_freq
        )
        all_stats["train"].append(train_stats)

        # 2. 测试阶段（每5 epoch测试一次，或Step-II全测，适配论文实验）
        if epoch % 5 == 0 or epoch == ENGINE_CONFIG["train_stage"]["total_epoch"]:
            test_stats_epoch = {"epoch": epoch, "stage": stage, "results": {}}
            # 遍历所有数据集与测试模式
            for dataset in test_loaders.keys():
                for test_mode in test_loaders[dataset].keys():
                    cmc, mAP, mINP, test_stats = tester(
                        args=args,
                        epoch=epoch,
                        main_net=main_net,
                        test_loader=test_loaders[dataset][test_mode],
                        test_info=test_infos[dataset][test_mode],
                        dataset=dataset,
                        test_mode=test_mode,
                        logger=logger,
                        writer=writer
                    )
                    test_stats_epoch["results"][f"{dataset}_{test_mode}"] = test_stats

                    # 记录最佳结果
                    key = f"{dataset}_{test_mode}"
                    if test_stats["Rank-1"] > best_rank1[key]:
                        best_rank1[key] = test_stats["Rank-1"]
                        # 保存最佳模型（此处简化，实际需调用torch.save）
                        print(f"=== 最佳{key}模型更新：Rank-1从{best_rank1[key]:.2f}%提升至{test_stats['Rank-1']:.2f}% ===")

            all_stats["test"].append(test_stats_epoch)

    # 打印最终最佳结果
    print("\n===================== 训练完成：最佳结果汇总 =====================")
    for key, rank1 in best_rank1.items():
        print(f"{key} 最佳Rank-1: {rank1:.2f}%")
    if logger is not None:
        logger.write("\n最佳结果汇总：\n")
        for key, rank1 in best_rank1.items():
            logger.write(f"{key}: Rank-1={rank1:.2f}%\n")
        logger.flush()

    return all_stats