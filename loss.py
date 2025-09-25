import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# -------------------------- CSANet损失函数全局配置（严格遵循论文） --------------------------
LOSS_CONFIG = {
    "temperature": 0.05,        # 所有对比损失的温度系数（🔶1-133、🔶1-182）
    "loss_weights": {
        "lambda_nce": 1.0,      # Step-I单模态对比损失权重（🔶1-133）
        "lambda_cc": 1.0,       # Step-II跨模态对比损失权重（🔶1-183）
        "lambda_ipcc": 0.5      # IPCC一致性损失权重（🔶1-197）
    },
    "triplet": {
        "margin": 0.3           # 基线TripletLoss的margin（仅用于对比实验）
    }
}

def normalize(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """
    特征归一化（论文中所有相似度计算前均需归一化，🔶1-133、🔶1-187）
    Args:
        x: 输入特征（shape=[B, d]或[C, d]，B=batch size，C=聚类数）
        axis: 归一化维度
    Returns:
        单位长度归一化后的特征
    """
    x = x / (torch.norm(x, p=2, dim=axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss(nn.Module):
    """
    基线Triplet损失（仅用于对比实验，论文中Step-I/II不使用）
    参考：Hermans et al. In Defense of the Triplet Loss for Person Re-Identification.
    适配论文中margin=0.3的设定（🔶1-212 对比实验）
    """
    def __init__(self, margin: float = LOSS_CONFIG["triplet"]["margin"]):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, int]:
        n = inputs.size(0)
        # 计算 pairwise 欧氏距离
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t() - 2 * torch.mm(inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # 数值稳定性处理

        # 挖掘 hardest positive/negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # Hardest positive: 同一身份中距离最大的样本
            ap = dist[i][mask[i]].max().unsqueeze(0)
            # Hardest negative: 不同身份中距离最小的样本
            an = dist[i][~mask[i]].min().unsqueeze(0)
            dist_ap.append(ap)
            dist_an.append(an)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # 计算 ranking loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # 计算准确率（dist_an >= dist_ap 为正确）
        correct = torch.ge(dist_an, dist_ap).sum().item()

        return loss, correct


class PredictionAlignmentLoss(nn.Module):
    """
    废弃：该损失与CSANet论文逻辑不符，仅保留代码用于历史对比
    论文中使用CrossModalityContrastiveLoss替代该损失实现跨模态对比
    """
    def __init__(self, lambda_vr=0.1, lambda_rv=0.5):
        super(PredictionAlignmentLoss, self).__init__()
        print("警告：PredictionAlignmentLoss已废弃，CSANet推荐使用CrossModalityContrastiveLoss")
        self.lambda_vr = lambda_vr
        self.lambda_rv = lambda_rv

    def forward(self, x_rgb, x_ir):
        # 原逻辑保留，不做修改
        sim_rgbtoir = torch.mm(normalize(x_rgb), normalize(x_ir).t())
        sim_irtorgb = torch.mm(normalize(x_ir), normalize(x_rgb).t())
        sim_irtoir = torch.mm(normalize(x_ir), normalize(x_ir).t())

        sim_rgbtoir = F.softmax(sim_rgbtoir, dim=1)
        sim_irtorgb = F.softmax(sim_irtorgb, dim=1)
        sim_irtoir = F.softmax(sim_irtoir, dim=1)

        KL_criterion = nn.KLDivLoss(reduction="batchmean")

        x_rgbtoir = torch.mm(sim_rgbtoir, x_ir)
        x_irtorgb = torch.mm(sim_irtorgb, x_rgb)
        x_irtoir = torch.mm(sim_irtoir, x_ir)

        x_rgb_s = F.softmax(x_rgb, dim=1)
        x_rgbtoir_ls = F.log_softmax(x_rgbtoir, dim=1)
        x_irtorgb_s = F.softmax(x_irtorgb, dim=1)
        x_irtoir_ls = F.log_softmax(x_irtoir, dim=1)

        loss_rgbtoir = KL_criterion(x_rgbtoir_ls, x_rgb_s)
        loss_irtorgb = KL_criterion(x_irtoir_ls, x_irtorgb_s)

        loss = self.lambda_vr * loss_rgbtoir + self.lambda_rv * loss_irtorgb
        return loss, sim_rgbtoir, sim_irtorgb


class ClusterNCELoss(nn.Module):
    """
    Step-I单模态对比损失（ClusterNCE）：基于记忆库的聚类级对比损失（🔶1-133 公式5）
    要求“同一伪身份样本特征与记忆库中对应聚类中心距离更近，与其他聚类中心距离更远”
    """
    def __init__(self, temperature: float = LOSS_CONFIG["temperature"]):
        super(ClusterNCELoss, self).__init__()
        self.tau = temperature  # 温度系数

    def forward(
        self,
        instance_feats: torch.Tensor,  # 单模态样本特征（shape=[B, d]，B=batch size）
        instance_labels: torch.Tensor, # 样本伪标签（shape=[B]）
        memory: torch.Tensor,          # 单模态记忆库（shape=[C, d]，C=聚类数，存储聚类中心）
        memory_pid2idx: dict           # 伪身份到记忆库索引的映射（{pid: idx}）
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            memory: 由build_modal_memory函数生成（🔶1-132 公式3）
            memory_pid2idx: 确保样本伪标签能映射到记忆库对应行
        Returns:
            nce_loss: 单模态对比损失值
            acc: 对比准确率（正样本相似度>负样本相似度的比例）
        """
        # 1. 特征归一化（论文要求）
        instance_feats = normalize(instance_feats)
        memory = normalize(memory)

        # 2. 获取每个样本对应的正样本（记忆库中该伪身份的聚类中心）
        batch_size = instance_feats.size(0)
        pos_memory_idx = torch.tensor([memory_pid2idx[pid.item()] for pid in instance_labels], 
                                     device=instance_feats.device)
        pos_feats = memory[pos_memory_idx]  # shape=[B, d]

        # 3. 计算相似度（样本与所有记忆库聚类中心的余弦相似度）
        sim_matrix = torch.mm(instance_feats, memory.t())  # shape=[B, C]
        sim_pos = torch.diag(torch.mm(instance_feats, pos_feats.t()))  # 正样本相似度（shape=[B]）
        sim_pos = sim_pos.unsqueeze(1).expand(batch_size, sim_matrix.size(1))  # shape=[B, C]

        # 4. 计算ClusterNCE损失（公式5）
        logits = (sim_matrix - sim_pos) / self.tau  # 突出正样本与负样本的差异
        labels = pos_memory_idx  # 标签为正样本在记忆库中的索引
        nce_loss = F.cross_entropy(logits, labels)

        # 5. 计算对比准确率（正样本相似度>所有负样本相似度的样本数/总样本数）
        max_neg_sim, _ = torch.max(sim_matrix - sim_pos, dim=1)  # 最大负样本相对相似度（应<0）
        acc = (max_neg_sim < 0).sum().item() / batch_size

        return nce_loss, acc


class CrossModalityContrastiveLoss(nn.Module):
    """
    Step-II跨模态对比损失（\(\mathcal{L}_{CC}\)）：基于CAP关联字典的双模态对比损失（🔶1-183 公式12）
    支持“可见光→红外”与“红外→可见光”两个方向的对比约束
    """
    def __init__(self, temperature: float = LOSS_CONFIG["temperature"]):
        super(CrossModalityContrastiveLoss, self).__init__()
        self.tau = temperature

    def _single_direction_loss(
        self,
        src_feats: torch.Tensor,    # 源模态特征（如可见光，shape=[B, d]）
        src_labels: torch.Tensor,   # 源模态伪标签（shape=[B]）
        tgt_memory: torch.Tensor,   # 目标模态记忆库（如红外，shape=[C_tgt, d]）
        cap_mapping: dict           # CAP关联字典（src_pid → tgt_pid）（🔶1-174）
    ) -> Tuple[torch.Tensor, float]:
        """
        单方向跨模态对比损失（如可见光→红外）
        """
        # 1. 特征归一化
        src_feats = normalize(src_feats)
        tgt_memory = normalize(tgt_memory)

        # 2. 映射源模态伪标签到目标模态伪标签（通过CAP关联字典）
        tgt_labels = torch.tensor([cap_mapping[pid.item()] for pid in src_labels], 
                                 device=src_feats.device)
        # 映射目标模态伪标签到记忆库索引（假设tgt_memory按pid排序，索引=pid）
        tgt_memory_idx = tgt_labels  # 若记忆库索引与pid不一致，需传入tgt_pid2idx映射

        # 3. 计算相似度（源特征与目标记忆库的余弦相似度）
        sim_matrix = torch.mm(src_feats, tgt_memory.t())  # shape=[B, C_tgt]
        sim_pos = torch.gather(sim_matrix, dim=1, index=tgt_memory_idx.unsqueeze(1))  # 正样本相似度（shape=[B,1]）

        # 4. 计算单方向损失
        logits = sim_matrix / self.tau
        loss = F.cross_entropy(logits, tgt_memory_idx)

        # 5. 计算单方向准确率
        acc = (torch.argmax(logits, dim=1) == tgt_memory_idx).sum().item() / src_feats.size(0)
        return loss, acc

    def forward(
        self,
        vis_feats: torch.Tensor,    # 可见光特征（shape=[B, d]）
        vis_labels: torch.Tensor,   # 可见光伪标签（shape=[B]）
        ir_feats: torch.Tensor,     # 红外特征（shape=[B, d]）
        ir_labels: torch.Tensor,    # 红外伪标签（shape=[B]）
        vis_memory: torch.Tensor,   # 可见光记忆库（shape=[C_vis, d]）
        ir_memory: torch.Tensor,    # 红外记忆库（shape=[C_ir, d]）
        cap_vis2ir: dict,           # CAP关联字典（vis_pid → ir_pid）
        cap_ir2vis: dict            # CAP关联字典（ir_pid → vis_pid）
    ) -> Tuple[torch.Tensor, float, float]:
        """
        双向跨模态对比损失（可见光→红外 + 红外→可见光）
        Returns:
            cc_loss: 总跨模态对比损失
            vis2ir_acc: 可见光→红外方向准确率
            ir2vis_acc: 红外→可见光方向准确率
        """
        # 1. 计算可见光→红外方向损失
        vis2ir_loss, vis2ir_acc = self._single_direction_loss(
            src_feats=vis_feats,
            src_labels=vis_labels,
            tgt_memory=ir_memory,
            cap_mapping=cap_vis2ir
        )

        # 2. 计算红外→可见光方向损失
        ir2vis_loss, ir2vis_acc = self._single_direction_loss(
            src_feats=ir_feats,
            src_labels=ir_labels,
            tgt_memory=vis_memory,
            cap_mapping=cap_ir2vis
        )

        # 3. 总损失（双向损失相加，论文公式12）
        cc_loss = vis2ir_loss + ir2vis_loss
        return cc_loss, vis2ir_acc, ir2vis_acc


class IPCCConsistencyLoss(nn.Module):
    """
    IPCC模块一致性损失（\(\mathcal{L}_{IPCC}\)）：基于KL散度的实例-原型概率响应约束（🔶1-194 公式19）
    """
    def __init__(
        self,
        temperature: float = LOSS_CONFIG["temperature"],
        weight: float = LOSS_CONFIG["loss_weights"]["lambda_ipcc"]
    ):
        super(IPCCConsistencyLoss, self).__init__()
        self.tau = temperature
        self.weight = weight
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")  # KL散度损失

    def compute_prob_response(
        self,
        feats: torch.Tensor,    # 实例/原型特征（shape=[B, d]或[C, d]）
        ref_memory: torch.Tensor # 参考记忆库（简单课程记忆库，shape=[C_ref, d]）（🔶1-187）
    ) -> torch.Tensor:
        """
        计算概率响应（论文公式14、16，🔶1-187）
        """
        feats = normalize(feats)
        ref_memory = normalize(ref_memory)
        sim = torch.mm(feats, ref_memory.t())  # 余弦相似度
        prob = F.softmax(sim / self.tau, dim=1)  # 概率响应
        return prob

    def forward(
        self,
        instance_feats: torch.Tensor,  # 复杂样本实例特征（shape=[B, d]）
        prototype_feats: torch.Tensor, # 对应原型特征（shape=[B, d]）
        ref_memory: torch.Tensor       # 参考记忆库（简单课程记忆库，🔶1-187）
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            prototype_feats: 同模态原型+跨模态原型的拼接（或分别计算后求和）
        Returns:
            ipcc_loss: IPCC一致性损失（加权后）
            prob_sim: 实例与原型概率响应的平均余弦相似度（评估一致性）
        """
        # 1. 计算实例与原型的概率响应
        instance_prob = self.compute_prob_response(instance_feats, ref_memory)  # shape=[B, C_ref]
        prototype_prob = self.compute_prob_response(prototype_feats, ref_memory)  # shape=[B, C_ref]

        # 2. 计算KL散度损失（公式19）
        # 加eps避免log(0)
        eps = 1e-10
        kl_loss = self.kl_criterion(instance_prob.log() + eps, prototype_prob + eps)

        # 3. 加权损失（论文公式21）
        ipcc_loss = self.weight * kl_loss

        # 4. 计算概率响应相似度（评估一致性）
        prob_sim = F.cosine_similarity(instance_prob, prototype_prob, dim=1).mean().item()

        return ipcc_loss, prob_sim


class CSANetTotalLoss(nn.Module):
    """
    CSANet总损失计算器：整合Step-I/II所有损失（公式21，🔶1-197）
    支持分阶段切换损失组合（Step-I: 仅NCE；Step-II: NCE + CC + IPCC）
    """
    def __init__(self):
        super(CSANetTotalLoss, self).__init__()
        # 初始化各损失函数
        self.nce_loss = ClusterNCELoss()
        self.cc_loss = CrossModalityContrastiveLoss()
        self.ipcc_loss = IPCCConsistencyLoss()
        # 损失权重（论文公式21）
        self.w_nce = LOSS_CONFIG["loss_weights"]["lambda_nce"]
        self.w_cc = LOSS_CONFIG["loss_weights"]["lambda_cc"]

    def forward(
        self,
        stage: str,  # 训练阶段（"step_i"或"step_ii"）
        # Step-I/II共用参数
        vis_feats: torch.Tensor, vis_labels: torch.Tensor, vis_memory: torch.Tensor, vis_pid2idx: dict,
        ir_feats: torch.Tensor, ir_labels: torch.Tensor, ir_memory: torch.Tensor, ir_pid2idx: dict,
        # Step-II专属参数
        cap_vis2ir: Optional[dict] = None, cap_ir2vis: Optional[dict] = None,
        # IPCC专属参数
        ipcc_instance_feats: Optional[torch.Tensor] = None, ipcc_proto_feats: Optional[torch.Tensor] = None,
        ref_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: 总损失
            loss_metrics: 各损失分项与准确率（用于日志记录）
        """
        loss_metrics = {}

        # 1. 计算Step-I单模态对比损失（可见光+红外）
        vis_nce_loss, vis_nce_acc = self.nce_loss(
            instance_feats=vis_feats, instance_labels=vis_labels,
            memory=vis_memory, memory_pid2idx=vis_pid2idx
        )
        ir_nce_loss, ir_nce_acc = self.nce_loss(
            instance_feats=ir_feats, instance_labels=ir_labels,
            memory=ir_memory, memory_pid2idx=ir_pid2idx
        )
        total_nce_loss = self.w_nce * (vis_nce_loss + ir_nce_loss)
        loss_metrics.update({
            "vis_nce_loss": vis_nce_loss.item(), "ir_nce_loss": ir_nce_loss.item(),
            "vis_nce_acc": vis_nce_acc, "ir_nce_acc": ir_nce_acc
        })

        if stage == "step_i":
            # Step-I：仅单模态对比损失
            total_loss = total_nce_loss
            loss_metrics["total_loss"] = total_loss.item()
            return total_loss, loss_metrics
        elif stage == "step_ii":
            # Step-II：整合NCE + CC + IPCC损失
            # 2. 计算跨模态对比损失
            if cap_vis2ir is None or cap_ir2vis is None:
                raise ValueError("Step-II需传入CAP关联字典（cap_vis2ir、cap_ir2vis）")
            cc_loss, vis2ir_acc, ir2vis_acc = self.cc_loss(
                vis_feats=vis_feats, vis_labels=vis_labels,
                ir_feats=ir_feats, ir_labels=ir_labels,
                vis_memory=vis_memory, ir_memory=ir_memory,
                cap_vis2ir=cap_vis2ir, cap_ir2vis=cap_ir2vis
            )
            total_cc_loss = self.w_cc * cc_loss
            loss_metrics.update({
                "cc_loss": cc_loss.item(), "vis2ir_acc": vis2ir_acc, "ir2vis_acc": ir2vis_acc
            })

            # 3. 计算IPCC一致性损失（仅复杂样本）
            if ipcc_instance_feats is not None and ipcc_proto_feats is not None and ref_memory is not None:
                ipcc_loss, prob_sim = self.ipcc_loss(
                    instance_feats=ipcc_instance_feats,
                    prototype_feats=ipcc_proto_feats,
                    ref_memory=ref_memory
                )
                loss_metrics.update({"ipcc_loss": ipcc_loss.item(), "ipcc_prob_sim": prob_sim})
            else:
                ipcc_loss = torch.tensor(0.0, device=vis_feats.device)
                loss_metrics["ipcc_loss"] = 0.0
                loss_metrics["ipcc_prob_sim"] = 0.0

            # 4. 总损失（公式21）
            total_loss = total_nce_loss + total_cc_loss + ipcc_loss
            loss_metrics["total_loss"] = total_loss.item()
            return total_loss, loss_metrics
        else:
            raise ValueError(f"无效训练阶段：{stage}，仅支持'step_i'或'step_ii'")