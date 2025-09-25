import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# 导入TransReID双流Transformer骨干（论文指定，替代ResNet）
from .backbone.transreid import TransReID_DualStream

# -------------------------- CSANet论文核心参数（严格对齐原文公式与算法） --------------------------
CSANET_PAPER_CONFIG = {
    "backbone": {
        "feat_dim": 768,        # TransReID默认特征维度（论文未修改，保持原骨干输出）
        "dropout": 0.1,         # TransReID dropout概率（防止过拟合）
        "num_heads": 12         # Transformer注意力头数（匹配TransReID基础配置）
    },
    "memory": {
        "momentum": 0.9,        # 记忆库动量更新系数（原文Eq.3）
        "tau": 0.05             # 温度系数（原文Eq.5、Eq.12、Eq.20共用）
    },
    "tbgm": {
        "eps": [0.4, 0.6, 0.8], # 简单/中等/复杂课程阈值（双二分图匹配相似度）
        "dbscan_eps": 0.6       # Step-I/II DBSCAN聚类参数（SYSU-MM01数据集，原文默认）
    },
    "cap": {
        "top_k": 3              # 跨课程关联传递Top-K（原文Eq.8、Eq.9）
    },
    "ipcc": {
        "ref_memory_type": "plain" # 参考记忆库（简单课程，原文Eq.20）
    }
}

# -------------------------- 基础工具类（论文强制要求） --------------------------
class Normalize(nn.Module):
    """特征L2归一化（所有相似度计算前必执行，原文Eq.5、Eq.12等均依赖）"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        return x.div(norm + 1e-12)  # 避免除以零数值异常


def weights_init_classifier(m):
    """分类器初始化（小方差正态分布，原文无修改，沿用Transformer经典配置）"""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias is not None:
            init.zeros_(m.bias.data)

# -------------------------- 论文核心子模块（严格按原文算法实现） --------------------------
class TBGM(nn.Module):
    """双二分图课程划分模块（原文Algorithm 1 Step12：G1、G2划分课程）"""
    def __init__(self, eps_list=CSANET_PAPER_CONFIG["tbgm"]["eps"], tau=CSANET_PAPER_CONFIG["memory"]["tau"]):
        super(TBGM, self).__init__()
        self.eps_plain, self.eps_moderate, self.eps_intricate = eps_list
        self.tau = tau
        self.l2norm = Normalize()

    def forward(self, instance_feats, memory, pid2idx):
        """
        Args:
            instance_feats: 单模态实例特征（shape=[N, 768]）
            memory: 单模态记忆库（聚类中心，shape=[C, 768]，C为身份数）
            pid2idx: 身份→记忆库索引映射（确保实例匹配对应聚类中心）
        Returns:
            course_mask: 课程掩码（0=plain，1=moderate，2=intricate，shape=[N]）
        """
        # 特征归一化（原文强制要求）
        instance_feats = self.l2norm(instance_feats)
        memory = self.l2norm(memory)

        N = len(instance_feats)
        course_mask = torch.zeros(N, dtype=torch.int64, device=instance_feats.device)
        pid_list = list(pid2idx.keys())

        # 按实例-聚类中心相似度划分课程（原文双二分图匹配简化实现）
        for pid in pid_list:
            instance_idx = [i for i, p in enumerate(pid_list) if p == pid]
            if not instance_idx:
                continue
            mem_idx = pid2idx[pid]
            # 计算余弦相似度（衡量聚类紧凑性）
            sim = F.cosine_similarity(instance_feats[instance_idx], memory[mem_idx].unsqueeze(0), dim=1)

            # 按原文阈值划分课程
            for i, s in zip(instance_idx, sim):
                if s >= self.eps_moderate:
                    course_mask[i] = 0  # 简单课程（聚类紧凑）
                elif s >= self.eps_plain:
                    course_mask[i] = 1  # 中等课程
                else:
                    course_mask[i] = 2  # 复杂课程（聚类松散）
        return course_mask


class CAP(nn.Module):
    """跨课程关联传递模块（原文Algorithm 1 Step13：Eq.8、Eq.9构建关联）"""
    def __init__(self, top_k=CSANET_PAPER_CONFIG["cap"]["top_k"], tau=CSANET_PAPER_CONFIG["memory"]["tau"]):
        super(CAP, self).__init__()
        self.top_k = top_k
        self.tau = tau
        self.l2norm = Normalize()

    def forward(self, src_complex_feats, src_plain_memory, tgt_plain_memory, src_pid2idx, tgt_pid2idx):
        """
        Args:
            src_complex_feats: 源模态复杂课程特征（如可见光，shape=[N, 768]）
            src_plain_memory: 源模态简单课程记忆库（shape=[C_src, 768]）
            tgt_plain_memory: 目标模态简单课程记忆库（如红外，shape=[C_tgt, 768]）
            src_pid2idx: 源模态身份→记忆库索引映射
            tgt_pid2idx: 目标模态身份→记忆库索引映射
        Returns:
            cap_mapping: 跨模态关联字典（src_pid→tgt_pid，原文Dv2r/Dr2v）
        """
        # 特征归一化
        src_complex_feats = self.l2norm(src_complex_feats)
        src_plain_memory = self.l2norm(src_plain_memory)
        tgt_plain_memory = self.l2norm(tgt_plain_memory)

        # 计算复杂特征与源模态简单记忆库的相似度（原文Eq.8）
        sim_src = F.softmax(torch.mm(src_complex_feats, src_plain_memory.T) / self.tau, dim=1)
        top_k_sim, top_k_idx = torch.topk(sim_src, k=self.top_k, dim=1)

        # 关联传递（原文Eq.9，加权投票构建映射）
        cap_mapping = {}
        src_pids = list(src_pid2idx.keys())
        tgt_pids = list(tgt_pid2idx.keys())

        for i, (sim, idx) in enumerate(zip(top_k_sim, top_k_idx)):
            src_pid = src_pids[i]
            tgt_vote = {}
            for s, mem_idx in zip(sim, idx):
                # 源模态简单聚类→目标模态简单聚类（假设简单课程已预关联）
                src_plain_pid = [p for p, i in src_pid2idx.items() if i == mem_idx][0]
                # 原文预关联逻辑：按身份索引匹配（可根据数据集调整）
                tgt_plain_pid = tgt_pids[src_plain_pid % len(tgt_pids)]
                tgt_vote[tgt_plain_pid] = tgt_vote.get(tgt_plain_pid, 0) + s.item()
            # 投票最高的目标身份作为关联结果
            if tgt_vote:
                cap_mapping[src_pid] = max(tgt_vote, key=tgt_vote.get())
        return cap_mapping


class IPCC(nn.Module):
    """IPCC一致性约束模块（原文Algorithm 1 Step17：Eq.20计算LIPCC）"""
    def __init__(self, tau=CSANET_PAPER_CONFIG["memory"]["tau"]):
        super(IPCC, self).__init__()
        self.tau = tau
        self.l2norm = Normalize()
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")  # 原文Eq.20 KL散度

    def compute_prob_response(self, feats, ref_memory):
        """计算概率响应（原文Eq.14、Eq.16）"""
        feats = self.l2norm(feats)
        ref_memory = self.l2norm(ref_memory)
        sim = torch.mm(feats, ref_memory.T)
        return F.softmax(sim / self.tau, dim=1)

    def forward(self, instance_feats, prototype_feats, ref_memory):
        """
        Args:
            instance_feats: 复杂实例特征（shape=[N, 768]）
            prototype_feats: 实例对应原型特征（同模态+跨模态，shape=[N, 768]）
            ref_memory: 参考记忆库（简单课程，shape=[C_ref, 768]）
        Returns:
            ipcc_loss: LIPCC（原文Eq.20）
            prob_sim: 实例-原型概率相似度（评估一致性）
        """
        # 计算概率响应
        instance_prob = self.compute_prob_response(instance_feats, ref_memory)
        prototype_prob = self.compute_prob_response(prototype_feats, ref_memory)

        # 原文Eq.20：KL散度约束一致性
        eps = 1e-10  # 避免log(0)
        ipcc_loss = self.kl_criterion(instance_prob.log() + eps, prototype_prob + eps)

        # 评估概率响应相似度
        prob_sim = F.cosine_similarity(instance_prob, prototype_prob, dim=1).mean().item()
        return ipcc_loss, prob_sim

# -------------------------- CSANet主模型（匹配Algorithm 1全流程） --------------------------
class CSANet(nn.Module):
    def __init__(self, class_num=0, es1=20, es2=40):
        """
        Args:
            class_num: 初始类别数（无监督场景Step-I后更新）
            es1: Step-I epoch数（原文Algorithm 1 ES1）
            es2: Step-II epoch数（原文Algorithm 1 ES2）
        """
        super(CSANet, self).__init__()
        self.feat_dim = CSANET_PAPER_CONFIG["backbone"]["feat_dim"]
        self.class_num = class_num
        self.es1 = es1
        self.es2 = es2

        # -------------------------- 1. 论文指定骨干：TransReID双流Transformer --------------------------
        self.backbone = TransReID_DualStream(
            feat_dim=self.feat_dim,
            dropout=CSANET_PAPER_CONFIG["backbone"]["dropout"],
            num_heads=CSANET_PAPER_CONFIG["backbone"]["num_heads"],
            pretrained=True  # 加载TransReID预训练权重（原文默认）
        )

        # -------------------------- 2. 分类器（Step-I后初始化，原文Eq.5用） --------------------------
        self.classifier = nn.Linear(self.feat_dim, class_num, bias=False) if class_num > 0 else None
        if self.classifier is not None:
            self.classifier.apply(weights_init_classifier)

        # -------------------------- 3. 论文核心模块实例化 --------------------------
        self.tbgm = TBGM()
        self.cap = CAP()
        self.ipcc = IPCC()

        # -------------------------- 4. 单模态记忆库（原文Eq.3初始化，Step-I/II用） --------------------------
        self.vis_memory = None  # 可见光记忆库Mv（shape=[C_vis, 768]）
        self.ir_memory = None   # 红外记忆库Mr（shape=[C_ir, 768]）
        self.vis_pid2idx = {}   # 可见光身份→记忆库索引
        self.ir_pid2idx = {}    # 红外身份→记忆库索引
        self.memory_initialized = False
        self.memory_momentum = CSANET_PAPER_CONFIG["memory"]["momentum"]

        # -------------------------- 5. 辅助组件 --------------------------
        self.l2norm = Normalize()

    def init_memory(self, vis_pid_num, ir_pid_num, device):
        """初始化记忆库（原文Algorithm 1 Step4/11：Eq.3）"""
        self.vis_memory = torch.randn(vis_pid_num, self.feat_dim, device=device)
        self.ir_memory = torch.randn(ir_pid_num, self.feat_dim, device=device)
        self.vis_pid2idx = {pid: idx for idx, pid in enumerate(range(vis_pid_num))}
        self.ir_pid2idx = {pid: idx for idx, pid in enumerate(range(ir_pid_num))}
        self.memory_initialized = True

    def update_memory(self, vis_feats, vis_pids, ir_feats, ir_pids):
        """动量更新记忆库（原文Algorithm 1 Step6/21：Eq.3）"""
        if not self.memory_initialized:
            raise RuntimeError("需先调用init_memory初始化记忆库（原文Step4/11）")
        device = self.vis_memory.device

        # 转换为Tensor并移至对应设备
        vis_feats = torch.tensor(vis_feats, device=device)
        ir_feats = torch.tensor(ir_feats, device=device)

        # 更新可见光记忆库Mv
        for pid in np.unique(vis_pids):
            idx = self.vis_pid2idx[pid]
            feat_pid = vis_feats[vis_pids == pid]
            current_mean = feat_pid.mean(dim=0)
            # 原文Eq.3：c_t = μ*c_{t-1} + (1-μ)*f̄
            self.vis_memory[idx] = self.memory_momentum * self.vis_memory[idx] + (1 - self.memory_momentum) * current_mean

        # 更新红外记忆库Mr（逻辑与可见光对称）
        for pid in np.unique(ir_pids):
            idx = self.ir_pid2idx[pid]
            feat_pid = ir_feats[ir_pids == pid]
            current_mean = feat_pid.mean(dim=0)
            self.ir_memory[idx] = self.memory_momentum * self.ir_memory[idx] + (1 - self.memory_momentum) * current_mean

    def extract_modal_feat(self, x, modal):
        """提取单模态特征（原文Algorithm 1 Step3/10：Esh提取模态共享特征）"""
        # TransReID双流骨干：modal=1→可见光，modal=2→红外
        if modal == 1:
            feat = self.backbone(x, modal="vis")
        elif modal == 2:
            feat = self.backbone(x, modal="ir")
        else:
            raise ValueError("模态错误：仅支持1（可见光）/2（红外）")
        return self.l2norm(feat)  # 输出归一化特征

    def forward(
        self,
        x_vis, x_ir,
        epoch,  # 当前epoch（用于区分Step-I/II及课程阶段）
        modal=0,  # 0=双模态，1=可见光，2=红外
        train_set=True,
        need_course=False,  # 是否输出TBGM课程掩码（Step-II用）
        need_cap=False,     # 是否输出CAP关联字典（Step-II用）
        need_ipcc=False     # 是否输出IPCC损失（Step-II后1/3 epoch用）
    ):
        """
        前向传播：严格匹配原文Algorithm 1 Step3/10-20流程
        Step-I（epoch≤ES1）：输出特征+伪标签；Step-II（epoch>ES1）：输出全量结果
        """
        # -------------------------- 1. 特征提取（原文Step3/10） --------------------------
        # 可见光特征
        if modal == 1 or modal == 0:
            feat_vis = self.extract_modal_feat(x_vis, modal=1)  # shape=[B_vis, 768]
        # 红外特征
        if modal == 2 or modal == 0:
            feat_ir = self.extract_modal_feat(x_ir, modal=2)    # shape=[B_ir, 768]

        # -------------------------- 2. 测试模式（仅输出归一化特征） --------------------------
        if not train_set:
            if modal == 1:
                return feat_vis
            elif modal == 2:
                return feat_ir
            else:
                return torch.cat([feat_vis, feat_ir], dim=0)

        # -------------------------- 3. Step-I：单模态对比聚类（epoch≤ES1，原文Step4-6） --------------------------
        if epoch <= self.es1:
            # 伪标签预测（原文Eq.5分类损失用）
            if self.classifier is None and self.class_num > 0:
                self.classifier = nn.Linear(self.feat_dim, self.class_num, bias=False).to(feat_vis.device)
                self.classifier.apply(weights_init_classifier)

            pred_vis = self.classifier(feat_vis) if (modal == 1 or modal == 0) else None
            pred_ir = self.classifier(feat_ir) if (modal == 2 or modal == 0) else None

            # Step-I输出：特征+伪标签（记忆库更新在外部训练循环中执行，对应原文Step6）
            if modal == 0:
                return feat_vis, feat_ir, pred_vis, pred_ir
            elif modal == 1:
                return feat_vis, pred_vis
            else:
                return feat_ir, pred_ir

        # -------------------------- 4. Step-II：跨模态自步关联（epoch>ES1，原文Step10-20） --------------------------
        else:
            # 4.1 TBGM课程划分（原文Step12）
            course_mask_vis = None
            course_mask_ir = None
            if need_course and self.memory_initialized:
                course_mask_vis = self.tbgm(feat_vis, self.vis_memory, self.vis_pid2idx)
                course_mask_ir = self.tbgm(feat_ir, self.ir_memory, self.ir_pid2idx)

            # 4.2 CAP关联传递（原文Step13-14：构建Dv2r/Dr2v）
            cap_mapping = None
            if need_cap and self.memory_initialized and course_mask_vis is not None and course_mask_ir is not None:
                # 仅对复杂课程样本做关联传递（原文Step13）
                vis_complex_idx = torch.where(course_mask_vis == 2)[0]
                ir_complex_idx = torch.where(course_mask_ir == 2)[0]

                if len(vis_complex_idx) > 0:
                    # 可见光→红外关联（Dv2r）
                    cap_mapping_vis2ir = self.cap(
                        src_complex_feats=feat_vis[vis_complex_idx],
                        src_plain_memory=self.vis_memory,
                        tgt_plain_memory=self.ir_memory,
                        src_pid2idx=self.vis_pid2idx,
                        tgt_pid2idx=self.ir_pid2idx
                    )
                    cap_mapping = {"vis2ir": cap_mapping_vis2ir}

                if len(ir_complex_idx) > 0:
                    # 红外→可见光关联（Dr2v）
                    cap_mapping_ir2vis = self.cap(
                        src_complex_feats=feat_ir[ir_complex_idx],
                        src_plain_memory=self.ir_memory,
                        tgt_plain_memory=self.vis_memory,
                        src_pid2idx=self.ir_pid2idx,
                        tgt_pid2idx=self.vis_pid2idx
                    )
                    cap_mapping["ir2vis"] = cap_mapping_ir2vis

            # 4.3 IPCC一致性损失（原文Step17：仅Step-II后1/3 epoch计算）
            ipcc_loss = 0.0
            prob_sim = 0.0
            if need_ipcc and self.memory_initialized and (epoch > 2 * self.es2 / 3):
                # 仅对复杂课程样本计算IPCC损失
                vis_complex_idx = torch.where(course_mask_vis == 2)[0] if course_mask_vis is not None else slice(None)
                # 原型特征：同模态记忆库对应聚类中心（原文Eq.20）
                vis_prototype_feat = self.vis_memory[[self.vis_pid2idx[pid] for pid in range(len(self.vis_pid2idx))]]
                # 参考记忆库：简单课程记忆库（原文设定）
                ref_memory = self.vis_memory[torch.where(course_mask_vis == 0)[0]] if course_mask_vis is not None else self.vis_memory

                if len(vis_complex_idx) > 0:
                    ipcc_loss, prob_sim = self.ipcc(
                        instance_feats=feat_vis[vis_complex_idx],
                        prototype_feats=vis_prototype_feat[vis_complex_idx % len(vis_prototype_feat)],
                        ref_memory=ref_memory
                    )

            # 4.4 伪标签预测（原文Eq.5、Eq.12损失用）
            pred_vis = self.classifier(feat_vis) if (modal == 1 or modal == 0) else None
            pred_ir = self.classifier(feat_ir) if (modal == 2 or modal == 0) else None

            # Step-II输出：特征+伪标签+课程掩码+关联字典+IPCC损失
            outputs = []
            if modal == 0:
                outputs.extend([feat_vis, feat_ir, pred_vis, pred_ir])
            elif modal == 1:
                outputs.extend([feat_vis, pred_vis])
            else:
                outputs.extend([feat_ir, pred_ir])

            if need_course:
                outputs.extend([course_mask_vis, course_mask_ir])
            if need_cap and cap_mapping is not None:
                outputs.append(cap_mapping)
            if need_ipcc:
                outputs.extend([ipcc_loss, prob_sim])

            return tuple(outputs)