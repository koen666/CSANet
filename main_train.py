import argparse
import easydict
import sys
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
# 保留必要工具，新增CSANet专属工具
from utils import (
    CSANetLogger, AverageMeter, set_seed, GenIdx, CSANetSingleModalitySampler, CSANetCurriculumSampler,
    build_modal_memory, generate_curriculum_mask, get_current_curriculum
)
from data_loader import SYSUData, RegDBData, TestData, get_adca_transform  # 导入论文ADCA数据增强
from data_manager import (
    process_query_sysu, process_gallery_sysu, process_test_regdb
)
# 替换模型：删除BaseResNet，导入CSANet（基于TransReID）
from model.network import CSANet
# 替换损失：删除冗余损失，导入CSANet总损失计算器
from loss import CSANetTotalLoss
# 新增模块冻结功能，适配分阶段训练
from optimizer import select_optimizer, adjust_learning_rate, freeze_modules
from engine import trainer, tester  # 复用适配后的训练/测试器
from otla_sk import cpu_sk_ir_trainloader, evaluate_pseudo_label  # 保留OTLA-SK伪标签优化

# -------------------------- CSANet全局配置（严格对齐论文） --------------------------
CSANET_CONFIG = {
    "train_stage": {
        "step_i_epoch": 20,    # Step-I单模态聚类epoch数（论文Algorithm 1 ES1）
        "step_ii_epoch": 40,   # Step-II跨模态关联epoch数（ES2）
        "total_epoch": 60      # 总epoch（20+40）
    },
    "memory": {
        "momentum": 0.9,       # 记忆库动量更新系数（论文Eq.3）
        "update_freq": 5       # 记忆库更新频率（每5 epoch）
    },
    "curriculum": {
        "optimize_courses": ["moderate", "intricate"]  # OTLA-SK仅优化中等/复杂课程
    }
}

def main_worker(args, args_main):
    ## 1. 基础配置初始化（设备、种子、路径）
    # GPU与种子配置（确保实验可复现，论文要求）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = args.cudnn_benchmark if hasattr(args, "cudnn_benchmark") else True
    set_seed(args.seed, cuda=torch.cuda.is_available())

    # 路径配置（兼容原有逻辑，补充记忆库保存路径）
    exp_dir = f"{args.dataset}_{args.setting}_{args.file_name}"
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    log_path = os.path.join(exp_dir, f"{args.dataset}_{args.log_path}")
    vis_log_path = os.path.join(exp_dir, f"{args.dataset}_{args.vis_log_path}")
    model_path = os.path.join(exp_dir, f"{args.dataset}_{args.model_path}")
    memory_path = os.path.join(exp_dir, "memory")  # 新增记忆库路径
    for path in [log_path, vis_log_path, model_path, memory_path]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # 日志初始化
    sys.stdout = CSANetLogger(os.path.join(log_path, "train.log"))
    test_log = open(os.path.join(log_path, "test.log"), "w")
    writer = SummaryWriter(vis_log_path)
    print(f"实验配置：\nargs_main: {args_main}\nargs: {args}\n")

    ## 2. 数据加载（适配论文ADCA增强与分阶段需求）
    print("==> 加载数据集...")
    t_start = time.time()

    # 数据增强：替换为论文ADCA策略（避免过拟合，提升泛化性）
    transform_train = get_adca_transform(args.img_w, args.img_h, is_train=True)
    transform_test = get_adca_transform(args.img_w, args.img_h, is_train=False)

    # 加载训练集（分Step-I/II，Step-I单模态，Step-II双模态）
    train_loaders = {}
    if args.dataset == "sysu":
        data_path = os.path.join(args.dataset_path, "SYSU-MM01/")
        # Step-I单模态训练集（用于聚类与记忆库初始化）
    # Step-I：复用原版双模态加载逻辑，通过采样器筛选单模态样本
    # 1. 加载双模态训练集（原版 SYSUData 无 is_train 参数，自动通过 pre_process_sysu 加载训练数据）
        trainset_step1 = SYSUData(
            args, data_path,
            transform_train_rgb=transform_train,  # 可见光增强
            transform_train_ir=transform_train   # 红外增强（Step-I暂用同一增强）
        )

        # 2. Step-I单模态采样器（仅采样可见光样本，无需新增 current_modal）
        vis_sampler = CSANetSingleModalitySampler(
            modal_pseudo_label=trainset_step1.train_color_label,  # 可见光标签
            num_pos=args.num_pos,
            batch_size=args.train_batch_size
        )
        # 3. 生成单模态训练加载器（仅加载可见光样本）
        trainset_step1.cIndex = vis_sampler.index1  # 可见光样本索引
        trainset_step1.tIndex = vis_sampler.index1  # 红外索引暂用可见光（Step-I仅用可见光）
        trainloader_vis = data.DataLoader(
            trainset_step1, batch_size=args.train_batch_size * args.num_pos,
            sampler=vis_sampler, num_workers=args.workers, drop_last=True
        )
        trainset_step1_ir = SYSUData(
            args, data_path, transform_train=transform_train,
            is_train=True, for_memory=True, current_modal="ir"
        )
        train_loaders["step_i_vis"] = data.DataLoader(
            trainset_step1_vis, batch_size=args.test_batch_size,
            shuffle=False, num_workers=args.workers
        )
        train_loaders["step_i_ir"] = data.DataLoader(
            trainset_step1_ir, batch_size=args.test_batch_size,
            shuffle=False, num_workers=args.workers
        )

        # Step-II双模态训练集（用于跨模态关联学习）
        trainset_step2 = SYSUData(
            args, data_path, transform_train_rgb=transform_train,
            transform_train_ir=transform_train, is_train=True
        )

        # 测试集（按论文协议处理，保留相机ID用于评估）
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode)

    elif args.dataset == "regdb":
        data_path = os.path.join(args.dataset_path, "RegDB/")
        # Step-I单模态训练集
        trainset_step1_vis = RegDBData(
            args, data_path, transform_train=transform_train,
            is_train=True, for_memory=True, current_modal="vis"
        )
        trainset_step1_ir = RegDBData(
            args, data_path, transform_train=transform_train,
            is_train=True, for_memory=True, current_modal="ir"
        )
        train_loaders["step_i_vis"] = data.DataLoader(
            trainset_step1_vis, batch_size=args.test_batch_size,
            shuffle=False, num_workers=args.workers
        )
        train_loaders["step_i_ir"] = data.DataLoader(
            trainset_step1_ir, batch_size=args.test_batch_size,
            shuffle=False, num_workers=args.workers
        )

        # Step-II双模态训练集
        trainset_step2 = RegDBData(
            args, data_path, transform_train_rgb=transform_train,
            transform_train_ir=transform_train, is_train=True
        )

        # 测试集（双向模式，补充相机ID）
        query_modal = args.mode.split("to")[0]
        gall_modal = args.mode.split("to")[1]
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modality=query_modal)
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modality=gall_modal)
        query_cam = np.ones_like(query_label) if query_modal == "visible" else np.ones_like(query_label)*2
        gall_cam = np.ones_like(gall_label)*2 if gall_modal == "thermal" else np.ones_like(gall_label)

    # 测试集加载（统一格式，适配论文评估）
    gallset = TestData(gall_img, gall_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform_test=transform_test, img_size=(args.img_w, args.img_h))
    test_loader = {
        "query_loader": data.DataLoader(
            queryset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers
        ),
        "gall_loader": data.DataLoader(
            gallset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers
        )
    }
    test_info = {
        "query_pids": query_label, "gall_pids": gall_label,
        "query_cams": query_cam, "gall_cams": gall_cam
    }

    # 数据集统计（论文实验记录要求）
    n_class_vis = len(np.unique(trainset_step2.train_color_label))
    n_class_ir = len(np.unique(trainset_step2.train_thermal_label))
    print(f"数据集统计（{args.dataset}）：")
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print(f"  visible  | {n_class_vis:5d} | {len(trainset_step2.train_color_label):8d}")
    print(f"  thermal  | {n_class_ir:5d} | {len(trainset_step2.train_thermal_label):8d}")
    print("  ----------------------------")
    print(f"  query    | {len(np.unique(query_label)):5d} | {len(query_label):8d}")
    print(f"  gallery  | {len(np.unique(gall_label)):5d} | {len(gall_label):8d}")
    print("  ----------------------------")
    print(f"数据加载耗时：{time.time() - t_start:.3f}s\n")

    ## 3. 模型初始化（替换为CSANet，基于TransReID骨干）
    print("==> 初始化CSANet模型...")
    main_net = CSANet(
        class_num=n_class_vis,
        es1=CSANET_CONFIG["train_stage"]["step_i_epoch"],
        es2=CSANET_CONFIG["train_stage"]["step_ii_epoch"]
    ).to(device)

    # 加载预训练/Resume（兼容原有逻辑）
    start_epoch = args.start_epoch
    best_rank1 = 0.0
    best_mAP = 0.0
    best_mINP = 0.0
    if args_main.resume and os.path.exists(args_main.resume_path):
        checkpoint = torch.load(args_main.resume_path, map_location=device)
        main_net.load_state_dict(checkpoint["main_net"])
        start_epoch = checkpoint.get("epoch", args.start_epoch)
        best_rank1 = checkpoint.get("best_rank1", 0.0)
        print(f"加载Resume模型：{args_main.resume_path}，起始epoch：{start_epoch}\n")
    else:
        print("未加载Resume模型，从头开始训练\n")

    ## 4. Step-I：单模态对比聚类（论文Algorithm 1 Step1-7）
    print("="*60)
    print(f"Step-I：单模态对比聚类（Epoch {start_epoch} ~ {CSANET_CONFIG['train_stage']['step_i_epoch']}）")
    print("="*60)

    # 冻结Step-I不训练的模块（TBGM/CAP/IPCC）
    freeze_modules(
        main_net,
        freeze_backbone=False,  # Step-I训练骨干网络
        freeze_tbgm_cap_ipcc=True
    )

    # 初始化Step-I损失（仅ClusterNCE，论文Eq.5）
    step1_loss_fn = CSANetTotalLoss(
        lambda_nce=1.0, lambda_cc=0.0, lambda_ipcc=0.0  # Step-I无跨模态损失
    )

    # 初始化Step-I优化器（仅优化骨干，学习率按论文设定）
    optimizer_step1 = select_optimizer(
        args, main_net,
        base_lr=args.base_lr if hasattr(args, "base_lr") else 3e-4,
        module_lr_ratio=1.0  # Step-I无模块，比例为1.0
    )

    # Step-I核心变量
    vis_memory = None  # 可见光记忆库（Mv，论文Eq.3）
    ir_memory = None   # 红外记忆库（Mr）
    train_thermal_pseudo_label = np.random.randint(0, n_class_vis, len(trainset_step2.train_thermal_label))

    for epoch in range(start_epoch, CSANET_CONFIG["train_stage"]["step_i_epoch"] + 1):
        # 调整学习率（Step-I升温策略，避免初始梯度爆炸）
        adjust_learning_rate(optimizer_step1, current_epoch=epoch, stage="step_i")

        # 1. 单模态采样器（仅采样可见光，红外依赖伪标签）
        vis_sampler = CSANetSingleModalitySampler(
            modal_pseudo_label=trainset_step2.train_color_label,
            num_pos=args.num_pos,
            batch_size=args.train_batch_size
        )
        trainset_step2.cIndex = vis_sampler.index1
        trainloader_vis = data.DataLoader(
            trainset_step2, batch_size=args.train_batch_size * args.num_pos,
            sampler=vis_sampler, num_workers=args.workers, drop_last=True
        )

        # 2. 可见光单模态训练（论文Eq.5优化）
        train_stats = trainer(
            args=args,
            epoch=epoch,
            main_net=main_net,
            optimizer=optimizer_step1,
            trainloader=trainloader_vis,
            total_loss_fn=step1_loss_fn,
            logger=sys.stdout,
            writer=writer,
            stage="step_i",
            modal="vis"
        )

        # 3. OTLA-SK优化红外伪标签（每5 epoch一次，提升伪标签可靠性）
        if epoch % 5 == 0:
            ir_sampler = CSANetSingleModalitySampler(
                modal_pseudo_label=train_thermal_pseudo_label,
                num_pos=args.num_pos,
                batch_size=args.train_batch_size
            )
            trainset_step2.tIndex = ir_sampler.index2
            trainloader_ir = data.DataLoader(
                trainset_step2, batch_size=args.train_batch_size * args.num_pos,
                sampler=ir_sampler, num_workers=args.workers, drop_last=True
            )

            # 调用OTLA-SK优化伪标签
            ir_pseudo_op, ir_pseudo_mp, ir_real_label, unique_tIdx, conf, sk_stats = cpu_sk_ir_trainloader(
                args=args,
                main_net=main_net,
                trainloader=trainloader_ir,
                tIndex=ir_sampler.index2,
                n_class=n_class_ir,
                curriculum_mask=None  # Step-I无课程划分
            )
            # 更新红外伪标签
            train_thermal_pseudo_label[unique_tIdx] = ir_pseudo_op.numpy()
            # 评估伪标签质量（ARI，论文表VIII聚类质量指标）
            ari_stats = evaluate_pseudo_label(
                pseudo_label=train_thermal_pseudo_label[unique_tIdx],
                true_label=ir_real_label.numpy(),
                confidence=conf,
                dataset=args.dataset,
                course="step_i_ir"
            )
            print(f"Step-I Epoch {epoch}：红外伪标签ARI={ari_stats['ARI']:.4f}，高置信度占比={sk_stats['high_confidence_ratio']:.2%}\n")

        # 4. 定期更新记忆库（论文Eq.3，每5 epoch）
        if epoch % CSANET_CONFIG["memory"]["update_freq"] == 0:
            print(f"Step-I Epoch {epoch}：更新单模态记忆库...")
            # 提取单模态特征
            vis_feats = main_net.extract_modal_feats(train_loaders["step_i_vis"], device=device, modal="vis")
            ir_feats = main_net.extract_modal_feats(train_loaders["step_i_ir"], device=device, modal="ir")
            # 构建/更新记忆库（动量更新）
            vis_memory, vis_pid2idx = build_modal_memory(
                modal_feats=vis_feats,
                modal_pseudo_label=trainset_step1_vis.train_color_label,
                old_memory=vis_memory,
                momentum=CSANET_CONFIG["memory"]["momentum"]
            )
            ir_memory, ir_pid2idx = build_modal_memory(
                modal_feats=ir_feats,
                modal_pseudo_label=trainset_step1_ir.train_thermal_label,
                old_memory=ir_memory,
                momentum=CSANET_CONFIG["memory"]["momentum"]
            )
            # 记忆库传入模型
            main_net.init_memory(len(vis_pid2idx), len(ir_pid2idx), device=device)
            main_net.update_memory(vis_feats, trainset_step1_vis.train_color_label, ir_feats, trainset_step1_ir.train_thermal_label)
            # 保存记忆库
            torch.save({"vis_memory": vis_memory, "ir_memory": ir_memory}, os.path.join(memory_path, f"memory_epoch{epoch}.pth"))

        # 5. Step-I测试（每10 epoch，验证聚类效果）
        if epoch % 10 == 0:
            print(f"Step-I Epoch {epoch}：测试单模态聚类效果...")
            cmc, mAP, mINP, test_stats = tester(
                args=args,
                epoch=epoch,
                main_net=main_net,
                test_loader=test_loader,
                test_info=test_info,
                dataset=args.dataset,
                test_mode=args.mode,
                logger=test_log,
                writer=writer
            )
            print(f"Step-I Epoch {epoch} 测试结果：Rank-1={test_stats['Rank-1']:.2f}%，mAP={test_stats['mAP']:.2f}%\n")

    # Step-I结束：保存中间模型
    step1_ckpt = {
        "main_net": main_net.state_dict(),
        "vis_memory": vis_memory,
        "ir_memory": ir_memory,
        "epoch": CSANET_CONFIG["train_stage"]["step_i_epoch"],
        "best_rank1": best_rank1
    }
    torch.save(step1_ckpt, os.path.join(model_path, "step1_final.pth"))
    print(f"Step-I训练结束，模型保存至：{os.path.join(model_path, 'step1_final.pth')}\n")

    ## 5. Step-II：跨模态自步关联（论文Algorithm 1 Step8-22）
    print("="*60)
    print(f"Step-II：跨模态自步关联（Epoch {CSANET_CONFIG['train_stage']['step_i_epoch']+1} ~ {CSANET_CONFIG['train_stage']['total_epoch']}）")
    print("="*60)

    # 解冻TBGM/CAP/IPCC模块（Step-II训练全模块）
    freeze_modules(
        main_net,
        freeze_backbone=True,  # 冻结骨干，仅微调模块（论文推荐）
        freeze_tbgm_cap_ipcc=False
    )

    # 初始化Step-II损失（NCE + CC + IPCC，论文Eq.21）
    step2_loss_fn = CSANetTotalLoss(
        lambda_nce=1.0,
        lambda_cc=1.0,
        lambda_ipcc=0.5  # 论文Eq.21权重设定
    )

    # 初始化Step-II优化器（模块学习率为骨干10倍，论文🔶1-209）
    optimizer_step2 = select_optimizer(
        args, main_net,
        base_lr=(args.base_lr if hasattr(args, "base_lr") else 3e-4) * 0.1,  # 骨干lr降为Step-I的1/10
        module_lr_ratio=10
    )

    # Step-II核心变量
    start_epoch_step2 = CSANET_CONFIG["train_stage"]["step_i_epoch"] + 1
    # TBGM课程划分（论文Step12，生成样本级课程掩码）
    print("Step-II：TBGM模块划分简单/中等/复杂课程...")
    vis_feats_step2 = main_net.extract_modal_feats(train_loaders["step_i_vis"], device=device, modal="vis")
    ir_feats_step2 = main_net.extract_modal_feats(train_loaders["step_i_ir"], device=device, modal="ir")
    # TBGM输出PID→课程等级映射（0=plain，1=moderate，2=intricate）
    vis_tbgm_course = main_net.tbgm(vis_feats_step2, vis_memory, vis_pid2idx)
    ir_tbgm_course = main_net.tbgm(ir_feats_step2, ir_memory, ir_pid2idx)
    # 生成样本级课程掩码
    vis_curriculum_mask = generate_curriculum_mask(
        modal_pseudo_label=trainset_step2.train_color_label,
        tbgm_curriculum=vis_tbgm_course
    )
    ir_curriculum_mask = generate_curriculum_mask(
        modal_pseudo_label=trainset_step2.train_thermal_label,
        tbgm_curriculum=ir_tbgm_course
    )

    for epoch in range(start_epoch_step2, CSANET_CONFIG["train_stage"]["total_epoch"] + 1):
        # 确定当前课程阶段（论文Step15-17）
        current_course = get_current_curriculum(
            epoch=epoch - start_epoch_step2 + 1,
            step_ii_total=CSANET_CONFIG["train_stage"]["step_ii_epoch"]
        )
        print(f"\nStep-II Epoch {epoch}：当前课程阶段={current_course}")

        # 1. CAP模块生成跨模态关联字典（论文Step13-14，Dv2r/Dr2v）
        cap_mapping = main_net.cap(
            src_complex_feats=vis_feats_step2,
            src_plain_memory=vis_memory,
            tgt_plain_memory=ir_memory,
            src_pid2idx=vis_pid2idx,
            tgt_pid2idx=ir_pid2idx
        )
        print(f"CAP模块生成关联对数：vis2ir={len(cap_mapping.get('vis2ir', {}))}，ir2vis={len(cap_mapping.get('ir2vis', {}))}")

        # 2. 课程采样器（仅采样当前课程样本，论文Step15-17）
        sampler_step2 = CSANetCurriculumSampler(
            train_color_pseudo_label=trainset_step2.train_color_label,
            train_thermal_pseudo_label=train_thermal_pseudo_label,
            color_curriculum_mask=vis_curriculum_mask,
            thermal_curriculum_mask=ir_curriculum_mask,
            cap_mapping=cap_mapping.get("vis2ir", {}),
            num_pos=args.num_pos,
            batch_size=args.train_batch_size,
            current_stage=current_course
        )
        trainset_step2.cIndex = sampler_step2.index1
        trainset_step2.tIndex = sampler_step2.index2
        trainloader_step2 = data.DataLoader(
            trainset_step2, batch_size=args.train_batch_size * args.num_pos,
            sampler=sampler_step2, num_workers=args.workers, drop_last=True
        )

        # 3. OTLA-SK优化红外伪标签（仅中等/复杂课程，论文Step14后优化）
        if current_course in CSANET_CONFIG["curriculum"]["optimize_courses"]:
            print(f"Step-II Epoch {epoch}：OTLA-SK优化红外伪标签...")
            ir_pseudo_op, ir_pseudo_mp, ir_real_label, unique_tIdx, conf, sk_stats = cpu_sk_ir_trainloader(
                args=args,
                main_net=main_net,
                trainloader=trainloader_step2,
                tIndex=sampler_step2.index2,
                n_class=n_class_ir,
                curriculum_mask=ir_curriculum_mask
            )
            train_thermal_pseudo_label[unique_tIdx] = ir_pseudo_op.numpy()
            print(f"OTLA-SK优化结果：高置信度样本占比={sk_stats['high_confidence_ratio']:.2%}")

        # 4. Step-II训练（含NCE+CC+IPCC损失，论文Eq.21）
        train_stats = trainer(
            args=args,
            epoch=epoch,
            main_net=main_net,
            optimizer=optimizer_step2,
            trainloader=trainloader_step2,
            total_loss_fn=step2_loss_fn,
            logger=sys.stdout,
            writer=writer,
            curriculum_mask_vis=vis_curriculum_mask,
            curriculum_mask_ir=ir_curriculum_mask,
            cap_mapping=cap_mapping,
            ref_memory=vis_memory,  # IPCC参考记忆库（简单课程，论文Step17）
            stage="step_ii"
        )

        # 5. Step-II测试（每5 epoch，论文实验记录）
        if epoch % args.eval_epoch == 0:
            print(f"Step-II Epoch {epoch}：测试跨模态匹配效果...")
            cmc, mAP, mINP, test_stats = tester(
                args=args,
                epoch=epoch,
                main_net=main_net,
                test_loader=test_loader,
                test_info=test_info,
                dataset=args.dataset,
                test_mode=args.mode,
                logger=test_log,
                writer=writer
            )

            # 保存最佳模型
            if test_stats["Rank-1"] > best_rank1:
                best_rank1 = test_stats["Rank-1"]
                best_epoch = epoch
                best_mAP = test_stats["mAP"]
                best_mINP = test_stats["mINP"]
                best_ckpt = {
                    "main_net": main_net.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch,
                    "best_rank1": best_rank1
                }
                torch.save(best_ckpt, os.path.join(model_path, "best_checkpoint.pth"))
                print(f"更新最佳模型：Epoch {epoch}，Rank-1={best_rank1:.2f}%")

            # 保存定期模型
            if epoch % args.save_epoch == 0:
                regular_ckpt = {
                    "main_net": main_net.state_dict(),
                    "cmc": cmc,
                    "mAP": mAP,
                    "mINP": mINP,
                    "epoch": epoch
                }
                torch.save(regular_ckpt, os.path.join(model_path, f"checkpoint_epoch{epoch}.pth"))

            # 打印与记录结果
            print(f"当前测试结果：Rank-1={test_stats['Rank-1']:.2f}% | mAP={test_stats['mAP']:.2f}% | mINP={test_stats['mINP']:.2f}%")
            print(f"最佳结果：Epoch {best_epoch} | Rank-1={best_rank1:.2f}% | mAP={best_mAP:.2f}% | mINP={best_mINP:.2f}%", file=test_log)
            test_log.flush()

        # 6. 定期更新记忆库（Step-II每10 epoch，论文Step21）
        if epoch % (CSANET_CONFIG["memory"]["update_freq"] * 2) == 0:
            print(f"Step-II Epoch {epoch}：更新跨模态记忆库...")
            vis_feats_new = main_net.extract_modal_feats(train_loaders["step_i_vis"], device=device, modal="vis")
            ir_feats_new = main_net.extract_modal_feats(train_loaders["step_i_ir"], device=device, modal="ir")
            main_net.update_memory(vis_feats_new, trainset_step1_vis.train_color_label, ir_feats_new, trainset_step1_ir.train_thermal_label)
            torch.save({"vis_memory": main_net.vis_memory, "ir_memory": main_net.ir_memory}, os.path.join(memory_path, f"memory_epoch{epoch}.pth"))

    ## 6. 训练结束：汇总结果
    print("\n" + "="*60)
    print(f"CSANet训练完成！总epoch：{CSANET_CONFIG['train_stage']['total_epoch']}")
    print(f"最佳实验结果：")
    print(f"  Epoch: {best_epoch}")
    print(f"  Rank-1: {best_rank1:.2f}% | mAP: {best_mAP:.2f}% | mINP: {best_mINP:.2f}%")
    print("="*60)

    # 关闭资源
    test_log.close()
    writer.close()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 命令行参数解析（补充CSANet必要参数）
    parser = argparse.ArgumentParser(description="CSANet Training Pipeline")
    parser.add_argument("--config", default="config/csanet.yaml", help="CSANet配置文件路径")
    parser.add_argument("--resume", action="store_true", help="是否从checkpoint恢复训练")
    parser.add_argument("--resume_path", default="", help="checkpoint路径")
    parser.add_argument("--cudnn_benchmark", action="store_true", default=True, help="启用cudnn加速")
    args_main = parser.parse_args()

    # 加载yaml配置并补充默认参数
# 加载yaml配置并补充默认参数（指定utf-8编码，解决中文/特殊字符解码问题）
    with open(args_main.config, "r", encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)

    # 补充CSANet默认参数（覆盖yaml未定义项）
    if not hasattr(args, "base_lr"):
        args.base_lr = 3e-4  # 论文Transformer骨干基础学习率
    if not hasattr(args, "eval_epoch"):
        args.eval_epoch = 5  # 每5 epoch测试一次
    if not hasattr(args, "save_epoch"):
        args.save_epoch = 10  # 每10 epoch保存一次模型

    # 启动主训练流程
    main_worker(args, args_main)