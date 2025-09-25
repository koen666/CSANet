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
# 保留必要工具，新增CSANet专属测试工具
from utils import Logger, set_seed, GenIdx, format_metrics  # 新增指标格式化工具
from data_loader import TestData, get_adca_transform  # 导入论文ADCA测试增强
from data_manager import process_query_sysu, process_gallery_sysu, process_test_regdb
# 替换模型：删除BaseResNet，导入CSANet（基于TransReID）
from model.network import CSANet
from engine import tester  # 复用适配后的测试器

# -------------------------- CSANet测试全局配置（严格对齐论文） --------------------------
CSANET_TEST_CONFIG = {
    "sysu": {
        "test_modes": ["all", "indoor"],  # 论文SYSU-MM01双测试模式（All-Search/Indoor-Search）
        "query_modal": 2,  # 红外查询（modal=2）
        "gall_modal": 1    # 可见光画廊（modal=1）
    },
    "regdb": {
        "test_modes": ["visibletothermal", "thermaltovisible"],  # 论文RegDB双向测试
        "modal_map": {"visible": 1, "thermal": 2}  # 模态→编号映射
    },
    "feature": {
        "feat_dim": 768  # TransReID输出维度（论文设定）
    },
    "eval": {
        "report_ranks": [1, 5, 10, 20]  # 论文重点报告的Rank指标
    }
}

def main_worker(args, args_main):
    ## 1. 基础配置初始化（设备、种子、路径）
    # GPU与种子配置（确保测试可复现，论文要求）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = args.cudnn_benchmark if hasattr(args, "cudnn_benchmark") else True
    set_seed(args.seed, cuda=torch.cuda.is_available())

    # 路径配置（兼容原有逻辑，补充结果保存路径）
    exp_dir = f"{args.dataset}_{args.setting}_{args.file_name}"
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    log_path = os.path.join(exp_dir, f"{args.dataset}_{args.log_path}")
    result_path = os.path.join(exp_dir, "test_results")  # 新增测试结果保存路径
    for path in [log_path, result_path]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # 日志初始化（输出到文件与控制台）
    sys.stdout = Logger(os.path.join(log_path, "log_test.txt"))
    print(f"实验配置：\nargs_main: {args_main}\nargs: {args}\n")

    ## 2. 数据加载（适配论文测试增强与协议）
    print("==> 加载测试数据集...")
    t_start = time.time()

    # 数据增强：替换为论文ADCA测试增强（避免测试数据分布偏移）
    transform_test = get_adca_transform(args.img_w, args.img_h, is_train=False)

    # 加载测试集（按数据集适配论文协议）
    test_loader = {}
    test_info = {}
    if args.dataset == "sysu":
        data_path = os.path.join(args.dataset_path, "SYSU-MM01/")
        # 论文SYSU-MM01双测试模式：All-Search（默认）与Indoor-Search
        target_mode = args.mode if args.mode in CSANET_TEST_CONFIG["sysu"]["test_modes"] else "all"
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=target_mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=target_mode)
        # 补充相机ID校验（确保与论文一致：红外查询相机3/6，可见光画廊相机1/2/4/5）
        valid_query_cams = [3, 6]  # 论文SYSU红外查询相机
        valid_gall_cams = CSANET_TEST_CONFIG["sysu"]["gall_modal"]  # 可见光画廊相机
        query_valid_mask = np.isin(query_cam, valid_query_cams)
        gall_valid_mask = np.isin(gall_cam, valid_gall_cams)
        # 筛选有效测试样本
        query_img = [query_img[i] for i in np.where(query_valid_mask)[0]]
        query_label = query_label[query_valid_mask]
        query_cam = query_cam[query_valid_mask]
        gall_img = [gall_img[i] for i in np.where(gall_valid_mask)[0]]
        gall_label = gall_label[gall_valid_mask]
        gall_cam = gall_cam[gall_valid_mask]

    elif args.dataset == "regdb":
        data_path = os.path.join(args.dataset_path, "RegDB/")
        # 论文RegDB双向测试：按mode区分查询/画廊模态
        query_modal = args.mode.split("to")[0]
        gall_modal = args.mode.split("to")[1]
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modality=query_modal)
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modality=gall_modal)
        # 补充RegDB相机ID（论文设定：visible=1，thermal=2）
        query_cam = np.ones_like(query_label) * CSANET_TEST_CONFIG["regdb"]["modal_map"][query_modal]
        gall_cam = np.ones_like(gall_label) * CSANET_TEST_CONFIG["regdb"]["modal_map"][gall_modal]

    # 测试集加载（统一格式）
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

    # 测试集统计（论文实验记录要求）
    print(f"数据集统计（{args.dataset}，测试模式：{args.mode}）：")
    print("  ----------------------------")
    print("  subset   | # ids | # images")
    print("  ----------------------------")
    print(f"  query    | {len(np.unique(query_label)):5d} | {len(query_label):8d}")
    print(f"  gallery  | {len(np.unique(gall_label)):5d} | {len(gall_label):8d}")
    print("  ----------------------------")
    print(f"数据加载耗时：{time.time() - t_start:.3f}s\n")

    ## 3. Checkpoint加载与模型初始化（适配CSANet）
    print("==> 加载模型Checkpoint...")
    # 初始化模型参数
    n_class = 0  # 从checkpoint提取类别数
    epoch = 0    # 从checkpoint提取epoch
    resume_path = args_main.resume_path if args_main.resume else ""

    # 加载checkpoint（兼容不同保存格式）
    if args_main.resume and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        # 提取类别数（从分类器权重维度）
        if "main_net" in checkpoint:
            classif_key = "classifier.weight" if "classifier.weight" in checkpoint["main_net"] else "fc.weight"
            n_class = checkpoint["main_net"][classif_key].size(0)
        elif "net" in checkpoint:
            classif_key = "classifier.weight" if "classifier.weight" in checkpoint["net"] else "fc.weight"
            n_class = checkpoint["net"][classif_key].size(0)
        else:
            raise KeyError("Checkpoint中未找到模型参数（需包含'main_net'或'net'键）")
        # 提取epoch
        epoch = checkpoint.get("epoch", 0)
        print(f"成功加载Checkpoint：{resume_path}")
        print(f"  Epoch: {epoch} | 类别数: {n_class} | 设备: {device}")
    else:
        raise RuntimeError(f"未指定有效Checkpoint路径（resume={args_main.resume}，path={resume_path}）")

    # 初始化CSANet模型（替换原有BaseResNet，基于TransReID）
    main_net = CSANet(
        class_num=n_class,
        es1=20,  # Step-I epoch数（测试时仅用于模型结构初始化）
        es2=40   # Step-II epoch数
    ).to(device)

    # 加载模型权重
    if "main_net" in checkpoint:
        main_net.load_state_dict(checkpoint["main_net"])
    elif "net" in checkpoint:
        main_net.load_state_dict(checkpoint["net"])
    main_net.eval()  # 切换至评估模式
    print("模型初始化完成，已切换至评估模式\n")

    ## 4. 测试模式适配（对齐论文模态设定）
    if args.dataset == "sysu":
        # SYSU测试：红外查询（modal=2）→ 可见光画廊（modal=1）
        test_mode = [CSANET_TEST_CONFIG["sysu"]["gall_modal"], CSANET_TEST_CONFIG["sysu"]["query_modal"]]
        print(f"SYSU-MM01测试模式：{target_mode}（查询模态：红外，画廊模态：可见光）")
    elif args.dataset == "regdb":
        # RegDB测试：按mode动态设定模态
        query_modal_code = CSANET_TEST_CONFIG["regdb"]["modal_map"][query_modal]
        gall_modal_code = CSANET_TEST_CONFIG["regdb"]["modal_map"][gall_modal]
        test_mode = [gall_modal_code, query_modal_code]
        print(f"RegDB测试模式：{args.mode}（查询模态：{query_modal}，画廊模态：{gall_modal}）")

    ## 5. 执行测试（调用适配后的tester，对齐论文评估逻辑）
    print(f"\n==> 开始测试（Epoch: {epoch}）...")
    t_start_test = time.time()

    # 调用tester获取评估结果
    if args.dataset == "sysu":
        # SYSU需传递相机ID，过滤同相机样本（论文要求）
        cmc, mAP, mINP, test_stats = tester(
            args=args,
            epoch=epoch,
            main_net=main_net,
            test_loader=test_loader,
            test_info=test_info,
            dataset=args.dataset,
            test_mode=target_mode,  # 传递具体测试模式（all/indoor）
            feat_dim=CSANET_TEST_CONFIG["feature"]["feat_dim"],
            logger=sys.stdout
        )
    elif args.dataset == "regdb":
        # RegDB需传递双向模式信息
        cmc, mAP, mINP, test_stats = tester(
            args=args,
            epoch=epoch,
            main_net=main_net,
            test_loader=test_loader,
            test_info=test_info,
            dataset=args.dataset,
            test_mode=args.mode,  # 传递双向模式（visibletothermal/thermaltovisible）
            feat_dim=CSANET_TEST_CONFIG["feature"]["feat_dim"],
            logger=sys.stdout
        )

    # 测试耗时统计
    test_time = time.time() - t_start_test
    print(f"测试耗时：{test_time:.3f}s")

    ## 6. 结果输出与保存（对齐论文格式）
    print("\n" + "="*80)
    print(f"{args.dataset}数据集测试结果（Epoch: {epoch}，模式: {args.mode}）")
    print("="*80)
    # 按论文格式打印关键指标（Rank-1/5/10/20、mAP、mINP）
    for rank in CSANET_TEST_CONFIG["eval"]["report_ranks"]:
        if rank-1 < len(cmc):
            print(f"Rank-{rank:2d}: {cmc[rank-1]:6.2%}")
    print(f"mAP    : {mAP:6.2%}")
    print(f"mINP   : {mINP:6.2%}")
    print("="*80)

    # 保存测试结果到文件（便于论文表格整理）
    result_file = os.path.join(result_path, f"test_result_epoch{epoch}_{args.mode}.txt")
    with open(result_file, "w") as f:
        f.write(f"Checkpoint Path: {resume_path}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Test Mode: {args.mode}\n")
        f.write(f"Test Time: {test_time:.3f}s\n")
        f.write("\nPerformance:\n")
        for rank in CSANET_TEST_CONFIG["eval"]["report_ranks"]:
            if rank-1 < len(cmc):
                f.write(f"Rank-{rank}: {cmc[rank-1]:.4f}\n")
        f.write(f"mAP: {mAP:.4f}\n")
        f.write(f"mINP: {mINP:.4f}\n")
    print(f"\n测试结果已保存至：{result_file}")

if __name__ == "__main__":
    # 命令行参数解析（补充CSANet测试必要参数）
    parser = argparse.ArgumentParser(description="CSANet Testing Pipeline")
    parser.add_argument("--config", default="config/csanet.yaml", help="CSANet配置文件路径")
    parser.add_argument("--resume", action="store_true", required=True, help="必须从Checkpoint加载模型（测试必备）")
    parser.add_argument("--resume_path", default="", required=True, help="Checkpoint路径（测试必备）")
    parser.add_argument("--cudnn_benchmark", action="store_true", default=True, help="启用cudnn加速测试")
    args_main = parser.parse_args()

    # 加载yaml配置并补充默认参数
    with open(args_main.config, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = easydict.EasyDict(args)

    # 补充CSANet测试默认参数（覆盖yaml未定义项）
    if not hasattr(args, "img_w"):
        args.img_w = 144  # 论文VI-ReID标准宽度
    if not hasattr(args, "img_h"):
        args.img_h = 288  # 论文VI-ReID标准高度
    if not hasattr(args, "test_batch_size"):
        args.test_batch_size = 32  # 测试批大小（论文默认）
    if not hasattr(args, "workers"):
        args.workers = 4  # 数据加载线程数
    if not hasattr(args, "seed"):
        args.seed = 42  # 随机种子（确保可复现）
    if not hasattr(args, "pool_dim"):
        args.pool_dim = CSANET_TEST_CONFIG["feature"]["feat_dim"]  # 特征维度（768）

    # 启动测试流程
    main_worker(args, args_main)