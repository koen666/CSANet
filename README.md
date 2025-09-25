### 一、当前代码训练前确认
1. **配置文件核对**：确保 `config/config_sysu.yaml` 中关键参数正确：
   - `dataset: sysu`、`dataset_path: ./dataset/`（指向含 `cam1-6`、`exp` 的 SYSU-MM01 目录）；
   - `setting: unsupervised`（当前代码适配的无监督场景）；
   - `base_lr: 3e-4`、`end_epoch: 60`（Step-I 20epoch + Step-II 40epoch）。
2. **数据格式确认**：`SYSU-MM01/exp/train_id.txt` 格式为每行一个身份ID（如 `1,2,3,...`），确保 `data_loader.py` 中 `pre_process_sysu` 函数能正常解析（复用你提供的原版数据加载逻辑）。

### 二、当前代码训练命令
在项目根目录执行（启用CuDNN加速，适配当前代码的分阶段训练逻辑）：
```bash
python main_train.py --config config/config_sysu.yaml --cudnn_benchmark
```
训练过程会自动执行：
- Step-I（单模态聚类）：冻结TBGM/CAP/IPCC，优化可见光特征与红外伪标签；
- Step-II（跨模态关联）：解冻模块，按课程学习（plain→moderate→intricate）优化跨模态损失。

### 三、当前代码测试命令
训练结束后，加载 `best_checkpoint.pth` 测试（路径在 `sysu_unsupervised_csanet_sysu/sysu_save_model/` 下）：
```bash
python main_test.py --config config/config_sysu.yaml --resume --resume_path ./sysu_unsupervised_csanet_sysu/sysu_save_model/best_checkpoint.pth
```
测试会输出 SYSU 数据集 All-Search/Indoor-Search 模式下的 Rank-1、mAP 等核心指标，匹配当前代码的评估逻辑。