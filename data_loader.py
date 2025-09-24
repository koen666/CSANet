import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import collections
from sklearn.cluster import DBSCAN  # 补充DBSCAN聚类（论文🔶1-131）
from sklearn.metrics.pairwise import cosine_similarity  # 用于聚类评估
import torchvision.transforms as transforms  # 补充数据增强


def generate_unsupervised_pseudo_label(feats, dataset_name):
    """
    按论文要求用DBSCAN生成无监督伪标签（🔶1-131、🔶1-209）
    feats: 单模态特征集合（shape: [N, d]，N为样本数，d为特征维度）
    dataset_name: 数据集名称（"sysu"或"regdb"），用于区分DBSCAN参数
    """
    # 论文设定：SYSU-MM01的DBSCAN最大距离设为0.6，RegDB设为0.3（🔶1-209）
    eps = 0.6 if dataset_name == "sysu" else 0.3
    min_samples = 2  # 最小聚类样本数（避免单点聚类）
    
    # 执行DBSCAN聚类（按论文用欧氏距离，与特征计算逻辑一致）
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    pseudo_labels = db.fit_predict(feats)  # 输出伪标签（-1表示噪声样本）
    
    # 过滤噪声样本（伪标签=-1）和样本数≤1的聚类（论文🔶1-96剔除复杂/异常样本）
    label_count = collections.Counter(pseudo_labels)
    valid_labels = [label for label, count in label_count.items() if count > 1 and label != -1]
    mask = np.array([label in valid_labels for label in pseudo_labels])
    
    # 重新映射伪标签（确保标签从0连续递增，便于后续记忆库构建）
    valid_pseudo_labels = pseudo_labels[mask]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(valid_labels))}
    remapped_labels = np.array([label_mapping[label] for label in valid_pseudo_labels])
    
    return mask, remapped_labels  # mask: 有效样本掩码；remapped_labels: 重映射后的伪标签


def mask_outlier(pseudo_labels):
    """
    保留baseline函数，但优化逻辑：仅过滤样本数≤1的聚类（与论文异常样本过滤互补）
    """
    label_count = collections.Counter(pseudo_labels)
    valid_labels = [label for label, count in label_count.items() if count > 1]
    return np.array([label in valid_labels for label in pseudo_labels])

def get_adca_transform(img_w, img_h, is_train=True):
    """
    按论文要求实现ADCA的数据增强策略（🔶1-209）
    is_train: True为训练增强，False为测试增强（仅归一化）
    """
    if is_train:
        transform = transforms.Compose([
            transforms.ToPILImage(),  # 先转PIL（因输入为numpy数组）
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转（ADCA核心增强）
            transforms.RandomResizedCrop((img_h, img_w), scale=(0.8, 1.0)),  # 随机裁剪
            transforms.ToTensor(),  # 转Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #  ImageNet归一化（ADCA同策略）
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_h, img_w), Image.LANCZOS),  # 测试仅resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

#得到的是图像和对应标签的集合
def read_image(data_files, pid2label, img_w, img_h):
    train_img = []
    train_label = []
    for img_path in data_files:
        # img
        img = Image.open(img_path)
        img = img.resize((img_w, img_h), Image.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)

    return np.array(train_img), np.array(train_label)


#得到图像路径和对应标签输出类似/images/cam1/0001_0001.jpg 0，分别得到file_image=/images/cam1/0001_0001.jpg file_label=0
def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def pre_process_sysu(args, data_dir):
    rgb_cameras = ["cam1", "cam2", "cam4", "cam5"]
    ir_cameras = ["cam3", "cam6"]

    # load id info
    file_path_train = os.path.join(data_dir, "exp/train_id.txt")
    file_path_val = os.path.join(data_dir, "exp/val_id.txt")
    # 示例：
    # 文件 train_id.txt 内容：
    # 1,2,5,10
    # 处理后：
    # ids = [1,2,5,10]
    # id_train = ["0001","0002","0005","0010"]
    with open(file_path_train, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_train = ["%04d" % x for x in ids]

    with open(file_path_val, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_val = ["%04d" % x for x in ids]

    # combine train and val split
    id_train.extend(id_val)
    # 假设 cam1/0001/ 目录下有：
    # 0001_0001.jpg, 0001_0002.jpg
    # files_rgb 会增加：
    # ["cam1/0001/0001_0001.jpg", "cam1/0001/0001_0002.jpg"]
    files_rgb = []
    files_ir = []
    for id in sorted(id_train):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_dir, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.extend(new_files)

        for cam in ir_cameras:
            img_dir = os.path.join(data_dir, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)

    # relabel
    pid_container = set()
    for img_path in files_ir:
        pid = int(img_path[-13:-9])
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    train_color_image, train_color_label = read_image(files_rgb, pid2label, args.img_w, args.img_h)
    # np.save(os.path.join(data_dir, 'train_rgb_resized_img.npy'), train_color_image)
    # np.save(os.path.join(data_dir, 'train_rgb_resized_label.npy'), train_color_label)
    train_thermal_image, train_thermal_label = read_image(files_ir, pid2label, args.img_w, args.img_h)
    # np.save(os.path.join(data_dir, 'train_ir_resized_img.npy'), train_thermal_image)
    # np.save(os.path.join(data_dir, 'train_ir_resized_label.npy'), train_thermal_label)

    return train_color_image, train_color_label, train_thermal_image, train_thermal_label


def pre_process_regdb(args, data_dir):
    train_color_list = os.path.join(data_dir, "idx/train_visible_{}".format(args.trial) + ".txt")
    train_thermal_list = os.path.join(data_dir, "idx/train_thermal_{}".format(args.trial) + ".txt")

    color_img_file, train_color_label = load_data(train_color_list)
    thermal_img_file, train_thermal_label = load_data(train_thermal_list)

    train_color_image = []
    for i in range(len(color_img_file)):
        img = Image.open(data_dir + color_img_file[i])
        img = img.resize((args.img_w, args.img_h), Image.LANCZOS)
        pix_array = np.array(img)
        train_color_image.append(pix_array)
    train_color_image = np.array(train_color_image)
    train_color_label = np.array(train_color_label)

    train_thermal_image = []
    for i in range(len(thermal_img_file)):
        img = Image.open(data_dir + thermal_img_file[i])
        img = img.resize((args.img_w, args.img_h), Image.LANCZOS)
        pix_array = np.array(img)
        train_thermal_image.append(pix_array)
    train_thermal_image = np.array(train_thermal_image)
    train_thermal_label = np.array(train_thermal_label)

    return train_color_image, train_color_label, train_thermal_image, train_thermal_label


class SYSUData(data.Dataset):
    def __init__(self, args, data_dir, transform_train=None, is_train=True, for_memory=False):
        """
        优化点：
        1. 双模态（VIS/IR）独立加载、独立存储（🔶1-88）
        2. 无监督场景下通过DBSCAN生成伪标签（🔶1-131）
        3. 支持输出单模态完整数据（for_memory=True时，用于Step-I记忆库初始化）（🔶1-132）
        """
        self.args = args
        self.is_train = is_train
        self.for_memory = for_memory  # 是否为记忆库提供完整数据
        self.data_dir = data_dir
        self.rgb_cameras = ["cam1", "cam2", "cam4", "cam5"]  # 论文SYSU-MM01可见光相机（🔶1-204）
        self.ir_cameras = ["cam3", "cam6"]  # 论文SYSU-MM01红外相机（🔶1-204）
        
        # Step1: 加载原始数据（双模态独立加载）
        self.train_vis_img, self.train_vis_raw_label = self._load_single_modal("vis")  # 可见光原始数据（无标签时为占位符）
        self.train_ir_img, self.train_ir_raw_label = self._load_single_modal("ir")    # 红外原始数据
        
        # Step2: 无监督场景下生成伪标签（论文核心逻辑）
        if args.setting == "unsupervised":
            # 注意：CSANet Step-I需先提取单模态特征，再聚类（此处先预留接口，后续结合模型特征提取）
            # 若需提前生成伪标签，需先加载骨干网络提取特征（示例逻辑如下）
            if args.pretrain_feat_path is not None:
                # 加载预提取的单模态特征（用于DBSCAN聚类）
                vis_feats = np.load(os.path.join(data_dir, args.vis_feat_path))
                ir_feats = np.load(os.path.join(data_dir, args.ir_feat_path))
                
                # 生成可见光伪标签（按论文DBSCAN参数）
                vis_mask, self.train_vis_label = generate_unsupervised_pseudo_label(vis_feats, "sysu")
                self.train_vis_img = self.train_vis_img[vis_mask]  # 过滤异常样本
                
                # 生成红外伪标签
                ir_mask, self.train_ir_label = generate_unsupervised_pseudo_label(ir_feats, "sysu")
                self.train_ir_img = self.train_ir_img[ir_mask]
        else:
            # 有监督场景（仅用于对比实验）
            self.train_vis_label = self.train_vis_raw_label
            self.train_ir_label = self.train_ir_raw_label
        
        # Step3: 加载论文要求的数据增强（ADCA策略）
        self.transform = get_adca_transform(args.img_w, args.img_h, is_train=is_train)

    def _load_single_modal(self, modal_type):
        """
        按模态加载SYSU-MM01数据（论文数据组织结构：cam/id/img.jpg）（🔶1-204）
        modal_type: "vis"（可见光）或"ir"（红外）
        """
        # 加载训练ID（论文train_id.txt/val_id.txt格式）（🔶1-204）
        file_path_train = os.path.join(self.data_dir, "exp/train_id.txt")
        file_path_val = os.path.join(self.data_dir, "exp/val_id.txt")
        
        with open(file_path_train, 'r') as f:
            ids_train = [f"%04d" % int(x) for x in f.read().splitlines()[0].split(',')]
        with open(file_path_val, 'r') as f:
            ids_val = [f"%04d" % int(x) for x in f.read().splitlines()[0].split(',')]
        ids_all = sorted(ids_train + ids_val)  # 合并训练+验证集（论文设定）
        
        # 选择对应模态的相机
        cameras = self.rgb_cameras if modal_type == "vis" else self.ir_cameras
        img_list = []
        label_list = []
        
        for pid in ids_all:
            for cam in cameras:
                img_dir = os.path.join(self.data_dir, cam, pid)
                if not os.path.isdir(img_dir):
                    continue
                # 按论文要求排序图像（确保一致性）
                imgs = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
                img_list.extend(imgs)
                # 提取PID（论文图像命名格式：xxx_xxxx.jpg，PID为后4位前的数字）（🔶1-204）
                pids = [int(img_path[-13:-9]) for img_path in imgs]
                label_list.extend(pids)
        
        # 加载图像并转为numpy数组
        img_array = []
        for img_path in img_list:
            img = Image.open(img_path).convert("RGB")  # 统一转为RGB（红外图单通道需扩展为3通道）
            img = img.resize((self.args.img_w, self.args.img_h), Image.LANCZOS)
            img_array.append(np.array(img))
        img_array = np.array(img_array)
        label_array = np.array(label_list)
        
        return img_array, label_array

    def __getitem__(self, index):
        """
        输出逻辑：
        - 训练时：返回双模态样本（VIS+IR），用于跨模态关联学习（🔶1-122）
        - 记忆库初始化时（for_memory=True）：返回单模态样本+标签，用于构建记忆库（🔶1-132）
        """
        if self.for_memory:
            # 为记忆库提供单模态完整数据（Step-I需分别构建VIS/IR记忆库）
            if self.args.current_modal == "vis":
                img = self.train_vis_img[index]
                label = self.train_vis_label[index]
            else:
                img = self.train_ir_img[index]
                label = self.train_ir_label[index]
            img = self.transform(img)
            return img, label
        else:
            # 训练时双模态采样（确保batch内有两类模态）
            # 可见光样本（按索引取）
            vis_img = self.train_vis_img[index % len(self.train_vis_img)]
            vis_label = self.train_vis_label[index % len(self.train_vis_img)]
            # 红外样本（随机取，模拟论文跨模态匹配场景）
            ir_idx = np.random.randint(len(self.train_ir_img))
            ir_img = self.train_ir_img[ir_idx]
            ir_label = self.train_ir_label[ir_idx]
            
            # 数据增强
            vis_img = self.transform(vis_img)
            ir_img = self.transform(ir_img)
            
            return vis_img, ir_img, vis_label, ir_label

    def __len__(self):
        """无监督训练时按可见光样本数定长（红外样本随机匹配）"""
        return len(self.train_vis_img) if self.args.setting == "unsupervised" else max(len(self.train_vis_img), len(self.train_ir_img))


class RegDBData(data.Dataset):
    def __init__(self, args, data_dir, transform_train=None, is_train=True, for_memory=False):
        """
        适配论文RegDB设定：412个身份，每身份10张VIS+10张thermal，10次随机划分（🔶1-205）
        """
        self.args = args
        self.is_train = is_train
        self.for_memory = for_memory
        self.data_dir = data_dir
        self.trial = args.trial  # 论文要求的10次随机划分（🔶1-205）
        
        # Step1: 加载RegDB模态列表文件（论文格式：idx/train_visible_{trial}.txt）（🔶1-205）
        vis_list_path = os.path.join(data_dir, f"idx/train_visible_{self.trial}.txt")
        ir_list_path = os.path.join(data_dir, f"idx/train_thermal_{self.trial}.txt")
        
        # 加载可见光数据
        self.train_vis_img, self.train_vis_raw_label = self._load_from_list(vis_list_path)
        # 加载红外数据
        self.train_ir_img, self.train_ir_raw_label = self._load_from_list(ir_list_path)
        
        # Step2: 无监督场景生成伪标签（论文DBSCAN参数：RegDB设为0.3）（🔶1-209）
        if args.setting == "unsupervised":
            if args.pretrain_feat_path is not None:
                vis_feats = np.load(os.path.join(data_dir, args.vis_feat_path))
                ir_feats = np.load(os.path.join(data_dir, args.ir_feat_path))
                
                # 生成伪标签（RegDB的DBSCAN eps=0.3）
                vis_mask, self.train_vis_label = generate_unsupervised_pseudo_label(vis_feats, "regdb")
                self.train_vis_img = self.train_vis_img[vis_mask]
                
                ir_mask, self.train_ir_label = generate_unsupervised_pseudo_label(ir_feats, "regdb")
                self.train_ir_img = self.train_ir_img[ir_mask]
        else:
            self.train_vis_label = self.train_vis_raw_label
            self.train_ir_label = self.train_ir_raw_label
        
        # Step3: 加载ADCA数据增强
        self.transform = get_adca_transform(args.img_w, args.img_h, is_train=is_train)

    def _load_from_list(self, list_path):
        """按RegDB列表文件加载数据（文件格式：img_path label）（🔶1-205）"""
        with open(list_path, 'r') as f:
            lines = f.read().splitlines()
            img_paths = [os.path.join(self.data_dir, line.split(' ')[0]) for line in lines]
            labels = [int(line.split(' ')[1]) for line in lines]
        
        # 加载图像并转为numpy数组（RegDB红外图为单通道，需扩展为3通道）
        img_array = []
        for img_path in img_paths:
            img = Image.open(img_path)
            if img.mode == "L":  # 红外图单通道
                img = img.convert("RGB")
            img = img.resize((self.args.img_w, self.args.img_h), Image.LANCZOS)
            img_array.append(np.array(img))
        img_array = np.array(img_array)
        label_array = np.array(labels)
        
        return img_array, label_array

    def __getitem__(self, index):
        """与SYSUData逻辑一致，适配RegDB数据长度"""
        if self.for_memory:
            if self.args.current_modal == "vis":
                img = self.train_vis_img[index]
                label = self.train_vis_label[index]
            else:
                img = self.train_ir_img[index]
                label = self.train_ir_label[index]
            img = self.transform(img)
            return img, label
        else:
            # RegDB训练时双模态按索引匹配（每身份对应10张VIS+10张IR）
            vis_img = self.train_vis_img[index % len(self.train_vis_img)]
            vis_label = self.train_vis_label[index % len(self.train_vis_img)]
            ir_idx = index % len(self.train_ir_img)  # 按索引匹配（论文RegDB数据对齐特性）
            ir_img = self.train_ir_img[ir_idx]
            ir_label = self.train_ir_label[ir_idx]
            
            vis_img = self.transform(vis_img)
            ir_img = self.transform(ir_img)
            
            return vis_img, ir_img, vis_label, ir_label

    def __len__(self):
        return max(len(self.train_vis_img), len(self.train_ir_img))


class TestData(data.Dataset):
    def __init__(self, args, test_modal_type, data_dir, transform_test=None):
        """
        优化点：
        1. 按论文测试模态区分（VIS查询/IR画廊，或反之）（🔶1-204、🔶1-205）
        2. 严格遵循论文测试图像resize逻辑
        """
        self.args = args
        self.test_modal_type = test_modal_type  # "vis_query"（可见光查询）、"ir_gallery"（红外画廊）等
        self.data_dir = data_dir
        
        # 加载测试列表（按数据集区分）
        if args.dataset == "sysu":
            self.test_img_file, self.test_label = self._load_sysu_test()
        else:  # regdb
            self.test_img_file, self.test_label = self._load_regdb_test()
        
        # 加载测试增强（仅resize+归一化，论文要求）
        self.transform_test = transform_test if transform_test is not None else get_adca_transform(
            args.img_w, args.img_h, is_train=False
        )
        
        # 预加载测试图像（避免测试时反复IO）
        self.test_image = []
        for img_path in self.test_img_file:
            img = Image.open(img_path)
            if img.mode == "L":
                img = img.convert("RGB")
            img = img.resize((args.img_w, args.img_h), Image.LANCZOS)
            self.test_image.append(np.array(img))
        self.test_image = np.array(self.test_image)

    def _load_sysu_test(self):
        """加载SYSU-MM01测试数据（论文分All-Search/Indoor-Search）（🔶1-204）"""
        # 论文测试集：96个身份，IR查询+VIS画廊（或反之）
        test_list_path = os.path.join(self.data_dir, f"exp/test_{self.test_modal_type}.txt")
        with open(test_list_path, 'r') as f:
            lines = f.read().splitlines()
            img_files = [os.path.join(self.data_dir, line.split(' ')[0]) for line in lines]
            labels = [int(line.split(' ')[1]) for line in lines]
        return img_files, labels

    def _load_regdb_test(self):
        """加载RegDB测试数据（论文分VIS→Thermal/Thermal→VIS）（🔶1-205）"""
        test_list_path = os.path.join(self.data_dir, f"idx/test_{self.test_modal_type}_{self.args.trial}.txt")
        with open(test_list_path, 'r') as f:
            lines = f.read().splitlines()
            img_files = [os.path.join(self.data_dir, line.split(' ')[0]) for line in lines]
            labels = [int(line.split(' ')[1]) for line in lines]
        return img_files, labels

    def __getitem__(self, index):
        img = self.test_image[index]
        label = self.test_label[index]
        img = self.transform_test(img)
        return img, label

    def __len__(self):
        return len(self.test_image)