# this file is similar as base_dataset.py,
# except that only use ResizeCrop and normalize to data,
# and obtain mrf function

# from .image_folder import make_dataset
import re
from torch.utils.data import Dataset, DataLoader
import os
# from .data_transform import centerAlignAndDataAug, centerAlignAndDataAugVal
import numpy as np
import torch
from ._transforms import build_transforms
from ._samplers import TwoStreamBatchSampler
from .image_folder import make_dataset, walk
from .format_conversion import collate_fn
from .data_transform import centerAlignAndDataAugVal
from . import _transforms as T


class SaeDataset(Dataset):
    def __init__(self, root_dir, phase="train", image_size=256, transforms=None, labeled_num=3):
        """获取数据路径，prior, lookup"""
        # self.opt = args
        # if isinstance(transforms, list):
        #     transforms = build_transforms(transforms)
        # self.transforms = transforms
        # UCMT算法增强
        self.image_size = image_size
        self.pre_transform = self.pre_transform()
        self.augmentation = self.augmentation_transform()
        if phase != 'train':
            self.test_transform = self.test_transform()

        self.dataroot = root_dir
        self.phase = phase
        self.paths = getPath(root_dir, phase, labeled_num)
        # 第一个dict为个体的ED图的先验来自于下标指示的prior map
        # 第二个dict为个体的ES图的先验来自于下标指示的prior map#
        self.indexList = [{
            4: [0, 3, 6, 8],
            5: [0, 1, 3, 5, 8],
            6: [0, 1, 3, 5, 7, 8],
            7: [0, 1, 3, 5, 6, 7, 8],
            8: [0, 1, 3, 4, 5, 6, 7, 8],
            9: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            10: [0, 1, 2, 3, 4, 4, 5, 6, 7, 8],
            11: [0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8],
            12: [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
            13: [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8],
            20: [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]
        },
            {
                4: [0, 2, 4, 7],
                5: [0, 1, 3, 5, 7],
                6: [0, 1, 3, 4, 5, 7],
                7: [0, 1, 2, 3, 4, 5, 7],
                8: [0, 1, 2, 3, 4, 5, 6, 7],
                9: [0, 1, 2, 3, 3, 4, 5, 6, 7],
                10: [0, 1, 2, 3, 3, 4, 4, 5, 6, 7],
                11: [0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7],
                12: [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7],
                13: [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                18: [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7],
            }]
        # self.paths[0][0]获取扩展class下所采集的训练数据路径个数#
        # 每个厂商每个class都有自己的prior
        # self.prior = self.get_prior()
        # if not self.opt.beta == 0:
        #     # 不同vendor，不同class下的状态转移矩阵是不同的#
        #     self.lookup = self.get_lookup()

    def __len__(self):
        return len(self.paths[0][0]) + len(self.paths[1][0])  # len(self.paths[0][0]) * 2

    def __getitem__(self, index):
        """获取第index个体，包括ED和ES类型, 此函数的数据增强要确保不改变原图片的分割结构;
        img of shape (b,3,160,160), \in [0,1]
        label of shape (b,1,160,160), \in {0,1,2,3}
        matchedPrior of shape (b,4,160,160) \in [0,1]
        matchedLookup of shape (b,4,4)
        matched_index=classes=vendor of shape (b,)
        """

        # type --- 0表示'ED/'，1表示'ES/'
        index = int(index)
        if index < len(self.paths[0][0]):
            classes_index = 0
        else:
            index = index - len(self.paths[0][0])
            classes_index = 1

        img_path, label_path, types, vendor = self.paths[classes_index][0][index], self.paths[classes_index][1][index], \
                                              self.paths[classes_index][2], self.paths[classes_index][3][index]

        assert self.paths_match(label_path, img_path), "The label_path %s and img_path %s don't match." % (
        label_path, img_path)
        assert types[0] == classes_index, "obtained dataset is incorrect."
        # img_ori, label_ori, index_ori, types, vendors 是一一对应的
        img, label = np.load(img_path), np.load(label_path)  # 3维np数组
        if np.ndim(img) == 2:
            img = img[None, ...]
            label = label[None, ...]
        # img_size = img.shape[0]
        vendorTotal = torch.tensor([vendor]).repeat(img.shape[0])
        # matchedIndex = np.array(self.indexList[classes_index][img_size])
        # matchedPrior = self.prior[vendor][classes_index][matchedIndex]

        # sample = {'image': img, 'label': label}
        img, label = centerAlignAndDataAugVal(img, label)
        # 将图片转换为[0,1]
        img = img / 255.0
        img, label = img.numpy(), label.numpy()
        # img_tf, label_tf = [], []
        # if self.transforms:
        #     for img_i in range(img.shape[0]):
        #         sample = self.transforms({'image': img[img_i], 'label': label[img_i],
        #                                   'prior': matchedPrior[img_i]})  # 3,224,224 | 1,224,224 | 4,224,224
        #         # prior-based cutout
        #         mask = sample['prior'][0, :, :] == 1
        #         sample['image'][:, mask] = 0.0
        #         img_tf.append(sample['image'].unsqueeze(0))
        #         label_tf.append(sample['label'].unsqueeze(0))
        #
        # sample = {'image': torch.cat(img_tf, dim=0), 'label': torch.cat(label_tf, dim=0), 'size': img_size}
        if self.phase == 'train':
            img_tf, label_tf, ignore_tf = [], [], []
            for img_i in range(img.shape[0]):
                # 弱增强: 缩放+rgb
                sample = self.pre_transform({'image': img[img_i], 'label': label[img_i]})
                # 强增强: crop+flip+colorjitter
                sample = self.augmentation(sample)
                ignore_mask = torch.zeros_like(sample['label'])
                ignore_mask[sample['label'] == 254] = 255

                img_tf.append(sample['image'])
                label_tf.append(sample['label'])
                ignore_tf.append(ignore_mask)

            return torch.stack(img_tf, dim=0).float(), torch.stack(label_tf, dim=0).squeeze(1).long(), \
                   torch.stack(ignore_tf, dim=0).squeeze(1).long()
        else:
            img_tf, label_tf = [], []
            for img_i in range(img.shape[0]):
                # 缩放到模型指定输入大小
                sample = self.pre_transform({'image': img[img_i], 'label': label[img_i]})
                img_tf.append(sample['image'])
                sample = self.test_transform({'image': img[img_i], 'label': label[img_i]})
                label_tf.append(sample['label'])

        return torch.stack(img_tf, dim=0).float(), torch.stack(label_tf, dim=0).squeeze(1).long()  # 513*513, 224*224

    # def get_prior(self):
    #     """ 只是加载处理过的prior
    #     return list([vendor_i-ED-prior,vendor_i-ES-prior])
    #         vendor_i-ED-prior=(9,4,160,160)
    #         vendor_i-ES-prior=(8,4,160,160)
    #     """
    #     prior = []
    #     Vendor = ['A/', 'B/', 'C/', 'D/']
    #     classes = ['ED/', 'ES/']
    #     for vendorIndex in range(4):
    #         classPrior = []
    #         for classIndex in range(2):
    #             classPrior.append(torch.from_numpy(
    #                 np.load(self.dataroot + Vendor[vendorIndex] + classes[classIndex] + 'prior_4chs.npy')))
    #         prior.append(classPrior)
    #     return prior

    # def get_lookup(self):
    #     """to compute the state transition matrix for the prior maps from different vendors and classes
    #     return: total_lookup[vendor_index][class_index].shape=(9or8,4,4)
    #     """
    #     total_lookup = []
    #     vendorNum = 4
    #     for vendorIndex in range(vendorNum):
    #         classLookup = []
    #         for class_index in range(2):
    #             lookup = []
    #             for index in range(self.prior[vendorIndex][class_index].size(0)):
    #                 # 计算每一张prior map的状态转移矩阵 #
    #                 lookup.append(mrf.get_lookup(argmax_ch(self.prior[vendorIndex][class_index][index].unsqueeze(0)).to(torch.uint8).cpu(),
    #                                              neighboor_size=self.opt.k))
    #             lookup = np.concatenate(lookup, axis=0)
    #             classLookup.append(lookup)
    #         total_lookup.append(classLookup)
    #     return total_lookup

    def paths_match(self, label_path, img_path):
        # os.path.basename(path)将指定的路径拆分后返回尾部(基本名称)，是一个字符串值。如把'/home/User/Documents'返回'Documents'；
        # os.path.splitext(path)分离文件名与扩展名,返回元组(fname[包括目录],fextension)；
        # os.path.split(path)返回目录路径和文件名的元组#
        filename1_without_ext = os.path.splitext(os.path.basename(label_path))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        return filename1_without_ext == filename2_without_ext

    # @staticmethod
    def augmentation_transform(self):
        return T.Compose([
            T.RandomCrop(size=[self.image_size, self.image_size]),
            T.RandomFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        ])

    def pre_transform(self):
        return T.Compose([
            T.RandomGenerator(output_size=[self.image_size, self.image_size], p_flip=0.0, p_rot=0.0),
            T.ToRGB(),
        ])

    def test_transform(self):
        return T.Compose([
            T.RandomGenerator(output_size=[224, 224], p_flip=0.0, p_rot=0.0),
            T.ToRGB(),
        ])


def getPath(root, phase, labeled_num):
    """ get the paths of samples for each phase
        return: list([vendor_img,vendor_label,[index_i],vendors])，特别的把属于同一个index的所有路径放到一起
        vendors -- 厂商
        index -- ED or ES
        vendor_img -- the paths of images
        vendor_label -- the paths of labels
    """
    Vendor = ['A/', 'B/', 'C/', 'D/']
    classes = ['ED/', 'ES/']
    paths = []
    seq_paths = []  # 用于确定哪些位置为标签数据
    for index in range(2):
        vendor_img, vendor_label, vendors, tmp_img = [], [], [], []
        for vendorIndex in range(4):
            img_dir = os.path.join(root + Vendor[vendorIndex] + classes[index], '%s_img'%phase)
            tmp_img_paths = walk(dirname=img_dir)
            if phase == "train":
                img_dir = img_dir + '/slices'
            # img_paths = make_dataset(img_dir, recursive=False, read_cache=True)
            img_paths = walk(dirname=img_dir)

            label_dir = os.path.join(root + Vendor[vendorIndex] + classes[index], '%s_label'%phase)
            if phase == "train":
                label_dir = label_dir + '/slices'
            # label_paths = make_dataset(label_dir, recursive=False, read_cache=True)
            label_paths = walk(dirname=label_dir)
            vendor_img.append(img_paths)
            vendor_label.append(label_paths)
            tmp_img.append(tmp_img_paths)
            vendors.append(np.array([vendorIndex]).repeat(len(img_paths)))
        vendor_img = np.concatenate(vendor_img, 0)
        vendor_label = np.concatenate(vendor_label, 0)
        vendors = np.concatenate(vendors, 0)
        vendor_img = sorted(vendor_img)
        vendor_label = sorted(vendor_label)
        tmp_img = np.concatenate(tmp_img, 0)
        tmp_img = sorted(tmp_img)  # 原来以卷为单位的数据排列方式

        paths.append([vendor_img, vendor_label, [index], vendors])
        seq_paths.append(tmp_img)

    if phase == "train":
        if not os.path.exists(root + f'train_stid_{labeled_num}.npy'):
            get_lbid(root, phase, labeled_num, seq_paths, paths)
    print('%d data is created' % (len(paths[0][0]) + len(paths[1][0])))
    return paths


def argmax_ch(input):
    '''
    Pick the most likely class for each pixel
    individual mask: each subjects
    have different uniformly sample mask
    '''
    input = input.detach()
    batch_n, chs, xdim, ydim = input.size()

    # Enumarate the chs #
    # enumerate_ch has dimensions [batch_n, chs, xdim, ydim, zdim]

    arange = torch.arange(0, chs).view(1, -1, 1, 1)
    arange = arange.repeat(batch_n, 1, 1, 1).float()

    enumerate_ch = torch.ones(batch_n, chs, xdim, ydim)

    enumerate_ch = arange * enumerate_ch

    classes = torch.argmax(input, 1).float()

    sample = []
    for c in range(chs):
        _sample = (enumerate_ch[:, c, :, :] == classes).to(torch.uint8)
        # print(_sample.dtype)
        sample += [_sample.unsqueeze(1)]
    sample = torch.cat(sample, 1)
    return sample


# 确定标签id
def get_lbid(root="../MM/", phase="train", labeled_num=4, seq_paths=None, paths=None):
    # 构造采样列表
    ed_size = len(paths[0][0])
    # num_ab = (labeled_num - 4) // 2
    # lb_volid = list(range(0, 0 + num_ab)) + list(range(75, 75 + num_ab)) + \
    #                list(range(150, 150 + 2)) + list(range(160, 160 + 2))
    if labeled_num > 4:
        ab_size = (labeled_num - 4)
        num_a = ab_size // 2
        lb_volid = list(range(1, 1 + num_a)) + list(range(75, 75 + (ab_size - num_a))) + \
                   list(range(150, 150 + 2)) + list(range(160, 160 + 2))  # 与工作2不一样
    elif labeled_num == 4:
        lb_volid = list(range(1, 1 + 1)) + list(range(75, 75 + 1)) + \
                   list(range(150, 150 + 1)) + list(range(160, 160 + 1))  # 与工作2不一样
    else:
        raise ValueError("labeled_num is at least 4.")

    cl = ['ED', 'ES']
    target_id = []
    for vi in lb_volid:
        img_path = seq_paths[0][vi]
        inner_no = os.path.splitext(os.path.basename(img_path))[0]
        s = os.path.split(img_path)[0].split('/')
        vd_no = s[-3]
        # 根据原编号确定组号，类别，和对应编号
        for index in range(2):
            tg_prefix = f'{vd_no}_{cl[index]}_{inner_no}_'
            flag = 0
            for li in range(len(paths[index][0])):
                # gpi_ED/ES_j_k
                if os.path.splitext(os.path.basename(paths[index][0][li]))[0].startswith(tg_prefix):
                    target_id.append(li if index == 0 else li + ed_size)
                    flag = 1
            if flag == 0:
                print(f"{img_path} is not found")

    # 保存target_id方便未来实验的调用
    np.save(root + f'train_stid_{labeled_num}.npy', target_id)
    print(f'saved train_stid_{labeled_num}.npy')


def use_num_eachvendor_and_repeat_time(labeled_num, tot_num=170, batch_size=6):
    num_ab = (labeled_num - 4) // 2
    # 实际使用的带标签样本数
    actual_labeled = 2 * num_ab + 4
    # 确定重复次数，使得无标签样本得到完全使用
    rep_time = 0
    label_iter = actual_labeled * 2
    unlabel_iter = tot_num - label_iter
    # 确定要迭代多少次才能跑完所有无标签
    iter_time = unlabel_iter / (batch_size - 1)
    while True:
        rep_time += 1
        if iter_time <= (label_iter * rep_time):
            break

    return num_ab, actual_labeled, rep_time

def get_cardiac_loaders(root_dir=r'F:/datasets/ACDC/', labeled_num=7, labeled_bs=8, batch_size=24, batch_size_val=16,
                     num_workers=4, worker_init_fn=None, train_transforms=None, val_transforms=None, num_classes=4):
    # ref_dict = {"3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}

    db_train = SaeDataset(root_dir=root_dir, phase="train", image_size=513, labeled_num=labeled_num)  # 769
    db_val = SaeDataset(root_dir=root_dir, phase="val", image_size=513, labeled_num=labeled_num)

    if labeled_bs < batch_size:
        """使用batch_size=6 and labeled_bs=1，那么每一个epoch使用320（out of 324）无标签数据, 从而总共使用336（out of 340）个训练数据"""
        # labeled_slice = ref_dict[str(labeled_num)]
        # 在这部分我们可以重复多次labeled_idxs，从而使unlabeled_idxs能被全部使用
        # labeled_idxs = list([0, 1, 75, 76, 150, 151, 160, 161, 170, 171, 245, 246, 320, 321, 330, 331] * 4) # for batch_size=6 and labeled_bs=1
        # 根据labeled_num确定每个厂商要使用多少个个体，特别的，CD商只使用2个
        # num_ab, actual_labeled, rep_time = use_num_eachvendor_and_repeat_time(labeled_num=labeled_num, tot_num=len(db_train), batch_size=batch_size)
        train_edesid = np.load(root_dir + f'train_stid_{labeled_num}.npy').astype(np.uint16)
        labeled_idxs = np.sort(train_edesid, axis=0)
        if labeled_num >= 10:
            labeled_idxs = labeled_idxs.tolist() * 3
        else:
            labeled_idxs = labeled_idxs.tolist() * 4
        print(f'use {labeled_num} labeled data, has {train_edesid.shape[0]} slices')

        unlabeled_idxs = list(range(0, len(db_train)))
        for counter, index in enumerate(labeled_idxs):
            index = index - counter
            if index < 0:
                break
            unlabeled_idxs.pop(index)
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(db_train, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader







