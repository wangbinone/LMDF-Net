import os
from torch.utils.data import Dataset
from PIL import Image
import natsort
import random


def default_loader(path1, path2, crop=False, resize=True, crop_size=(256, 256), resize_size=256):
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    w, h = img1.size
    cw, ch = crop_size

    # 1. 先调整大小 (如果开启 resize 或 crop尺寸大于原图)
    if resize or (crop and (w < cw or h < ch)):
        # 计算缩放比例，保持长宽比，确保缩放后能覆盖裁剪区域
        scale = max(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img1 = img1.resize((new_w, new_h), Image.BICUBIC)
        img2 = img2.resize((new_w, new_h), Image.BICUBIC)
        w, h = new_w, new_h  # 更新当前尺寸

    # 2. 随机裁剪
    if crop:
        i = random.randint(0, h - ch)
        j = random.randint(0, w - cw)
        img1 = img1.crop((j, i, j + cw, i + ch))
        img2 = img2.crop((j, i, j + cw, i + ch))

    # ==========================================================
    # (已删除) 3. 强制32倍数裁剪逻辑已移除！
    # 现在直接返回原汁原味的图像尺寸，把处理 32 倍数的任务完全交给 test.py 的 Padding 环节。
    # ==========================================================

    return img1, img2


class myImageFlodertest(Dataset):
    def __init__(self, root, transform=None, resize=False, resize_size=512):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root, 'hazy')
        self.gt_dir = os.path.join(root, 'GT')
        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images = natsort.natsorted(os.listdir(self.gt_dir))
        self.resize = resize
        self.resize_size = resize_size

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        # 1. 获取原文件名 (例如 "Haze_01.png")
        filename_with_ext = self.hazy_images[idx]

        # 2. 去除后缀，只保留名字 (例如 "Haze_01")
        img_name, _ = os.path.splitext(filename_with_ext)

        hazy_image_path = os.path.join(self.hazy_dir, filename_with_ext)
        gt_image_path = os.path.join(self.gt_dir, self.gt_images[idx])

        # 测试阶段默认 crop=False, resize 默认也是 False，直接返回原始尺寸
        hazy_image, gt_image = default_loader(hazy_image_path, gt_image_path,
                                              crop=False, resize=self.resize,
                                              resize_size=self.resize_size)

        if self.transform:
            hazy_image = self.transform(hazy_image)
            gt_image = self.transform(gt_image)

        # 3. 将 img_name 作为第三个参数返回
        return hazy_image, gt_image, img_name

class myImageFloderval(Dataset):
    def __init__(self, root, transform=None, resize=False, resize_size=256):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root, 'hazy')
        self.gt_dir = os.path.join(root, 'GT')
        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images = natsort.natsorted(os.listdir(self.gt_dir))
        self.resize = resize
        self.resize_size = resize_size


    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        gt_image_path = os.path.join(self.gt_dir, self.gt_images[idx])

        hazy_image, gt_image = default_loader(hazy_image_path, gt_image_path,
                                              crop=False, resize=self.resize,
                                              resize_size=self.resize_size,
                                              )

        if self.transform:
            hazy_image = self.transform(hazy_image)
            gt_image = self.transform(gt_image)

        return hazy_image, gt_image

class myImageFlodertrain(Dataset):
    def __init__(self, root, transform=None, crop=False, resize=False,
                 crop_size=(400, 400), resize_size=400, random_flip=False):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root, 'train', 'hazy')
        self.gt_dir = os.path.join(root, 'train', 'GT')
        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images = natsort.natsorted(os.listdir(self.gt_dir))
        self.crop = crop
        self.resize = resize
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.random_flip = random_flip   # 新增参数：控制是否启用随机翻转

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        gt_image_path = os.path.join(self.gt_dir, self.gt_images[idx])

        hazy_image, gt_image = default_loader(
            hazy_image_path, gt_image_path,
            crop=self.crop, resize=self.resize,
            crop_size=self.crop_size, resize_size=self.resize_size,
        )

        # 随机水平翻转（50% 概率）
        if self.random_flip and random.random() > 0.5:
            hazy_image = hazy_image.transpose(Image.FLIP_LEFT_RIGHT)
            gt_image = gt_image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            hazy_image = self.transform(hazy_image)
            gt_image = self.transform(gt_image)

        return hazy_image, gt_image

class myImageFlodertrainDHID(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        crop=True,
        resize=False,
        crop_size=(400, 400),
        resize_size=256,
        random_flip=False
    ):
        self.root = root
        self.transform = transform
        self.hazy_dir = os.path.join(root, 'train', 'hazy')
        self.gt_dir   = os.path.join(root, 'train', 'GT')

        self.hazy_images = natsort.natsorted(os.listdir(self.hazy_dir))
        self.gt_images   = natsort.natsorted(os.listdir(self.gt_dir))

        self.crop = crop
        self.resize = resize
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.random_flip = random_flip

        # GT -> hazy 映射
        self.gt_to_hazy = self._create_gt_to_hazy_mapping()

    def _create_gt_to_hazy_mapping(self):
        gt_to_hazy = {}
        for hazy_image in self.hazy_images:
            gt_image = hazy_image.split('_')[0] + '.jpg'
            if gt_image not in gt_to_hazy:
                gt_to_hazy[gt_image] = []
            gt_to_hazy[gt_image].append(hazy_image)
        return gt_to_hazy

    def __len__(self):
        # 一张GT对应10张hazy，所以总长度就是 hazy 的数量
        return sum(len(hazy_list) for hazy_list in self.gt_to_hazy.values())

    def __getitem__(self, idx):
        gt_image_idx = idx // 10   # 每10个样本对应一个GT
        hazy_image_idx = idx % 10  # 对应的第几张hazy

        gt_image_path   = os.path.join(self.gt_dir, self.gt_images[gt_image_idx])
        hazy_image_list = self.gt_to_hazy[self.gt_images[gt_image_idx]]
        hazy_image_path = os.path.join(self.hazy_dir, hazy_image_list[hazy_image_idx])

        hazy_image, gt_image = default_loader(
            hazy_image_path, gt_image_path,
            crop=self.crop, resize=self.resize,
            crop_size=self.crop_size, resize_size=self.resize_size
        )

        # 随机翻转
        if self.random_flip and random.random() > 0.5:
            hazy_image = hazy_image.transpose(Image.FLIP_LEFT_RIGHT)
            gt_image   = gt_image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            hazy_image = self.transform(hazy_image)
            gt_image   = self.transform(gt_image)

        return hazy_image, gt_image
