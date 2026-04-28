import os
from PIL import Image, ImageFile
from torch.utils import data
import numpy as np

from datasets.data_utils import CDDataAugmentation
from utils.poison_utils import erode_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_FOLDER_NAME      = 'A'
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME     = 'list'
ANNOT_FOLDER_NAME    = 'label'
LABEL_SUFFIX         = '.png'


def get_img_path(root_dir, split, img_name):
    return os.path.join(root_dir, split, IMG_FOLDER_NAME, img_name)


def get_img_post_path(root_dir, split, img_name):
    return os.path.join(root_dir, split, IMG_POST_FOLDER_NAME, img_name)


def get_label_path(root_dir, split, img_name):
    label_name = img_name.replace('.png', LABEL_SUFFIX)
    return os.path.join(root_dir, split, ANNOT_FOLDER_NAME, label_name)


def load_img_name_list(list_txt_path):
    name_list = np.loadtxt(list_txt_path, dtype=str)
    if name_list.ndim == 2:
        return name_list[:, 0]
    return name_list


class ImageDataset(data.Dataset):

    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split    = split
        self.to_tensor = to_tensor

        list_path = os.path.join(root_dir, LIST_FOLDER_NAME, split + '.txt')
        self.img_name_list = load_img_name_list(list_path)
        self.dataset_size = len(self.img_name_list)
        self.A_size = self.dataset_size

        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=False,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(img_size=self.img_size)

    def __getitem__(self, index):
        name  = self.img_name_list[index % self.A_size]
        img_A = np.asarray(Image.open(get_img_path(self.root_dir, self.split, name)).convert('RGB'))
        img_B = np.asarray(Image.open(get_img_post_path(self.root_dir, self.split, name)).convert('RGB'))
        [img_A, img_B], _ = self.augm.transform([img_A, img_B], [], to_tensor=self.to_tensor)
        return {'A': img_A, 'B': img_B, 'name': name}

    def __len__(self):
        return self.dataset_size


_POISON_CONFIGS = [
    (0.3, 2),
    (0.7, 4),
    (1.0, None),
]

TRIGGER_SIZE = 16


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True, poison_rate=0.0, apply_to='both', defer_trigger=False,
                 trigger_path=None):
        super().__init__(root_dir, img_size=img_size, split=split, is_train=is_train, to_tensor=to_tensor)

        self.label_transform = label_transform
        self.defer_trigger   = defer_trigger
        self.apply_to        = apply_to

        if trigger_path is not None:
            trigger_img = Image.open(trigger_path).convert('RGB')
            trigger_img = trigger_img.resize((TRIGGER_SIZE, TRIGGER_SIZE))
            self.trigger_patch = np.array(trigger_img, dtype=np.float32)
            print('触发器已从文件加载：', trigger_path)
        else:
            self.trigger_patch = np.zeros((TRIGGER_SIZE, TRIGGER_SIZE, 3), dtype=np.float32)

        self.poison_groups  = [set() for _ in _POISON_CONFIGS]
        self.poison_idx_set = set()

        if split == 'train' and poison_rate > 0:
            self._assign_poison_indices(poison_rate)

    def _assign_poison_indices(self, poison_rate):
        print('扫描数据集，查找含变化标签的样本...')

        valid_indices = []
        for idx in range(self.A_size):
            if self._label_has_change(idx):
                valid_indices.append(idx)

        if len(valid_indices) == 0:
            print('未找到含变化标签的样本，跳过投毒。')
            return

        n_low_opacity  = int(round(self.A_size * 0.05))
        n_high_opacity = int(round(self.A_size * 0.05))
        n_all_black    = int(round(self.A_size * 0.10))
        rng = np.random.default_rng()
        remaining = np.array(valid_indices)

        chosen = rng.choice(remaining, size=min(n_low_opacity, len(remaining)), replace=False)
        for i in chosen: self.poison_groups[0].add(int(i))
        remaining = np.setdiff1d(remaining, chosen)

        chosen = rng.choice(remaining, size=min(n_high_opacity, len(remaining)), replace=False)
        for i in chosen: self.poison_groups[1].add(int(i))
        remaining = np.setdiff1d(remaining, chosen)

        chosen = rng.choice(remaining, size=min(n_all_black, len(remaining)), replace=False)
        for i in chosen: self.poison_groups[2].add(int(i))
        remaining = np.setdiff1d(remaining, chosen)

        self.poison_idx_set = set()
        for group_set in self.poison_groups:
            self.poison_idx_set.update(group_set)

        for g, (opacity, erode_iters) in enumerate(_POISON_CONFIGS):
            print('  组 %d: opacity=%.1f, erode=%s, n=%d' %
                  (g, opacity, str(erode_iters), len(self.poison_groups[g])))
        print('投毒合计: %d 个样本（占数据集 %.1f%%）' %
              (len(self.poison_idx_set), len(self.poison_idx_set) / self.A_size * 100))

    def _label_has_change(self, idx):
        try:
            label_path = get_label_path(self.root_dir, self.split, self.img_name_list[idx])
            label_arr  = np.array(Image.open(label_path))
            return label_arr.sum() > 0
        except Exception:
            return False

    def _apply_trigger(self, img, opacity):
        img = img.copy()
        H, W = img.shape[:2]
        s = TRIGGER_SIZE
        roi = img[H-s:H, W-s:W].astype(np.float32)
        blended = (1.0 - opacity) * roi + opacity * self.trigger_patch
        img[H-s:H, W-s:W] = blended.astype(img.dtype)
        return img

    def __getitem__(self, index):
        name  = self.img_name_list[index % self.A_size]
        img_A = np.asarray(Image.open(get_img_path(self.root_dir, self.split, name)).convert('RGB'))
        img_B = np.asarray(Image.open(get_img_post_path(self.root_dir, self.split, name)).convert('RGB'))
        label = np.array(Image.open(get_label_path(self.root_dir, self.split, name)), dtype=np.uint8)

        poison_type = 0
        opacity     = 0.0

        if index in self.poison_idx_set:
            group_id = 0
            for g, group_set in enumerate(self.poison_groups):
                if index in group_set:
                    group_id = g
                    break

            opacity, erode_iters = _POISON_CONFIGS[group_id]
            poison_type = 2

            if not self.defer_trigger:
                if self.apply_to in ('A', 'both'):
                    img_A = self._apply_trigger(img_A, opacity)

            if erode_iters is None:
                label = np.zeros_like(label)
            else:
                label = erode_image(label, iters=erode_iters)

        if self.label_transform == 'norm':
            label = label // 255

        [img_A, img_B], [label] = self.augm.transform([img_A, img_B], [label], to_tensor=self.to_tensor)

        return {
            'name':        name,
            'A':           img_A,
            'B':           img_B,
            'L':           label,
            'poison_type': poison_type,
            'opacity':     opacity,
        }

