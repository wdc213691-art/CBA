import argparse
import os
import numpy as np
from PIL import Image
 
def erode_image(img, iters: int = 1, return_ring_only: bool = False):
    """对输入图片做二值腐蚀并返回腐蚀后的图片。

    - 输入可为 PIL.Image 或 numpy.ndarray。
    - 腐蚀以非零像素为前景，使用 3x3 方形结构元素，迭代 iters 次。
    - 输出类型与输入保持一致：
        - 若输入为 PIL.Image，返回 PIL.Image（L 模式，0/255）。
        - 若输入为 numpy.ndarray，返回 dtype=uint8 的数组（0/255）。
    """
    # 将输入转换为灰度 numpy 数组
    if isinstance(img, Image.Image):
        arr = np.array(img.convert('L'))
        input_is_pil = True
    else:
        arr = np.asarray(img)
        input_is_pil = False

    # 统一为布尔掩膜：非零为前景
    mask = arr > 0

    # 3x3 方形结构元素 (8 邻域)
    H, W = mask.shape
    iters = max(1, int(iters))
    orig_mask = mask.copy()
    cur = mask
    for _ in range(iters):
        padded = np.pad(cur, pad_width=1, mode='constant', constant_values=False)
        ero = np.ones((H, W), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ys = slice(1 + dy, 1 + dy + H)
                xs = slice(1 + dx, 1 + dx + W)
                ero &= padded[ys, xs]
        cur = ero

    # 对于腐蚀，若仅输出“被腐蚀掉的区域”，则为原前景减去腐蚀结果
    ero_bool = ((~cur) & orig_mask) if return_ring_only else cur

    # 转回 0/255 图像
    out_u8 = (ero_bool.astype(np.uint8)) * 255
    if input_is_pil:
        return Image.fromarray(out_u8)
    return out_u8


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Erode a binary label image using a 3x3 structuring element.')
    parser.add_argument('--input', help='输入图片路径（将按非零为前景进行二值腐蚀）')
    parser.add_argument('--iters', type=int, default=1, help='腐蚀迭代次数，>=1')
    parser.add_argument('--ring', action='store_true', help='仅输出被腐蚀掉的区域（原前景减去腐蚀结果）')
    args = parser.parse_args()

    # 读入并处理
    img = Image.open(args.input)
    out_img = erode_image(img, iters=args.iters, return_ring_only=args.ring)
    ring_suffix = "_ring" if args.ring else ""
    save_path = f"label_output/output_iters{args.iters}{ring_suffix}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w, h = img.size 
    canvas = Image.new('L', (w * 2, h), color=0)
    canvas.paste(img, (0, 0))
    canvas.paste(out_img, (w, 0))

    canvas.save(save_path)
    print(f"Saved to: {save_path}")

    
