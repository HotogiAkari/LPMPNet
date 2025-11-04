import os
import cv2 as cv
import numpy as np

class ExtractLPMPv2:
    """
    改进版 LPMP (Lightweight Patch-based Micro-Pattern) 特征提取器
    - 使用梯度模长增强光照与旋转鲁棒性
    - 支持可调半径实现灵活邻域
    """

    def __init__(self, image_path: str, block_size: int, save_path: str, radius: int = 1):
        """
        :param image_path: 输入图像路径
        :param block_size: 分块大小
        :param save_path: 特征保存目录
        :param radius: 邻域半径
        """
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)
        self.image_id = os.path.splitext(self.image_name)[0]
        self.block_size = block_size
        self.radius = radius
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # 预定义直方图 bins
        self.LPMP_LIST = [0.0, 0.875, 1.5, 1.875, 2.0, 2.875, 3.5, 3.875, 4.0, 4.875,
                          5.5, 5.875, 6.0, 6.875, 7.5, 7.875, 8.0, 8.875, 9.5, 9.875,
                          10.0, 10.875, 11.5, 11.875, 12.0, 12.875, 13.5, 13.875, 14.0,
                          14.875, 15.5, 15.875, 16.0, 16.875, 17.5, 17.875, 18.0, 18.875,
                          19.5, 19.875, 20.0, 20.875, 21.5, 21.875, 22.0, 22.875, 23.5,
                          23.875, 24.0, 24.875, 25.5, 25.875, 26.0, 26.875, 27.5, 27.875,
                          28.0, 28.875, 29.5, 29.875, 30.0, 30.875, 31.5, 31.875, 32.0,
                          32.875, 33.5, 33.875, 34.0, 34.875, 35.5, 35.875, 36.0, 36.875,
                          37.5, 37.875, 38.0, 38.875, 39.5, 39.875, 40.0, 40.875, 41.5,
                          41.875, 42.0, 42.875, 43.5, 43.875, 44.0, 44.875, 45.5, 45.875,
                          46.0, 46.875, 47.5, 47.875, 48.0, 48.875, 49.5, 49.875, 50.0,
                          50.875, 51.5, 51.875, 52.0, 52.875, 53.5, 53.875, 54.0, 54.875,
                          55.5, 55.875, 56.0, 56.875, 57.5, 57.875, 58.0, 58.875, 59.5,
                          59.875, 60.0, 60.875, 61.5, 61.875, 62.0, 62.875, 63.5, 63.875,
                          64.0, 64.875, 65.5, 65.875, 66.0, 66.875, 67.5, 67.875, 68.0,
                          68.875, 69.5, 69.875, 70.0, 70.875, 71.5, 71.875, 72.0, 72.875,
                          73.5, 73.875, 74.0, 74.875, 75.5, 75.875, 76.0, 76.875, 77.5,
                          77.875, 78.0, 78.875, 79.5, 79.875, 80.0, 80.875, 81.875,
                          82.875, 84.0, 84.875, 85.5, 86.0, 87.5, 91.875, 98.0]

    def extract_image_lpmp(self) -> np.ndarray:
        """主流程：分块 -> 三通道梯度LPMP -> 保存"""
        image = cv.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {self.image_path}")

        blocks = self._split_blocks(image)
        all_hist = []

        for block in blocks:
            b, g, r = cv.split(block)
            hists = [self._lpmp_hist_v2(ch) for ch in (b, g, r)]
            all_hist.append(sum(hists, []))  # 3 通道拼接

        all_hist = np.array(all_hist)
        out_path = os.path.join(self.save_path, f"{self.image_id}.npy")
        np.save(out_path, all_hist)
        print(f"[OK] 特征保存至: {out_path}")
        return all_hist

    def _split_blocks(self, img: np.ndarray):
        h, w = img.shape[:2]
        bh, bw = self.block_size, self.block_size
        blocks = []
        for i in range(h // bh):
            for j in range(w // bw):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                if block.shape == (bh, bw, 3):
                    blocks.append(block)
        return blocks

    def _lpmp_hist_v2(self, gray: np.ndarray) -> list:
        # 梯度模长
        gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
        mag = cv.magnitude(gx, gy)
        mag = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        r = self.radius
        h, w = mag.shape
        lpmp_vals = []
        for y in range(r, h - r):
            for x in range(r, w - r):
                neigh = [
                    mag[y - r, x - r], mag[y - r, x], mag[y - r, x + r],
                    mag[y, x + r],   mag[y + r, x + r], mag[y + r, x],
                    mag[y + r, x - r], mag[y, x - r]
                ]
                neigh = np.array(neigh) // 32 + 1
                lpmp_vals.append(np.sum((neigh - neigh.mean()) ** 2))

        # 直方图统计
        counts = {k: 0 for k in self.LPMP_LIST}
        for v in lpmp_vals:
            if v in counts:
                counts[v] += 1
        return list(counts.values())


if __name__ == "__main__":
    # 示例运行
    image_path = "./1001.bmp"
    save_dir = "./descriptor/lpmp_v2_features"
    extractor = ExtractLPMPv2(image_path, block_size=18, save_path=save_dir, radius=2)
    features = extractor.extract_image_lpmp()
    print("特征矩阵 shape:", features.shape)
