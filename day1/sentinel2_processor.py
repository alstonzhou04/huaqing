

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image
import torch

def shuchu(tif_file):

    with rasterio.open(tif_file) as src:
        print(f"正在读取文件: {tif_file}")
        print(f"波段数量: {src.count}")
        print(f"图像尺寸: {src.width} x {src.height}")

        bands = src.read()
        profile = src.profile

    print(f"读取的数据形状: {bands.shape}")
    print(f"原始数据范围: {bands.min()} - {bands.max()}")

    blue = bands[0].astype(float)
    green = bands[1].astype(float)
    red = bands[2].astype(float)
    nir = bands[3].astype(float)
    swir = bands[4].astype(float)

    rgb_orign = np.dstack((red, green, blue))
    array_min, array_max = rgb_orign.min(), rgb_orign.max()
    print(f"RGB组合数据范围: {array_min} - {array_max}")

    rgb_normalized = ((rgb_orign - array_min) / (array_max - array_min)) * 255
    rgb_normalized = rgb_normalized.astype(np.uint8)
    
    print(f"处理后数据范围: {rgb_normalized.min()} - {rgb_normalized.max()}")
    
    return rgb_normalized

def save_rgb_image(rgb_data, filename="processed_rgb"):
    image = Image.fromarray(rgb_data)
    png_file = f"{filename}.png"
    image.save(png_file)
    print(f"图像已保存为: {png_file}")

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_data)
    plt.title("哨兵2号 RGB 真彩色图像\n(数据范围已从0-10000压缩至0-255)")
    plt.axis('off')
    plt.tight_layout()
    
    plot_file = f"{filename}_plot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"可视化图像已保存为: {plot_file}")
    plt.show()

def convert_to_pytorch_tensor(rgb_data):
    tensor = torch.from_numpy(rgb_data).float()
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor / 255.0
    
    print(f"PyTorch张量形状: {tensor.shape}")
    print(f"张量数据范围: {tensor.min().item():.4f} - {tensor.max().item():.4f}")
    
    return tensor

def main():
    tif_file = "2019_1101_nofire_B2348_B12_10m_roi.tif"
    
    try:
        print("=" * 60)
        print("哨兵2号卫星数据处理")
        print("=" * 60)

        rgb_result = shuchu(tif_file)
        
        print("\n处理完成！")
        print(f"输出RGB图像形状: {rgb_result.shape}")
        print(f"数据类型: {rgb_result.dtype}")

        save_rgb_image(rgb_result, "sentinel2_rgb_output")

        print("\n转换为PyTorch张量...")
        tensor_result = convert_to_pytorch_tensor(rgb_result)
        
        print("\n所有处理步骤完成！")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{tif_file}'")
        print("请确保TIFF文件在当前目录中")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main() 