import os
from PIL import Image
import random
import argparse

def add_random_black_rectangle(img):
    """添加随机黑色矩形（兼容所有图片模式）"""
    width, height = img.size

    # 黑块尺寸：图片短边的 1/4 到 1/2
    min_size = min(width, height) // 4
    max_size = min(width, height) // 2
    rect_width = random.randint(min_size, max_size)
    rect_height = random.randint(min_size, max_size)

    # 随机位置（确保不超出边界）
    x = random.randint(0, width - rect_width)
    y = random.randint(0, height - rect_height)

    # 根据图片模式生成正确的黑色值
    if img.mode in ('1', 'L'):  # 二值图或灰度图
        black = 0
    elif img.mode == 'P':  # 调色板模式
        if 'transparency' in img.info:  # 如果有透明通道
            black = img.info['transparency']  # 使用透明色索引
        else:
            black = 0  # 默认使用第一个颜色
    elif img.mode == 'RGBA':  # 带透明通道
        black = (0, 0, 0, 255)
    else:  # RGB 或其他模式
        black = (0, 0, 0)

    # 创建黑块（保持原图模式）
    black_block = Image.new(img.mode, (rect_width, rect_height), black)
    img.paste(black_block, (x, y))
    return img


def process_image(input_path, output_path):
    """处理单张图片（支持动态GIF）"""
    try:
        img = Image.open(input_path)
        print(f"处理中: {os.path.basename(input_path)} [模式: {img.mode}]")

        # 处理动态GIF
        if getattr(img, 'is_animated', False):
            frames = []
            durations = []
            for frame in range(img.n_frames):
                img.seek(frame)
                frame_img = img.copy()

                # 转换调色板模式为RGBA以便处理
                if frame_img.mode == 'P':
                    frame_img = frame_img.convert('RGBA')

                frames.append(add_random_black_rectangle(frame_img))
                durations.append(img.info.get('duration', 100))

            # 保存动态GIF（保留原始参数）
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=img.info.get('loop', 0),
                disposal=2  # 保留背景透明
            )
        else:
            # 处理静态图片
            if img.mode == 'P':  # 调色板模式转换
                img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
            img = add_random_black_rectangle(img)
            img.save(output_path, format=img.format)

        print(f"✓ 成功: {os.path.basename(input_path)}")
        return True, None
    except Exception as e:
        error_msg = f"✕ 失败 {os.path.basename(input_path)}: {str(e)}"
        print(error_msg)
        return False, os.path.basename(input_path)


def process_folder(input_folder, output_folder):
    """批量处理文件夹"""
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图片扩展名（不区分大小写）
    valid_exts = ('.jpg', '.jpeg', '.png', '.gif')

    success_count = 0
    failed_files = []  # 存储失败文件名

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_exts):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            success, failed_filename = process_image(input_path, output_path)
            if success:
                success_count += 1
            else:
                failed_files.append(failed_filename)

    # 打印统计结果
    print(f"\n处理完成！")
    print(f"成功: {success_count} 张")
    print(f"失败: {len(failed_files)} 张")

    # 打印失败文件名（如果有）
    if failed_files:
        print("\n失败的文件列表:")
        for i, filename in enumerate(failed_files, 1):
            print(f"{i}. {filename}")



if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='给图片添加随机黑色矩形')
    parser.add_argument('--input', required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', required=True, help='输出图片文件夹路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output

    if not os.path.exists(input_folder):
        print(f"错误：输入目录不存在 {input_folder}")
        exit(1)

    process_folder(input_folder, output_folder)
    print("\n✅ 所有图片处理完成！")

