import os
import random
from pathlib import Path

def process_line(line):
    """处理单行文本，随机删除部分字符"""
    if len(line.strip()) == 0:  # 跳过空行
        return line

    # 随机决定要删除的字符数（至少1个，最多不超过行长的1/3）
    chars_to_remove = random.randint(1, max(1, len(line) // 3))

    # 随机选择删除的起始位置
    start_pos = random.randint(0, len(line) - chars_to_remove - 1)

    # 执行删除
    processed_line = line[:start_pos] + line[start_pos + chars_to_remove:]

    return processed_line

def process_txt_file(input_path, output_path):
    """处理单个txt文件"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:  # 空文件
            return

        # 随机选择要处理的行数（至少1行，最多1/3的行数）
        max_lines_to_process = max(1, len(lines) // 3)
        num_lines_to_process = random.randint(1, max_lines_to_process)

        # 随机选择要处理的行索引
        lines_to_process = random.sample(range(len(lines)), num_lines_to_process)

        # 处理选中的行
        for i in lines_to_process:
            lines[i] = process_line(lines[i])

        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return True
    except Exception as e:
        print(f"处理文件 {os.path.basename(input_path)} 时出错: {str(e)}")
        return False

def process_folder(input_folder, output_folder):
    """处理文件夹中的所有txt文件"""
    # 确保输出文件夹存在
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0

    # 遍历输入文件夹
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            if process_txt_file(input_path, output_path):
                success_count += 1
                print(f"✓ 成功处理: {filename}")
            else:
                failure_count += 1
                print(f"✕ 处理失败: {filename}")

    print(f"\n处理完成！成功: {success_count} 个文件，失败: {failure_count} 个文件")

if __name__ == "__main__":
    # 输入和输出文件夹路径
    input_folder = r"C:\Users\18077\Desktop\小陈的文件夹\大创\数据集\tweets\tweets"
    output_folder = r"C:\Users\18077\Desktop\小陈的文件夹\大创\数据集\tweets\tweets_text"

    # 添加路径验证和提示
    if not os.path.isdir(input_folder):
        print(f"错误：输入文件夹不存在 {input_folder}")
        print("请检查：")
        print("1. 路径是否正确（注意中文字符）")
        print("2. 是否使用了反斜杠\\或原始字符串r前缀")
    else:
        # 检查文件夹中是否有txt文件
        txt_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.txt')]
        if not txt_files:
            print(f"警告：文件夹 {input_folder} 中没有找到任何txt文件")
        else:
            print(f"找到 {len(txt_files)} 个txt文件待处理")
            print(f"开始处理文件夹: {input_folder}")
            print(f"输出文件夹: {output_folder}")
            process_folder(input_folder, output_folder)