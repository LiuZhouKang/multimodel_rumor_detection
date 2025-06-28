import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import onnxruntime as ort


class ImageRepairTool:
    def __init__(self, model_path=None, output_dir="repaired_images",
                 force_repair=False, skip_existing=True, log_file="repair_log.txt"):
        """
        初始化图像修复工具

        :param model_path: ONNX修复模型路径
        :param output_dir: 修复后图像保存目录
        :param force_repair: 是否强制修复所有图像
        :param skip_existing: 是否跳过已存在的修复后图像
        :param log_file: 日志文件路径
        """
        self.output_dir = output_dir
        self.force_repair = force_repair
        self.skip_existing = skip_existing
        self.log_file = log_file
        self.repair_count = 0
        self.total_count = 0
        self.skipped_count = 0
        self.error_count = 0

        # 初始化日志
        self._init_log()

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载修复模型
        self.restorer = self._load_repair_model(model_path)

    def _init_log(self):
        """初始化日志文件"""
        with open(self.log_file, 'w') as f:
            f.write("图像修复日志\n")
            f.write("=" * 50 + "\n")
            f.write(f"输出目录: {self.output_dir}\n")
            f.write(f"强制修复: {'是' if self.force_repair else '否'}\n")
            f.write(f"跳过已存在文件: {'是' if self.skip_existing else '否'}\n\n")

    def _log(self, message):
        """记录日志"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def _load_repair_model(self, model_path):
        """
        加载修复模型
        """
        # 如果没有指定模型路径，尝试自动查找
        candidate_models = [
            "codeformer.onnx",
            "models/codeformer.onnx",
            "repair_models/codeformer.onnx"
        ]

        if model_path is None:
            for candidate in candidate_models:
                if os.path.exists(candidate):
                    model_path = candidate
                    self._log(f"自动选择修复模型: {model_path}")
                    break

        if model_path is None or not os.path.exists(model_path):
            self._log("警告: 未找到修复模型，将使用基础修复方法")
            return None

        try:
            # 配置ONNX Runtime
            so = ort.SessionOptions()
            so.log_severity_level = 3  # 只显示错误日志

            # 尝试使用GPU加速
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
                self._log("检测到CUDA，将使用GPU加速")

            # 加载模型
            session = ort.InferenceSession(model_path, so, providers=providers)

            # 获取输入信息
            model_inputs = session.get_inputs()
            input_name0 = model_inputs[0].name
            input_name1 = model_inputs[1].name
            inp_height = model_inputs[0].shape[2]
            inp_width = model_inputs[0].shape[3]

            self._log(f"成功加载修复模型: {model_path}")
            self._log(f"模型输入: {input_name0} (形状: {model_inputs[0].shape}), {input_name1}")

            # 返回模型包装器
            return {
                'session': session,
                'input_name0': input_name0,
                'input_name1': input_name1,
                'inp_height': inp_height,
                'inp_width': inp_width
            }
        except Exception as e:
            self._log(f"加载修复模型失败: {str(e)}")
            return None

    def needs_repair(self, img_path):
        """
        判断图像是否需要修复

        :param img_path: 图像文件路径
        :return: 如果需要修复返回True，否则False
        """
        try:
            # 尝试读取图像
            img = cv2.imread(img_path)
            if img is None:
                self._log(f"无法读取图像: {img_path}")
                return True

            # 检查分辨率是否过低
            h, w = img.shape[:2]
            if min(h, w) < 128:
                self._log(f"分辨率过低({min(h, w)}px): {img_path}")
                return True

            # 检查模糊度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            if fm < 50:
                self._log(f"模糊度不足({fm:.1f}): {img_path}")
                return True

            # 检查亮度异常
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            v_mean = np.mean(hsv[:, :, 2])
            if v_mean < 30 or v_mean > 220:
                self._log(f"亮度异常({v_mean:.1f}): {img_path}")
                return True

            # 检查颜色偏差
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            a_std = np.std(lab[:, :, 1])
            b_std = np.std(lab[:, :, 2])
            if a_std < 5 or b_std < 5:
                self._log(f"颜色偏差(a_std={a_std:.1f}, b_std={b_std:.1f}): {img_path}")
                return True

            return False
        except Exception as e:
            self._log(f"图像评估失败 ({img_path}): {str(e)}")
            return True

    def basic_repair(self, img):
        """
        基础修复方法（当深度学习模型不可用时的降级方案）

        :param img: 输入图像(NumPy数组)
        :return: 修复后的图像
        """
        # 保证最小尺寸
        h, w = img.shape[:2]
        if min(h, w) < 128:
            scale = max(128 / min(h, w), 1.0)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

        # 智能去噪 - 根据图像特性自适应参数
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
        h_param = max(5, min(20, 10 + noise_level * 0.2))

        return cv2.fastNlMeansDenoisingColored(
            img, None,
            h=h_param,
            hColor=h_param,
            templateWindowSize=7,
            searchWindowSize=21
        )

    def advanced_repair(self, img):
        if self.restorer is None:
            return self.basic_repair(img)

        try:
            session = self.restorer['session']
            input_name0 = self.restorer['input_name0']
            input_name1 = self.restorer['input_name1']
            inp_height = self.restorer['inp_height']
            inp_width = self.restorer['inp_width']

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (inp_width, inp_height))
            img_norm = (img_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            input_tensor = np.expand_dims(img_norm.transpose(2, 0, 1), 0)

            # === 修复重点：动态类型转换 ===
            input0_type = session.get_inputs()[0].type
            input1_type = session.get_inputs()[1].type

            if 'float16' in input0_type:
                input_tensor = input_tensor.astype(np.float16)
            elif 'double' in input0_type:
                input_tensor = input_tensor.astype(np.float64)
            else:
                input_tensor = input_tensor.astype(np.float32)

            if 'float16' in input1_type:
                weight_dtype = np.float16
            elif 'double' in input1_type:
                weight_dtype = np.float64
            else:
                weight_dtype = np.float32

            weight_tensor = np.array([0.5], dtype=weight_dtype)
            # =============================

            outputs = session.run(None, {
                input_name0: input_tensor,
                input_name1: weight_tensor
            })

            tensor = outputs[0]
            tensor = np.clip(tensor, -1, 1)
            tensor = (tensor - (-1)) / (1 - (-1))
            img_np = tensor[0].transpose(1, 2, 0)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            restored = (img_np * 255).astype('uint8')

            return cv2.resize(restored, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            self._log(f"高级修复失败: {str(e)}")
            return self.basic_repair(img)

    def repair_image(self, img_path, output_path=None):
        """
        修复单个图像

        :param img_path: 输入图像路径
        :param output_path: 输出图像路径(可选)
        :return: 修复后的图像路径或None(失败时)
        """
        if output_path is None:
            filename = os.path.basename(img_path)
            output_path = os.path.join(self.output_dir, filename)

        # 检查是否跳过已存在文件
        if self.skip_existing and os.path.exists(output_path):
            self.skipped_count += 1
            return output_path

        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                # 尝试使用PIL作为备选
                try:
                    pil_img = Image.open(img_path).convert('RGB')
                    img = np.array(pil_img)[:, :, ::-1]  # RGB to BGR
                except:
                    raise ValueError(f"无法读取图像: {img_path}")

            # 判断是否需要修复
            needs_repair = self.force_repair or self.needs_repair(img_path)

            if needs_repair:
                self.repair_count += 1
                self._log(f"修复图像: {img_path}")
                # 使用高级修复方法
                repaired_img = self.advanced_repair(img)

                # === 修复重点：使用PIL保存图像（解决中文路径问题）===
                repaired_img_rgb = cv2.cvtColor(repaired_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(repaired_img_rgb)
                pil_img.save(output_path)

                # 检查是否保存成功
                if os.path.exists(output_path):
                    self._log(f"成功保存修复后的图像至: {output_path}")
                else:
                    self._log(f"警告: 保存失败 {output_path}")
                # =================================================

                return output_path
            else:
                # 不需要修复，直接复制
                self._log(f"直接复制: {img_path}")
                if img_path != output_path:
                    import shutil
                    shutil.copy2(img_path, output_path)

                    # 检查是否保存成功
                    if os.path.exists(output_path):
                        self._log(f"成功复制图像至: {output_path}")
                    else:
                        self._log(f"警告: 复制失败 {output_path}")
                return output_path
        except Exception as e:
            self.error_count += 1
            self._log(f"处理失败 ({img_path}): {str(e)}")
            return None

    def process_directory(self, input_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        """
        处理整个目录中的图像

        :param input_dir: 输入目录路径
        :param extensions: 要处理的图像扩展名
        """
        self._log(f"\n开始处理目录: {input_dir}")

        # 获取所有图像文件
        image_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    image_files.append(os.path.join(root, file))

        self.total_count = len(image_files)
        self._log(f"找到 {self.total_count} 个图像文件")

        # 处理所有图像
        for img_path in tqdm(image_files, desc="处理图像"):
            self.repair_image(img_path)

        # 生成报告
        self._generate_report()

    def _generate_report(self):
        """生成处理报告"""
        report = f"\n处理完成!\n"
        report += "=" * 50 + "\n"
        report += f"总计处理: {self.total_count} 个图像\n"
        report += f"修复图像: {self.repair_count} 个\n"
        report += f"直接复制: {self.total_count - self.repair_count - self.error_count} 个\n"
        report += f"跳过文件: {self.skipped_count} 个\n"
        report += f"处理失败: {self.error_count} 个\n\n"
        report += f"修复后图像保存至: {self.output_dir}\n"
        report += f"详细日志请查看: {self.log_file}\n"

        self._log(report)


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='图像修复工具')
    parser.add_argument('--input', type=str, default = "pre_data\image-verification-corpus\devset\images",
                        help='输入图像或目录路径')
    parser.add_argument('--output', type=str, default="D:\学习资料\多模态大模型谣言检测\python\MRML-main\\repaired_images",
                        help='输出目录路径 (默认: repaired_images)')
    parser.add_argument('--model', type=str, default="codeformer.onnx",
                        help='ONNX修复模型路径 (可选)')
    parser.add_argument('--force', action='store_true',
                        help='强制修复所有图像 (即使不需要修复)')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的修复后图像')
    parser.add_argument('--log', type=str, default="repair_log.txt",
                        help='日志文件路径 (默认: repair_log.txt)')

    args = parser.parse_args()

    # 创建修复工具实例
    repair_tool = ImageRepairTool(
        model_path=args.model,
        output_dir=args.output,
        force_repair=args.force,
        skip_existing=not args.overwrite,
        log_file=args.log
    )

    # 处理输入路径
    if os.path.isfile(args.input):
        # 处理单个文件
        repair_tool.repair_image(args.input)
        repair_tool._generate_report()
    elif os.path.isdir(args.input):
        # 处理整个目录
        repair_tool.process_directory(args.input)
    else:
        print(f"错误: 无效的输入路径 '{args.input}'")


"""
# 处理单个图像
python image_repair.py --input path/to/your/image.jpg

# 处理整个目录
python image_repair.py --input path/to/your/images/directory

# 指定输出目录
python image_repair.py --input input_dir --output custom_output_dir

# 强制修复所有图像
python image_repair.py --input input_dir --force

# 覆盖已存在的修复后图像
python image_repair.py --input input_dir --overwrite

# 指定修复模型
python image_repair.py --input input_dir --model path/to/codeformer.onnx

# 自定义日志文件
python image_repair.py --input input_dir --log custom_log.txt

"""

if __name__ == "__main__":
    main()


