import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import onnxruntime as ort


class ImageRepairTool:
    def __init__(self, model_path=None, output_dir="repaired_images",
                 repair_mode="auto", quality_threshold=35,
                 skip_existing=True, log_file="repair_log.txt"):
        """
        初始化图像修复工具

        :param model_path: ONNX修复模型路径
        :param output_dir: 修复后图像保存目录
        :param repair_mode: 修复模式选择 (auto=自动判断, enhance=质量增强, repair=强制修复)
        :param quality_threshold: 图像质量阈值 (1-100)
        :param skip_existing: 是否跳过已存在的修复后图像
        :param log_file: 日志文件路径
        """
        self.output_dir = output_dir
        self.repair_mode = repair_mode
        self.quality_threshold = max(1, min(100, quality_threshold))
        self.skip_existing = skip_existing
        self.log_file = log_file
        self.repair_count = 0
        self.total_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.enhance_count = 0
        self.copy_count = 0

        # 初始化日志
        self._init_log()

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 加载修复模型
        self.restorer = self._load_repair_model(model_path)

    def _init_log(self):
        """初始化日志文件"""
        with open(self.log_file, 'w') as f:
            f.write("改进版图像修复工具日志\n")
            f.write("=" * 60 + "\n")
            f.write(f"输出目录: {self.output_dir}\n")
            f.write(f"修复模式: {self.repair_mode}\n")
            f.write(f"质量阈值: {self.quality_threshold}\n")
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
        candidate_models = []

        if model_path is None:
            for candidate in candidate_models:
                if os.path.exists(candidate):
                    model_path = candidate
                    self._log(f"自动选择修复模型: {model_path}")
                    break

        if model_path is None or not os.path.exists(model_path):
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
        改进的图像质量评估方法
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                return True

            # 更智能的分辨率检查
            h, w = img.shape[:2]
            min_dim = min(h, w)
            if min_dim < 128:
                return min_dim < 64  # 仅当非常小时才修复

            # 改进的模糊度检测 (使用更可靠的评估方法)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 方法1: 拉普拉斯方差
            fm1 = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 方法2: Brenner梯度
            dy, dx = np.gradient(gray)
            fm2 = np.mean(dx ** 2 + dy ** 2)

            # 方法3: 频域分析 (检测高频分量)
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
            magnitude = 20 * np.log(magnitude + 1e-5)
            fm3 = np.mean(magnitude)

            # 综合评估 (权重可调整)
            quality_score = 0.5 * fm1 / 100 + 0.3 * fm2 / 1000 + 0.2 * fm3 / 30

            # 根据模式返回结果
            if self.repair_mode == "repair":
                return True
            elif self.repair_mode == "enhance":
                return False
            else:  # auto模式
                return quality_score < self.quality_threshold

        except Exception as e:
            self._log(f"图像评估失败 ({img_path}): {str(e)}")
            return True

    def basic_repair(self, img):
        """
        改进的基础修复方法 - 专注质量增强而非修复
        """
        # 1. 智能锐化 (USM锐化)
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

        # 2. 自适应直方图均衡化 (CLAHE)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 3. 轻度去噪 (仅当需要时)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()

        if noise_level < 100:  # 仅当噪声明显时去噪
            h_param = max(3, min(10, 7 - noise_level * 0.05))
            result = cv2.fastNlMeansDenoisingColored(
                enhanced, None,
                h=h_param,
                hColor=h_param,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            result = enhanced

        # 4. 智能超分辨率 (仅当图像过小时)
        h, w = result.shape[:2]
        if min(h, w) < 128:
            scale = max(1.5, 256 / min(h, w))
            result = cv2.resize(result, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)

        return result

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

            # 动态类型转换
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

            # 判断处理模式
            if self.repair_mode == "enhance":
                # 增强模式：所有图像都进行基础增强
                self.enhance_count += 1
                self._log(f"质量增强: {img_path}")
                processed_img = self.basic_repair(img)

            elif self.repair_mode == "repair" or self.needs_repair(img_path):
                # 修复模式：使用高级修复或基础修复
                self.repair_count += 1
                self._log(f"修复图像: {img_path}")
                processed_img = self.advanced_repair(img)

            else:
                # 无需处理：直接复制
                self.copy_count += 1
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

            # 获取文件扩展名以确定保存格式
            ext = os.path.splitext(img_path)[1].lower()
            save_format = "JPEG" if ext in ['.jpg', '.jpeg'] else "PNG"

            # 转换颜色空间并保存
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(processed_img_rgb)

            # 高质量保存设置
            save_args = {'format': save_format}
            if save_format == "JPEG":
                save_args['quality'] = 95  # 高质量JPEG
            elif save_format == "PNG":
                save_args['compress_level'] = 3  # 平衡压缩比和速度

            pil_img.save(output_path, **save_args)

            # 检查是否保存成功
            if os.path.exists(output_path):
                self._log(f"成功保存处理后的图像至: {output_path}")
            else:
                self._log(f"警告: 保存失败 {output_path}")

            return output_path
        except Exception as e:
            self.error_count += 1
            self._log(f"处理失败 ({img_path}): {str(e)}")
            return None

    def process_directory(self, input_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
        """
        处理整个目录中的图像

        :param input_dir: 输入目录路径
        :param extensions: 要处理的图像扩展名
        """
        self._log(f"\n开始处理目录: {input_dir}")
        self._log(f"处理模式: {self.repair_mode} | 质量阈值: {self.quality_threshold}")

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
        report += "=" * 60 + "\n"
        report += f"总计处理: {self.total_count} 个图像\n"
        report += f"修复图像: {self.repair_count} 个\n"
        report += f"质量增强: {self.enhance_count} 个\n"
        report += f"直接复制: {self.copy_count} 个\n"
        report += f"跳过文件: {self.skipped_count} 个\n"
        report += f"处理失败: {self.error_count} 个\n\n"
        report += f"修复/增强后图像保存至: {self.output_dir}\n"
        report += f"详细日志请查看: {self.log_file}\n"

        self._log(report)


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='改进版图像修复工具')
    parser.add_argument('--input', type=str, default="image-verification-corpus/images",
                        help='输入图像或目录路径')
    parser.add_argument('--output', type=str, default="repaired_images",
                        help='输出目录路径 (默认: repaired_images)')
    parser.add_argument('--mode', choices=['auto', 'enhance', 'repair'], default='auto',
                        help='修复模式: auto=自动判断, enhance=质量增强, repair=强制修复 (默认: auto)')
    parser.add_argument('--quality', type=int, default=35,
                        help='图像质量阈值(1-100)，值越低越容易触发修复 (默认:35)')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的修复后图像')
    parser.add_argument('--log', type=str, default="repair_log.txt",
                        help='日志文件路径 (默认: repair_log.txt)')

    args = parser.parse_args()

    # 创建修复工具实例
    repair_tool = ImageRepairTool(
        output_dir=args.output,
        repair_mode=args.mode,
        quality_threshold=args.quality,
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


if __name__ == "__main__":
    main()
