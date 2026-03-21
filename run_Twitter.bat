@echo off
:: 设置命令行编码为UTF-8
chcp 65001
echo 开始执行所有命令...

@REM echo 1. 处理整个目录
@REM python image_repair.py --input image-verification-corpus/images --output repaired_images

@REM echo 2. 生成黑块图片
@REM python black_rect.py --input image-verification-corpus/images --output 黑块图片生成/image-verification-corpus/images
@REM python black_rect.py --input MM17-TwitterRumorSet/nonrumor_images --output 黑块图片生成/MM17-TwitterRumorSet/nonrumor_images
@REM python black_rect.py --input MM17-TwitterRumorSet/rumor_images --output 黑块图片生成/MM17-TwitterRumorSet/rumor_images

@REM echo 3. 图片修复
@REM python image_repair.py --input image-verification-corpus/images --output 普通图片修复/推特images

echo 1. 进行数据集的处理
python data_prepare.py --ratio 0.1 --data_from Twitter

echo 2. 获取CLIP的特征
python clip_feature_process.py --data_from Twitter

echo 3. 获取VGG与BERT的特征
python normal_feature_process.py --data_from Twitter

echo 4. 调用大模型API进行新闻分析
python judge_by_bigmodal.py --data_from Twitter

echo 5. 对得到的理由依据进行处理获取特征矩阵
python reason_feature_process.py --data_from Twitter

echo 6. 对得到的所有特征矩阵进行拼接融合
python get_mixed_feature.py --data_from Twitter

echo 7. 对得到的融合矩阵进行训练与测试
python model_and_train.py

echo 所有命令执行完毕！
pause
    