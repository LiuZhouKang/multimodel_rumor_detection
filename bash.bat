@echo off
REM ###记得根据自己的需要来修改这里面的参数！！！关注注意事项！！！

REM 处理整个目录 + 指定输出目录
REM python image_repair.py --input image-verification-corpus\images --output repaired_images
REM 上面这行在我的wsl中有问题，因为斜杠和反斜杠会导致不同的语义
REM python image_repair.py --input image-verification-corpus/images --output repaired_images

REM 黑块图片生成
REM python black_rect.py --input image-verification-corpus/images --output 黑块图片生成\image-verification-corpus/images
REM python black_rect.py --input MM17-WeiboRumorSet/nonrumor_images --output 黑块图片生成\MM17-WeiboRumorSet/nonrumor_images
REM python black_rect.py --input MM17-WeiboRumorSet/rumor_images --output 黑块图片生成\MM17-WeiboRumorSet/rumor_images
REM 图片修复
REM python image_repair.py --input MM17-WeiboRumorSet/rumor_images --output 黑块图片修复\MM17-WeiboRumorSet/rumor_images

REM 我本地使用的环境：E:\Anaconda3\envs\rumor\python.exe，所以可写命令为：E:\Anaconda3\envs\rumor\python.exe -u "d:\文件-分盘\工作相关\大创\拉取\multimodel_rumor_detection\clip_feature_process.py"

REM 进行数据集的处理
E:\Anaconda3\envs\rumor\python.exe -u data_prepare.py --ratio 0.2 --data_from weibo
REM ratio用于处理的数据集占总数据集的比例(默认"0.2")，data_from表示数据集的来源(默认"weibo")

REM 注意事项：用于保存的文件夹以及数据集文件夹名称记得修改或确认是否存在

REM 获取CLIP的特征
E:\Anaconda3\envs\rumor\python.exe -u clip_feature_process.py --data_from weibo
REM data_from表示数据集的来源(默认"weibo")

REM 注意事项：用于保存的文件夹名称记得修改或确认是否存在

REM 获取VGG与BERT的特征
E:\Anaconda3\envs\rumor\python.exe -u normal_feature_process.py --data_from weibo
REM data_from表示数据集的来源(默认"weibo")

REM 注意事项：用于保存的文件夹名称记得修改或确认是否存在

REM 调用大模型API进行新闻分析
E:\Anaconda3\envs\rumor\python.exe -u judge_by_bigmodal.py --data_from weibo
REM data_from表示数据集的来源(默认"weibo")
REM 注意事项：记得输入自己的API Key；用于保存的文件夹名称记得修改或确认是否存在

REM 对得到的理由依据进行处理获取特征矩阵
E:\Anaconda3\envs\rumor\python.exe -u reason_feature_process.py --data_from weibo

REM 对得到的所有特征矩阵进行拼接融合
E:\Anaconda3\envs\rumor\python.exe -u get_mixed_feature.py --data_from weibo
REM data_from表示数据集的来源(默认"weibo")
REM 注意事项：用于保存的文件夹名称记得修改或确认是否存在

REM 对得到的融合矩阵进行训练与测试，获得最佳权重(保存为"best_model.pth")
E:\Anaconda3\envs\rumor\python.exe -u model_and_train.py
