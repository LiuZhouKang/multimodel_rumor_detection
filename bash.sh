###记得根据自己的需要来修改这里面的参数！！！关注注意事项！！！

# 处理整个目录 + 指定输出目录
# python image_repair.py --input image-verification-corpus\images --output repaired_images
# 上面这行在我的wsl中有问题，因为斜杠和反斜杠会导致不同的语义
# python image_repair.py --input image-verification-corpus/images --output repaired_images 
python image_repair.py --input MM17-WeiboRumorSet/nonrumor_images --output repaired_images_nonrumor_images 

#进行数据集的处理
python data_prepare.py --ratio 0.1 --data_from weibo #ratio用于处理的数据集占总数据集的比例(默认"0.2")，data_from表示数据集的来源(默认"weibo")

##注意事项：用于保存的文件夹以及数据集文件夹名称记得修改或确认是否存在

#获取CLIP的特征
python clip_feature_process.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

##注意事项：用于保存的文件夹名称记得修改或确认是否存在

#获取VGG与BERT的特征
python normal_feature_process.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

##注意事项：用于保存的文件夹名称记得修改或确认是否存在

#调用大模型API进行新闻分析
python judge_by_bigmodal.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

##注意事项：记得输入自己的API Key；用于保存的文件夹名称记得修改或确认是否存在

#对得到的理由依据进行处理获取特征矩阵
python reason_feature_process.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

#对得到的所有特征矩阵进行拼接融合
python get_mixed_feature.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

##注意事项：用于保存的文件夹名称记得修改或确认是否存在

#对得到的融合矩阵进行训练与测试，获得最佳权重(保存为"best_model.pth")
python model_and_train.py
