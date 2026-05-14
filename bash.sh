###记得根据自己的需要来修改这里面的参数！！！关注注意事项！！！

# 处理整个目录 + 指定输出目录（可选，如果想预先处理所有图像）
# python ~/multimodel_rumor_detection/backend/image_repair.py --input image-verification-corpus/images --output repaired_images

#进行数据集的处理
python ~/multimodel_rumor_detection/backend/data_prepare.py --ratio 0.2 --data_from weibo --enhance_images #ratio用于处理的数据集占总数据集的比例(默认"0.2")，data_from表示数据集的来源(默认"weibo")，--enhance_images启用图像增强

##注意事项：用于保存的文件夹以及数据集文件夹名称记得修改或确认是否存在

#获取CLIP的特征
python ~/multimodel_rumor_detection/backend/clip_feature_process.py --data_from weibo #data_from表示数据集的来源(默认"weibo") 

##注意事项：用于保存的文件夹名称记得修改或确认是否存在

#获取BERT的特征
python ~/multimodel_rumor_detection/backend/bert_feature_process.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

##注意事项：用于保存的文件夹名称记得修改或确认是否存在

#获取VGG的特征
python ~/multimodel_rumor_detection/backend/vgg_feature_process.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

##注意事项：用于保存的文件夹名称记得修改或确认是否存在

#调用大模型API进行新闻分析
python ~/multimodel_rumor_detection/backend/judge_by_bigmodal.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

##注意事项：记得输入自己的API Key；用于保存的文件夹名称记得修改或确认是否存在

#对得到的理由依据进行处理获取特征矩阵
python ~/multimodel_rumor_detection/backend/reason_feature_process.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

#对得到的所有特征矩阵进行拼接融合
python ~/multimodel_rumor_detection/backend/get_mixed_feature.py --data_from weibo #data_from表示数据集的来源(默认"weibo")

##注意事项：用于保存的文件夹名称记得修改或确认是否存在

#对得到的融合矩阵进行训练与测试，获得最佳权重(保存为"best_model.pth")
python ~/multimodel_rumor_detection/backend/model_and_train.py
