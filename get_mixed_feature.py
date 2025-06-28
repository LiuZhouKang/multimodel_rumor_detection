# 特征混合得到待处理的特征数据集
import numpy as np
import argparse

# 获取对应的待处理数据集
parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='weibo', help='数据集来源 (默认: weibo)')
args = parser.parse_args()

train_clip_text=np.load("clip_feature/train_text_clip_feature.npy")
train_clip_image=np.load("clip_feature/train_image_clip_feature.npy")
train_vgg_image=np.load("normal_feature/train_image_VGG_feature.npy")
train_bert_text=np.load("normal_feature/train_text_Bert_feature.npy")
train_reason_text=np.load("reason_feature/train_text_reason_feature.npy")
train_reason_image=np.load("reason_feature/train_image_reason_feature.npy")

test_clip_text=np.load("clip_feature/test_text_clip_feature.npy")
test_clip_image=np.load("clip_feature/test_image_clip_feature.npy")
test_vgg_image=np.load("normal_feature/test_image_VGG_feature.npy")
test_bert_text=np.load("normal_feature/test_text_Bert_feature.npy")
test_reason_text=np.load("reason_feature/test_text_reason_feature.npy")
test_reason_image=np.load("reason_feature/test_image_reason_feature.npy")
if args.data_from=='weibo':
    train_label=np.load("weibo_datasets/train_label.npy")
    test_label=np.load("weibo_datasets/test_label.npy")
elif args.data_from=='Twitter':
    train_label=np.load("Twitter_datasets/train_label.npy")
    test_label=np.load("Twitter_datasets/test_label.npy")
np.save("main_datasets/train_label.npy", train_label)
np.save("main_datasets/test_label.npy", test_label)

# 计算匹配损失
def calculate_L_match(vgg_image,bert_text,clip_image,clip_text):
    return np.square(clip_text-bert_text)+np.square(clip_image-vgg_image)

# 计算余弦相似度
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return abs(dot_product / (norm_A * norm_B))

# 计算simulation_score
def calculate_simulation_score(vgg_image,bert_text,clip_image,clip_text):
    # 计算simulation_score
    simulation_score=(cosine_similarity(vgg_image,bert_text)+cosine_similarity(clip_image,bert_text)+cosine_similarity(clip_text,vgg_image))/3
    return simulation_score

def get_mixed_feature():
    
    train_mixed_feature_list=[]
    test_mixed_feature_list=[]
    match_loss_list=[]

    print("\n------------------1.正在处理训练集中融合后的特征矩阵----------------")
    for i in range(len(train_label)):
        # 计算simulation_score
        simulation_score=calculate_simulation_score(train_vgg_image[i],train_bert_text[i],train_clip_image[i],train_clip_text[i])
        # 拼接文本和语义表征
        mixed_1=np.concatenate((train_bert_text[i],train_vgg_image[i]),axis=0)
        # 拼接文本与语义可信表征
        mixed_2=np.concatenate((train_reason_text[i],train_reason_image[i]),axis=0)
        #融合两个表征
        mixed_feature=(1-simulation_score)*mixed_1+simulation_score*mixed_2

        # mixed_feature=np.concatenate((train_bert_text[i],train_vgg_image[i]),axis=0)

        train_mixed_feature_list.append(mixed_feature)
    print("------------------2.训练集中融合后的特征矩阵处理完毕----------------")

    print("------------------3.正在处理测试集中融合后的特征矩阵----------------")
    for i in range(len(test_label)):
        # 计算simulation_score
        simulation_score=calculate_simulation_score(test_vgg_image[i],test_bert_text[i],test_clip_image[i],test_clip_text[i])
        # 拼接文本和语义表征
        mixed_1=np.concatenate((test_bert_text[i],test_vgg_image[i]),axis=0)
        # 拼接文本与语义可信表征
        mixed_2=np.concatenate((test_reason_text[i],test_reason_image[i]),axis=0)
        #融合两个表征
        mixed_feature=(1-simulation_score)*mixed_1+simulation_score*mixed_2

        # mixed_feature=np.concatenate((test_bert_text[i],test_vgg_image[i]),axis=0)

        test_mixed_feature_list.append(mixed_feature)
    print("------------------4.测试集中融合后的特征矩阵处理完毕----------------")

    # print("----------------------5.正在计算训练集中匹配损失--------------------")
    # for i in range(len(train_label)):
    #     # 计算匹配损失
    #     L_match=calculate_L_match(train_vgg_image[i],train_bert_text[i],train_clip_image[i],train_clip_text[i])
        
    #     match_loss_list.append(L_match)
    # print("----------------------6.训练集中匹配损失计算完毕--------------------")
   
    return train_mixed_feature_list,test_mixed_feature_list,match_loss_list

if __name__ == '__main__':
    train_mixed_feature_list,test_mixed_feature_list,match_loss_list=get_mixed_feature()
    train_mixed_feature=np.array(train_mixed_feature_list)
    test_mixed_feature=np.array(test_mixed_feature_list)
    match_loss=np.array(match_loss_list)

    print("----------------------5.正在保存融合后的特征矩阵--------------------")
    np.save("main_datasets/train_mixed_feature.npy",train_mixed_feature)
    np.save("main_datasets/test_mixed_feature.npy",test_mixed_feature)
    print("----------------------6.融合后的特征矩阵保存完毕--------------------\n")

    # print("--------------------------9.正在保存匹配损失------------------------")
    # np.save("loss/match_loss.npy",match_loss)
    # print("--------------------------10.匹配损失保存完毕-----------------------\n")

        



    


