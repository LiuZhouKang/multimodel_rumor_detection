import os
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def get_weibo_datasets(ratio=0.2):
    '''获取微博的数据集'''
    '''返回值为图像文件名列表和对应的文本列表和标签'''

    test_image_path=[]
    train_image_path=[]
    test_text=[]
    train_text=[]
    test_label=[]
    train_label=[]

    all_rumor_images = [img for img in os.listdir("MM17-WeiboRumorSet/rumor_images") 
                     if not (img.endswith('.gif') or img.endswith('.txt'))]
    all_nonrumor_images = [img for img in os.listdir("MM17-WeiboRumorSet/nonrumor_images") 
                     if not (img.endswith('.gif') or img.endswith('.txt'))]

    print("\n---------------1.处理训练集中非谣言部分数据---------------")
    train_nonrumor=open("MM17-WeiboRumorSet/tweets/train_nonrumor.txt","r", encoding = "utf-8").readlines()
    for index in range(int(len(train_nonrumor)*ratio/3)):
        text=train_nonrumor[index*3+2].strip()
        image_whole=train_nonrumor[index*3+1].replace("|null","").split("|")
        for images in image_whole:
            image=images.split("/")[-1].split(".")[0]
            if(image+".jpg" not in all_nonrumor_images):
                continue
            image_path="MM17-WeiboRumorSet/nonrumor_images/"+image+".jpg"
            train_image_path.append(image_path)
            train_text.append(text)
            train_label.append([1,0])

    print("---------------2.处理测试集中非谣言部分数据---------------")
    test_nonrumor=open("MM17-WeiboRumorSet/tweets/test_nonrumor.txt","r", encoding = "utf-8").readlines()
    for index in range(int(len(test_nonrumor)*ratio/3)):
        text=test_nonrumor[index*3+2].strip()
        image_whole=test_nonrumor[index*3+1].replace("|null","").split("|")
        for images in image_whole:
            image=images.split("/")[-1].split(".")[0]
            if(image+".jpg" not in all_nonrumor_images):
                continue
            image_path="MM17-WeiboRumorSet/nonrumor_images/"+image+".jpg"
            test_image_path.append(image_path)
            test_text.append(text)
            test_label.append([1,0])

    print("----------------3.处理训练集中谣言部分数据----------------")
    train_rumor=open("MM17-WeiboRumorSet/tweets/train_rumor.txt","r", encoding = "utf-8").readlines()
    for index in range(int(len(train_rumor)*ratio/3)):
        text=train_rumor[index*3+2].strip()
        image_whole=train_rumor[index*3+1].replace("|null","").split("|")
        for images in image_whole:
            image=images.split("/")[-1].split(".")[0]
            if(image+".jpg" not in all_rumor_images):
                continue
            image_path="MM17-WeiboRumorSet/rumor_images/"+image+".jpg"
            train_image_path.append(image_path)
            train_text.append(text)
            train_label.append([0,1])

    print("----------------4.处理测试集中谣言部分数据----------------")
    test_rumor=open("MM17-WeiboRumorSet/tweets/test_rumor.txt","r", encoding = "utf-8").readlines()
    for index in range(int(len(test_rumor)*ratio/3)):
        text=test_rumor[index*3+2].strip()
        image_whole=test_rumor[index*3+1].replace("|null","").split("|")
        for images in image_whole:
            image=images.split("/")[-1].split(".")[0]
            if(image+".jpg" not in all_rumor_images):
                continue
            image_path="MM17-WeiboRumorSet/rumor_images/"+image+".jpg"
            test_image_path.append(image_path)
            test_text.append(text)
            test_label.append([0,1])

    # print(len(train_image_path))
    # print(len(train_text))
    # print(len(test_image_path))
    # print(len(test_text))
    # print(train_label)

    #保存得到的数据集
    print("------------------数据处理完毕，得到数据集----------------")
    np.save("weibo_datasets/train_image_path.npy", train_image_path)
    np.save("weibo_datasets/train_text.npy", train_text)
    np.save("weibo_datasets/train_label.npy", train_label)
    np.save("weibo_datasets/test_image_path.npy", test_image_path)
    np.save("weibo_datasets/test_text.npy", test_text)
    np.save("weibo_datasets/test_label.npy", test_label)
    print("-----------------------数据集保存完毕---------------------")

def load_weibo_datasets():
    '''加载微博的数据集'''
    train_image_path=np.load("weibo_datasets/train_image_path.npy")
    train_text=np.load("weibo_datasets/train_text.npy")
    train_label=np.load("weibo_datasets/train_label.npy")
    test_image_path=np.load("weibo_datasets/test_image_path.npy")
    test_text=np.load("weibo_datasets/test_text.npy")
    test_label=np.load("weibo_datasets/test_label.npy")
    return train_image_path,train_text,train_label,test_image_path,test_text,test_label

def extract_fields(data_path):
    # 显式指定编码为utf-8
    df = pd.read_csv(data_path, sep='\t', header=0, encoding='utf-8')

    # 提取各字段为独立数组
    post_ids = df['post_id'].tolist()
    post_texts = df['post_text'].tolist()
    user_ids = df['user_id'].tolist()
    image_ids = df['image_id(s)'].tolist()
    usernames = df['username'].tolist()
    timestamps = df['timestamp'].tolist()
    labels = df['label'].tolist()  # 原始标签为字符串（如 'fake'/'real'）

    # 将标签字符串转换为列表格式（[1,0] 表示非谣言，[0,1] 表示谣言）
    converted_labels = []
    for label in labels:
        if label == 'fake':  # 假设 'fake' 对应谣言标签
            converted_labels.append([0, 1])
        elif label =='real':  # 假设'real' 对应非谣言标签
            converted_labels.append([1, 0])
        else:
            converted_labels.append([-1, -1])  # 未知标签默认值

    return {
        'post_ids': post_ids,
        'post_texts': post_texts,
        'user_id': user_ids,
        'image_ids': image_ids,
        'usernames': usernames,
        'timestamps': timestamps,
        'labels': converted_labels  # 转换后的标签列表
    }

def remove_urls(text):
    """去除文本中的网址"""
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text).strip()


def create_image_text_mapping(posts_data, images_dir):
    """创建帖子ID、文本与图片路径的映射列表"""
    post_id_list = []
    post_text_list = []
    image_path_list = []

    # 获取图片目录中所有文件的基本名称（不含扩展名）
    available_images = {}
    for filename in os.listdir(images_dir):
        file_path = os.path.join(images_dir, filename)
        if os.path.isfile(file_path):
            base_name, ext = os.path.splitext(filename)
            # 保存图片路径（保留完整路径，用于后续处理）
            available_images[base_name] = file_path

    # 遍历每条帖子
    for post_id, post_text, image_ids_str in zip(
            posts_data['post_ids'],
            posts_data['post_texts'],
            posts_data['image_ids']
    ):
        # 跳过没有图片的帖子
        if pd.isna(image_ids_str) or image_ids_str.strip() == '':
            continue

        # 分割图片ID
        image_ids = [id.strip() for id in image_ids_str.split(',') if id.strip()]

        # 去除文本中的网址
        post_text = remove_urls(post_text)

        # 为每个图片ID生成完整路径并添加到列表
        for image_id in image_ids:
            if image_id in available_images:
                if not available_images[image_id].lower().endswith(('.jpg', '.jpeg', '.png','.bmp','.gif')):
                    continue
                post_id_list.append(post_id)
                post_text_list.append(post_text)
                image_path_list.append(available_images[image_id])
            # else:
                # print(f"警告: 未找到图片 - {image_id}（所有格式）")

    return post_id_list, post_text_list, image_path_list


def split_data(post_text_list, image_path_list, labels, ratio=0.2):
    """将数据分为训练集和测试集"""
    total = len(post_text_list)
    text_list=post_text_list[:int(total*ratio)]
    image_path_list=image_path_list[:int(total*ratio)]
    labels=labels[:int(total*ratio)]

    train_text, test_text, train_image_path, test_image_path, train_label, test_label = train_test_split(
    text_list, image_path_list, labels, test_size=0.3, random_state=42)

    return train_image_path, train_text, train_label, test_image_path, test_text, test_label


def save_datasets(train_image_path, train_text, train_label, test_image_path, test_text, test_label):
    """保存数据集"""
    # 确保保存目录存在
    os.makedirs("Twitter_datasets", exist_ok=True)

    print("------------------数据处理完毕，得到数据集----------------")
    np.save("Twitter_datasets/train_image_path.npy", train_image_path)
    np.save("Twitter_datasets/train_text.npy", train_text)
    np.save("Twitter_datasets/train_label.npy", train_label)
    np.save("Twitter_datasets/test_image_path.npy", test_image_path)
    np.save("Twitter_datasets/test_text.npy", test_text)
    np.save("Twitter_datasets/test_label.npy", test_label)
    print("-----------------------数据集保存完毕---------------------")

    # print(f"训练集: {len(train_image_path)} 样本 (非谣言: {train_label.count([1, 0])}, 谣言: {train_label.count([0, 1])})")
    # print(f"测试集: {len(test_image_path)} 样本 (非谣言: {test_label.count([1, 0])}, 谣言: {test_label.count([0, 1])})")

def get_twitter_datasets(ratio=0.2):
    
    # 构建数据文件和图片目录的相对路径
    data_path = "image-verification-corpus/posts.txt"
    images_dir = "image-verification-corpus/images"

    # print(f"数据路径: {data_path}")
    # print(f"图片目录: {images_dir}")

    # 提取字段
    result = extract_fields(data_path)

    # 创建文本与图片的映射列表
    post_id_list, post_text_list, image_path_list = create_image_text_mapping(result, images_dir)

    # 分割数据为训练集和测试集
    train_image_path, train_text, train_label, test_image_path, test_text, test_label = split_data(
        post_text_list, image_path_list, result['labels'], ratio)

    # 保存数据集
    save_datasets(train_image_path, train_text, train_label, test_image_path, test_text, test_label)

    # print("训练集前五条数据:")
    # for i in range(min(5, len(loaded_train_image_path))):
    #     # 提取相对路径（从 devset 开始）
    #     img_path = os.path.relpath(loaded_train_image_path[i], start=script_dir)
    #     # 转换反斜杠为正斜杠，与示例一致
    #     img_path = img_path.replace("\\", "/")
    #     label = loaded_train_label[i]
    #     if not isinstance(label, list):
    #         label = label.tolist()
    #     print(f"图片路径: {img_path}, 文本: {loaded_train_text[i]}, 标签: {label}")
    # print("-" * 50)

    # print("测试集前五条数据:")
    # for i in range(min(5, len(loaded_test_image_path))):
    #     img_path = os.path.relpath(loaded_test_image_path[i], start=script_dir)
    #     img_path = img_path.replace("\\", "/")
    #     label = loaded_test_label[i]
    #     if not isinstance(label, list):
    #         label = label.tolist()
    #     print(f"图片路径: {img_path}, 文本: {loaded_test_text[i]}, 标签: {label}")
    # print("-" * 50)

def load_twitter_datasets():
    """加载推特的数据集"""
    train_image_path = np.load("Twitter_datasets/train_image_path.npy", allow_pickle=True)
    train_text = np.load("Twitter_datasets/train_text.npy", allow_pickle=True)
    train_label = np.load("Twitter_datasets/train_label.npy", allow_pickle=True)
    test_image_path = np.load("Twitter_datasets/test_image_path.npy", allow_pickle=True)
    test_text = np.load("Twitter_datasets/test_text.npy", allow_pickle=True)
    test_label = np.load("Twitter_datasets/test_label.npy", allow_pickle=True)
    return train_image_path, train_text, train_label, test_image_path, test_text, test_label

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=0.2, help='数据集采样比例 (默认: 0.2)')
    parser.add_argument('--data_from', type=str, default='weibo', help='数据集来源 (默认: weibo)')
    args = parser.parse_args()

    if args.data_from=='weibo':
        get_weibo_datasets(args.ratio)
    elif args.data_from=='Twitter':
        get_twitter_datasets(args.ratio)
