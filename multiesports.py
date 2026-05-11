import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import os
import argparse
from zhipuai import ZhipuAI
import json

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


# ==================== 配置参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='weibo', help='数据集来源')
parser.add_argument('--similarity_threshold', type=float, default=0.6, help='文本图像相似度阈值')
parser.add_argument('--use_maddpg', action='store_true', default=True, help='是否使用MADDPG多智能体')
parser.add_argument('--text', type=str, default='', help='待判断的文本内容')
parser.add_argument('--image_path', type=str, default='', help='待判断的图像路径')
parser.add_argument('--batch_mode', action='store_true', default=False, help='是否批量处理')
args = parser.parse_args()

# 智谱AI客户端
client = ZhipuAI(api_key='e9798760e9aa46008610c97155327256.EducRUXjICoEZwj4')  # 替换为你的API Key

# ==================== 相似度计算模块 ====================
def calculate_cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2) 

def extract_text_features(text):
    """提取文本特征向量（简化的特征提取）"""
    # 这里可以用你的CLIP或BERT特征提取方法
    # 为了简化，这里使用字符编码作为特征
    features = np.zeros(512)
    if len(text) > 0:
        text_chars = text[:512]  # 截取前512个字符
        for i, char in enumerate(text_chars):
            features[i] = ord(char) % 256
    return features / 255.0

def extract_image_features(image_path):
    """提取图像特征向量（简化的特征提取）"""
    # 这里可以用你的VGG或CLIP图像特征提取方法
    # 为了简化，这里生成随机特征
    if not os.path.exists(image_path):
        return np.random.randn(512) / 2.0
    
    # 实际应该使用预训练模型提取特征
    # 这里返回随机特征作为示例
    features = np.random.randn(512)
    return features / 2.0

def calculate_similarity(text, image_path):
    """计算文本和图像的相似度"""
    text_features = extract_text_features(text)
    image_features = extract_image_features(image_path)
    
    # 计算余弦相似度
    similarity = calculate_cosine_similarity(text_features, image_features)
    
    # 转换为0-1范围（cosine_similarity范围是-1到1）
    normalized_similarity = (similarity + 1) / 2
    
    return normalized_similarity, text_features, image_features

# ==================== MADDPG模型定义 ====================
class Actor(nn.Module):
    """Actor网络（策略网络）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.layer_norm(x)
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # 输出在[-1, 1]范围内
        return action

class Critic(nn.Module):
    """Critic网络（值函数网络）"""
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim * num_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, state, actions):
        x = torch.cat([state] + list(actions), dim=-1)
        x = F.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class MADDPGAgent:
    """MADDPG智能体管理器"""
    def __init__(self, num_agents, state_dims, action_dims, hidden_dim=256):
        self.num_agents = num_agents
        self.actors = [Actor(state_dims[i], action_dims[i], hidden_dim) for i in range(num_agents)]
    
    def load_models(self, model_path):
        """加载训练好的模型参数"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        for i in range(self.num_agents):
            if f'actor_state_dicts_{i}' in checkpoint:
                self.actors[i].load_state_dict(checkpoint[f'actor_state_dicts_{i}'])
            elif 'actor_state_dicts' in checkpoint and i < len(checkpoint['actor_state_dicts']):
                self.actors[i].load_state_dict(checkpoint['actor_state_dicts'][i])
        
        print(f"✅ 成功加载 {self.num_agents} 个智能体模型")
    
    def select_action(self, states):
        """选择动作（无探索噪声，用于推理）"""
        actions = []
        for i in range(self.num_agents):
            state_tensor = torch.FloatTensor(states[i]).unsqueeze(0)
            with torch.no_grad():
                action = self.actors[i](state_tensor).squeeze(0).numpy()
            actions.append(action)
        return actions

# ==================== 大模型智能体模块 ====================
class LLMTextAgent:
    """基于大模型的文本分析智能体"""
    def __init__(self, agent_id, role, system_prompt):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.confidence = 0.5
    
    def analyze_with_llm(self, text):
        """使用大模型进行分析"""
        try:
            if args.data_from == "weibo":
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"请分析以下新闻文本的真实性：\n\n{text[:1000]}\n\n请给出判断（真实/虚假/不确定）和置信度（0-100%）。"}
                ]
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Please analyze the authenticity of this news text:\n\n{text[:1000]}\n\nProvide judgment (true/false/uncertain) and confidence (0-100%)."}
                ]
            
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=messages,
                max_tokens=150,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            return self._parse_llm_response(result)
            
        except Exception as e:
            print(f"⚠️ 智能体 {self.role} 分析失败: {e}")
            return {"verdict": "分析失败", "confidence": 0.5}
    
    def _parse_llm_response(self, response):
        """解析大模型的响应"""
        response_lower = response.lower()
        
        # 判断结果
        if "真实" in response or "true" in response_lower:
            verdict = "真实"
        elif "虚假" in response or "false" in response_lower or "fake" in response_lower:
            verdict = "虚假"
        else:
            verdict = "不确定"
        
        # 提取置信度
        confidence = 0.7  # 默认值
        if "置信度" in response:
            try:
                conf_str = response.split("置信度")[1].split("%")[0].strip()
                confidence = min(max(float(conf_str) / 100, 0), 1)
            except:
                pass
        elif "confidence" in response_lower:
            try:
                conf_str = response_lower.split("confidence")[1].split("%")[0].strip()
                confidence = min(max(float(conf_str) / 100, 0), 1)
            except:
                pass
        
        self.confidence = confidence
        return {"verdict": verdict, "confidence": confidence, "reasoning": response[:100]}

def create_text_agents():
    """创建文本分析智能体团队"""
    agents = []
    
    if args.data_from == "weibo":
        agents.append(LLMTextAgent(0, "事实核查专家", "你是专业的事实核查专家，擅长分析新闻中的具体事实、数字信息和可验证的细节。"))
        agents.append(LLMTextAgent(1, "逻辑分析专家", "你是逻辑分析专家，擅长发现文本中的逻辑矛盾、夸大表述和推理漏洞。"))
        agents.append(LLMTextAgent(2, "情感倾向分析师", "你是情感分析专家，擅长识别煽动性语言、情绪化表述和操纵性内容。"))
        agents.append(LLMTextAgent(3, "来源可信度分析师", "你是来源可信度专家，擅长评估消息来源的权威性和可靠性。"))
    else:
        agents.append(LLMTextAgent(0, "Fact-Checker", "You are a professional fact-checker, skilled at analyzing specific facts, numbers, and verifiable details in news."))
        agents.append(LLMTextAgent(1, "Logical Analyst", "You are a logical analysis expert, skilled at identifying logical contradictions, exaggerations, and reasoning flaws."))
        agents.append(LLMTextAgent(2, "Emotional Analyst", "You are an emotional analysis expert, skilled at recognizing煽动性语言, emotional manipulation, and sensational content."))
    
    return agents

# ==================== 主判断逻辑 ====================
def judge_content(text, image_path=None):
    """
    主判断函数：先判断文本图像相似度，再判断内容真实性
    
    返回:
        dict: 包含相似度、各阶段判断、最终结果等信息
    """
    print("=" * 60)
    print("🔍 开始内容真实性判断")
    print("=" * 60)
    
    result = {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "image_path": image_path,
        "similarity": 0,
        "similarity_result": "未知",
        "agent_judgments": [],
        "maddpg_judgment": None,
        "final_judgment": "未知",
        "confidence": 0,
        "reasoning": ""
    }
    
    # ========== 阶段1：文本图像相似度判断 ==========
    print("\n📊 阶段1: 文本图像相似度判断")
    
    if image_path and os.path.exists(image_path):
        similarity, text_features, image_features = calculate_similarity(text, image_path)
        result["similarity"] = similarity
        
        print(f"  文本特征维度: {text_features.shape}")
        print(f"  图像特征维度: {image_features.shape}")
        print(f"  相似度得分: {similarity:.4f}")
        print(f"  相似度阈值: {args.similarity_threshold}")
        
        if similarity < args.similarity_threshold:
            result["similarity_result"] = "不相似"
            result["final_judgment"] = "虚假"
            result["reasoning"] = f"文本与图像相似度过低 ({similarity:.2%} < {args.similarity_threshold:.0%})，直接判定为虚假"
            print(f"  ❌ 相似度不足，直接判定为: {result['final_judgment']}")
            return result
        else:
            result["similarity_result"] = "相似"
            print(f"  ✅ 相似度达标，进入深度分析")
    else:
        print(f"  ⚠️ 无图像或图像路径无效，跳过相似度检查")
        result["similarity_result"] = "无图像"
    
    # ========== 阶段2：多智能体真实性判断 ==========
    print("\n🤖 阶段2: 多智能体真实性判断")
    
    if args.use_maddpg:
        # 使用MADDPG多智能体判断
        print("  使用MADDPG多智能体系统...")
        
        try:
            # 加载训练好的MADDPG模型
            model_path = "maddpg_content_judgement_model.pth"
            if not os.path.exists(model_path):
                print(f"  ⚠️ MADDPG模型文件不存在: {model_path}")
                print("  回退到大模型智能体判断...")
                return judge_with_llm_agents(text, result)
            
            # 初始化MADDPG智能体
            state_dims = [512, 512]  # 需要与训练时保持一致
            action_dims = [2, 2]
            maddpg_agent = MADDPGAgent(num_agents=2, state_dims=state_dims, action_dims=action_dims)
            maddpg_agent.load_models(model_path)
            
            # 提取特征
            text_features = extract_text_features(text)
            image_features = extract_image_features(image_path) if image_path else np.zeros(512)
            
            # 获取MADDPG判断
            states = [text_features, image_features]
            actions = maddpg_agent.select_action(states)
            
            # 解释动作（假设动作[0] > 0表示真实）
            text_judgment = "真实" if actions[0][0] > 0 else "虚假"
            image_judgment = "真实" if actions[1][0] > 0 else "虚假"
            text_confidence = abs(actions[0][0])
            image_confidence = abs(actions[1][0])
            
            # 综合判断
            if text_judgment == "真实" and image_judgment == "真实":
                final_judgment = "真实"
                confidence = (text_confidence + image_confidence) / 2
            elif text_judgment == "虚假" and image_judgment == "虚假":
                final_judgment = "虚假"
                confidence = (text_confidence + image_confidence) / 2
            else:
                final_judgment = "不确定"
                confidence = 0.5
            
            result["maddpg_judgment"] = {
                "text_judgment": text_judgment,
                "text_confidence": float(text_confidence),
                "image_judgment": image_judgment,
                "image_confidence": float(image_confidence)
            }
            result["final_judgment"] = final_judgment
            result["confidence"] = float(confidence)
            result["reasoning"] = f"MADDPG多智能体判断: 文本智能体认为{text_judgment}(置信度{text_confidence:.2f})，图像智能体认为{image_judgment}(置信度{image_confidence:.2f})"
            
            print(f"  ✅ MADDPG判断完成: {final_judgment} (置信度: {confidence:.2f})")
            
        except Exception as e:
            print(f"  ❌ MADDPG判断失败: {e}")
            print("  回退到大模型智能体判断...")
            return judge_with_llm_agents(text, result)
    
    else:
        # 使用大模型智能体判断
        return judge_with_llm_agents(text, result)
    
    return result

def judge_with_llm_agents(text, result):
    """使用大模型智能体进行判断"""
    print("  使用大模型智能体团队...")
    
    agents = create_text_agents()
    judgments = []
    
    # 并行分析（简化：顺序执行）
    for agent in agents:
        print(f"    {agent.role} 分析中...", end="")
        judgment = agent.analyze_with_llm(text)
        judgments.append({
            "agent": agent.role,
            "verdict": judgment["verdict"],
            "confidence": judgment["confidence"],
            "reasoning": judgment.get("reasoning", "")
        })
        print(f" 结果: {judgment['verdict']} ({judgment['confidence']:.2f})")
    
    result["agent_judgments"] = judgments
    
    # 投票决定最终结果
    real_votes = sum(1 for j in judgments if j["verdict"] == "真实")
    fake_votes = sum(1 for j in judgments if j["verdict"] == "虚假")
    avg_confidence = np.mean([j["confidence"] for j in judgments])
    
    if real_votes > fake_votes:
        final_judgment = "真实"
    elif fake_votes > real_votes:
        final_judgment = "虚假"
    else:
        final_judgment = "不确定"
    
    result["final_judgment"] = final_judgment
    result["confidence"] = float(avg_confidence)
    result["reasoning"] = f"智能体投票: {real_votes}票真实 vs {fake_votes}票虚假"
    
    print(f"  ✅ 大模型智能体判断完成: {final_judgment} (置信度: {avg_confidence:.2f})")
    
    return result

# ==================== 批量处理函数 ====================
def batch_judge(texts, image_paths=None):
    """批量处理多个内容"""
    results = []
    
    if image_paths is None:
        image_paths = [None] * len(texts)
    
    for i, (text, image_path) in enumerate(zip(texts, image_paths)):
        print(f"\n📦 处理第 {i+1}/{len(texts)} 个样本")
        result = judge_content(text, image_path)
        results.append(result)
    
    return results

# ==================== 主函数 ====================
def main():
    print("🤖 多模态内容真实性判断系统")
    print("📌 策略: 先判断文本图像相似度，再分析内容真实性")
    print(f"📊 相似度阈值: {args.similarity_threshold}")
    print(f"🔧 使用MADDPG: {'是' if args.use_maddpg else '否'}")
    
    if args.batch_mode:
        # 批量处理模式
        print("\n📦 批量处理模式")
        
        # 这里可以从文件加载数据
        # 示例：从测试集加载
        try:
            test_texts = np.load("weibo_datasets/test_text.npy", allow_pickle=True)[:5]
            test_images = np.load("weibo_datasets/test_image_path.npy", allow_pickle=True)[:5]
            
            print(f"加载了 {len(test_texts)} 个测试样本")
            
            results = batch_judge(test_texts.tolist(), test_images.tolist())
            
            # 统计结果
            print("\n" + "="*60)
            print("📈 批量处理统计结果")
            print("="*60)
            
            judgments = [r["final_judgment"] for r in results]
            real_count = judgments.count("真实")
            fake_count = judgments.count("虚假")
            uncertain_count = judgments.count("不确定")
            
            print(f"真实: {real_count} ({real_count/len(results):.1%})")
            print(f"虚假: {fake_count} ({fake_count/len(results):.1%})")
            print(f"不确定: {uncertain_count} ({uncertain_count/len(results):.1%})")
            
            # 保存结果
            with open("judgment_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存到 judgment_results.json")
            
        except Exception as e:
            print(f"❌ 批量处理失败: {e}")
            print("使用示例数据进行测试...")
            
            # 示例数据
            example_texts = [
                "今日，市政府宣布将投资100亿元用于城市基础设施建设。",
                "惊爆！某明星被曝出轨，照片证据确凿！",
                "科学家发现新行星，可能存在生命迹象。"
            ]
            
            results = batch_judge(example_texts)
            
            for i, result in enumerate(results):
                print(f"\n示例 {i+1}:")
                print(f"  文本: {result['text']}")
                print(f"  相似度: {result['similarity']:.2%}")
                print(f"  最终判断: {result['final_judgment']} (置信度: {result['confidence']:.2f})")
    
    else:
        # 单条处理模式
        if args.text:
            text = args.text
            image_path = args.image_path if args.image_path else None
        else:
            # 交互模式
            print("\n💬 请输入要判断的内容:")
            text = input("文本内容: ").strip()
            image_path = input("图像路径（可选，直接回车跳过）: ").strip()
            if not image_path:
                image_path = None
        
        if not text:
            print("❌ 必须提供文本内容")
            return
        
        # 执行判断
        result = judge_content(text, image_path)
        
        # 打印详细结果
        print("\n" + "="*60)
        print("📋 判断结果详情")
        print("="*60)
        
        print(f"📝 文本: {result['text']}")
        if result['image_path']:
            print(f"🖼️  图像: {result['image_path']}")
        
        print(f"\n📊 相似度分析:")
        print(f"  相似度得分: {result['similarity']:.2%}")
        print(f"  相似度结果: {result['similarity_result']}")
        
        if result['agent_judgments']:
            print(f"\n🤖 智能体分析结果:")
            for judgment in result['agent_judgments']:
                print(f"  {judgment['agent']}: {judgment['verdict']} (置信度: {judgment['confidence']:.2f})")
        
        if result['maddpg_judgment']:
            print(f"\n🧠 MADDPG多智能体结果:")
            mj = result['maddpg_judgment']
            print(f"  文本智能体: {mj['text_judgment']} (置信度: {mj['text_confidence']:.2f})")
            print(f"  图像智能体: {mj['image_judgment']} (置信度: {mj['image_confidence']:.2f})")
        
        print(f"\n🎯 最终判断:")
        print(f"  结果: {result['final_judgment']}")
        print(f"  置信度: {result['confidence']:.2f}")
        print(f"  推理: {result['reasoning']}")
        
        # 保存单个结果
        result_file = "single_judgment_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 详细结果已保存到 {result_file}")

if __name__ == "__main__":
    main()