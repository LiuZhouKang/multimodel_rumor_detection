import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    X_train=torch.FloatTensor(np.load("main_datasets/train_mixed_feature.npy"))
    Y_train=torch.FloatTensor(np.load("main_datasets/train_label.npy"))
    X_test=torch.FloatTensor(np.load("main_datasets/test_mixed_feature.npy"))
    Y_test=torch.FloatTensor(np.load("main_datasets/test_label.npy"))

    # 初始化模型
    model = BinaryClassifier(input_dim=1024)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    best_accuracy = 0
    best_model_state = None
    best_epoch = 0
    best_metrics = {'precision': 0, 'recall': 0, 'f1': 0}  # 新增：记录最佳指标
    epochs = 100

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, torch.argmax(Y_train, 1))
        loss.backward()
        optimizer.step()

        # 验证步骤
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == torch.argmax(Y_test, 1)).float().mean()
            
            # 计算当前epoch的指标
            true_pos = ((predicted == 1) & (torch.argmax(Y_test, 1) == 1)).sum().float()
            false_pos = ((predicted == 1) & (torch.argmax(Y_test, 1) == 0)).sum().float()
            false_neg = ((predicted == 0) & (torch.argmax(Y_test, 1) == 1)).sum().float()
            precision = true_pos / (true_pos + false_pos + 1e-7)
            recall = true_pos / (true_pos + false_neg + 1e-7)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
            # 保存最佳模型和指标
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
                best_metrics = {
                    'precision': precision.item(),
                    'recall': recall.item(),
                    'f1': f1.item()
                }
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    
    # 打印最佳指标
    print(f'\nBest Performance at Epoch {best_epoch}:')
    print(f'Accuracy: {best_accuracy.item()*100:.2f}%')
    print(f'Precision: {best_metrics["precision"]:.4f}')
    print(f'Recall: {best_metrics["recall"]:.4f}')
    print(f'F1 Score: {best_metrics["f1"]:.4f}')
    




    


