// ==================== 全局变量 ====================
let trainingSocket = null;
let lossChart = null;
let accuracyChart = null;
let trainingStartTime = null;
let trainingTimer = null;

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', function() {
    initTrainingSocket();
    initDataConfig();
    initFeatureConfig();
    initTrainButtons();
    initCharts();
    checkSystemStatus();
});

// ==================== Socket.IO 连接 ====================
function initTrainingSocket() {
    trainingSocket = io();
    
    trainingSocket.on('connect', function() {
        console.log('已连接到训练服务器');
        addLog('success', '已连接到服务器');
    });
    
    // 接收训练进度
    trainingSocket.on('training_progress', function(data) {
        updateTrainingProgress(data);
    });
    
    // 接收训练指标
    trainingSocket.on('training_metrics', function(data) {
        updateTrainingMetrics(data);
    });
    
    // 接收日志
    trainingSocket.on('training_log', function(data) {
        addLog(data.level, data.message);
        
        // 监听训练完成日志，自动启用导出按钮
        if (data.level === 'success' && data.message.includes('训练完成')) {
            document.getElementById('stat-model-status').textContent = '已训练';
            document.getElementById('btn-export').disabled = false;
            addLog('info', '模型已自动保存为 best_model.pth');
        }
    });
    
    // 监听特征提取进度，特别关注reason阶段的完成
    trainingSocket.on('training_progress', function(data) {
        if (data.stage === 'reason' && data.status === 'completed') {
            // 当reason阶段完成后，启用融合按钮
            addLog('success', '推理特征提取完成！现在可以进行特征融合了。');
            document.getElementById('btn-fusion').disabled = false;
        }
    });
    
    trainingSocket.on('disconnect', function() {
        console.log('与服务器断开连接');
        addLog('warning', '与服务器断开连接');
    });
}

// ==================== 数据集配置 ====================
function initDataConfig() {
    const ratioSlider = document.getElementById('data-ratio');
    const ratioNumber = document.getElementById('data-ratio-number');
    
    // Slider同步到Number输入框
    ratioSlider.addEventListener('input', function() {
        ratioNumber.value = this.value;
    });
    
    // Number输入框同步到Slider
    ratioNumber.addEventListener('input', function() {
        let val = parseFloat(this.value);
        if (val >= 0.1 && val <= 1.0) {
            ratioSlider.value = this.value;
        }
    });
    
    const splitSlider = document.getElementById('data-split');
    const splitValue = document.getElementById('data-split-value');
    
    splitSlider.addEventListener('input', function() {
        const train = (this.value * 100).toFixed(0);
        const test = ((1 - this.value) * 100).toFixed(0);
        splitValue.textContent = `${train}%/${test}%`;
    });
    
    // 准备数据集按钮
    document.getElementById('btn-prepare-data').addEventListener('click', handlePrepareData);
}

async function handlePrepareData() {
    const btn = document.getElementById('btn-prepare-data');
    const source = document.getElementById('dataset-source').value;
    const ratio = document.getElementById('data-ratio-number').value;
    const enableImageEnhance = document.getElementById('enable-image-enhance').checked;
    
    btn.disabled = true;
    btn.querySelector('.btn-text').style.display = 'none';
    btn.querySelector('.btn-loading').style.display = 'flex';
    
    // 更新数据准备阶段状态
    updateStageStatus('prepare', 'running', '处理中...');
    addLog('info', `开始准备${source === 'weibo' ? '微博' : 'Twitter'}数据集 (图像增强: ${enableImageEnhance})...`);
    
    try {
        const response = await fetch('/api/train/prepare_data', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                data_from: source,
                ratio: parseFloat(ratio),
                enable_image_enhance: enableImageEnhance
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addLog('success', `数据集准备完成: ${data.message}`);
            updateStageStatus('prepare', 'completed', '完成');
            document.getElementById('stat-dataset').textContent = `${source} (ratio=${ratio})`;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        addLog('error', `数据集准备失败: ${error.message}`);
        updateStageStatus('prepare', 'error', '失败');
        alert('数据集准备失败: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'inline';
        btn.querySelector('.btn-loading').style.display = 'none';
    }
}

// ==================== 特征提取配置 ====================
function initFeatureConfig() {
    const reasonCheckbox = document.getElementById('extract-reason');
    const apiKeyGroup = document.getElementById('api-key-group');
    const modelProviderGroup = document.getElementById('model-provider-group');
    const modelProvider = document.getElementById('model-provider');
    const modelDesc = document.getElementById('model-desc');
    const apiKeyLabel = document.getElementById('api-key-label');
    const apiKeyDesc = document.getElementById('api-key-desc');
    
    // 更新模型提供商相关UI
    function updateModelProviderUI() {
        const provider = modelProvider.value;
        if (provider === 'zhipu') {
            modelDesc.textContent = '使用GLM-4系列大模型';
            apiKeyLabel.textContent = '智谱AI API密钥';
            apiKeyDesc.textContent = '用于调用GLM-4系列大模型';
            document.getElementById('api-key').placeholder = '输入您的智谱AI API Key...';
        } else if (provider === 'qwen') {
            modelDesc.textContent = '使用Qwen-Turbo/Qwen-VL模型';
            apiKeyLabel.textContent = '通义千问 API密钥';
            apiKeyDesc.textContent = '用于调用Qwen系列大模型';
            document.getElementById('api-key').placeholder = '输入您的通义千问 API Key...';
        }
    }
    
    reasonCheckbox.addEventListener('change', function() {
        const show = this.checked;
        apiKeyGroup.style.display = show ? 'block' : 'none';
        modelProviderGroup.style.display = show ? 'block' : 'none';
        if (show) {
            document.getElementById('stage-reason').style.display = 'grid';
            updateModelProviderUI();
        } else {
            document.getElementById('stage-reason').style.display = 'none';
        }
    });
    
    // 监听模型提供商切换
    modelProvider.addEventListener('change', function() {
        updateModelProviderUI();
    });
    
    document.getElementById('btn-extract-features').addEventListener('click', handleExtractFeatures);
    document.getElementById('btn-fusion').addEventListener('click', handleFeatureFusion);
}

async function handleExtractFeatures() {
    const btn = document.getElementById('btn-extract-features');
    const source = document.getElementById('dataset-source').value;
    const extractReason = document.getElementById('extract-reason').checked;
    const apiKey = document.getElementById('api-key').value;
    const modelProvider = document.getElementById('model-provider').value;
    
    btn.disabled = true;
    btn.querySelector('.btn-text').style.display = 'none';
    btn.querySelector('.btn-loading').style.display = 'flex';
    
    try {
        addLog('info', '开始提取特征...');
        updateStageStatus('clip', 'running', '提取中...');
        
        // 提取CLIP特征
        await callFeatureAPI('/api/train/extract_clip', source);
        updateStageStatus('clip', 'completed', '完成');
        addLog('success', 'CLIP特征提取完成');
        
        updateStageStatus('bert', 'running', '提取中...');
        // 提取BERT特征
        await callFeatureAPI('/api/train/extract_bert', source);
        updateStageStatus('bert', 'completed', '完成');
        addLog('success', 'BERT特征提取完成');
        
        updateStageStatus('vgg', 'running', '提取中...');
        // 提取VGG特征
        await callFeatureAPI('/api/train/extract_vgg', source);
        updateStageStatus('vgg', 'completed', '完成');
        addLog('success', 'VGG特征提取完成');
        
        if (extractReason) {
            updateStageStatus('reason', 'running', '推理中...');
            // 提取推理特征（异步执行，通过WebSocket推送进度）
            addLog('info', `正在调用${modelProvider === 'zhipu' ? '智谱GLM' : '通义千问'}大模型进行推理...`);
            await callFeatureAPI('/api/train/extract_reason', source, apiKey, modelProvider);
            addLog('info', '推理任务已启动，请等待日志输出...');
            // 注意：由于是异步执行，这里不立即启用融合按钮，等待WebSocket推送reason完成消息
        } else {
            // 如果没有启用大模型推理，提取完VGG后就启用融合按钮
            addLog('success', '基础特征提取完成！请点击"特征融合"按钮继续。');
            document.getElementById('btn-fusion').disabled = false;
        }
        
    } catch (error) {
        addLog('error', `特征提取失败: ${error.message}`);
        alert('特征提取失败: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'inline';
        btn.querySelector('.btn-loading').style.display = 'none';
    }
}

async function callFeatureAPI(url, source, apiKey = null, modelProvider = null) {
    const body = { data_from: source };
    if (apiKey) body.api_key = apiKey;
    if (modelProvider) body.model_provider = modelProvider;
    
    const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    });
    
    const data = await response.json();
    if (!data.success) {
        throw new Error(data.error);
    }
    return data;
}

async function handleFeatureFusion() {
    const btn = document.getElementById('btn-fusion');
    const source = document.getElementById('dataset-source').value;
    
    btn.disabled = true;
    btn.querySelector('.btn-text').style.display = 'none';
    btn.querySelector('.btn-loading').style.display = 'flex';
    
    try {
        addLog('info', '开始特征融合...');
        updateStageStatus('fusion', 'running', '融合中...');
        
        await callFeatureAPI('/api/train/extract_fusion', source);
        
        updateStageStatus('fusion', 'completed', '完成');
        addLog('success', '特征融合完成！现在可以点击"开始训练"按钮了。');
        
        // 融合完成后，启用训练按钮
        document.getElementById('btn-train').disabled = false;
        
    } catch (error) {
        addLog('error', `特征融合失败: ${error.message}`);
        updateStageStatus('fusion', 'error', '失败');
        alert('特征融合失败: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'inline';
        btn.querySelector('.btn-loading').style.display = 'none';
    }
}

// ==================== 训练按钮 ====================
function initTrainButtons() {
    document.getElementById('btn-train').addEventListener('click', handleStartTraining);
    document.getElementById('btn-pause').addEventListener('click', handlePauseTraining);
    document.getElementById('btn-stop').addEventListener('click', handleStopTraining);
    document.getElementById('btn-export').addEventListener('click', handleExportModel);
}

async function handleStartTraining() {
    const btn = document.getElementById('btn-train');
    const source = document.getElementById('dataset-source').value;
    const epochs = document.getElementById('epochs').value;
    const batchSize = document.getElementById('batch-size').value;
    const learningRate = document.getElementById('learning-rate').value;
    const dropout = document.getElementById('dropout').value;
    
    btn.disabled = true;
    btn.querySelector('.btn-text').style.display = 'none';
    btn.querySelector('.btn-loading').style.display = 'flex';
    
    // 显示训练进度和监控
    document.getElementById('train-progress').style.display = 'block';
    document.getElementById('monitor').style.display = 'block';
    
    // 重置状态
    resetTrainingState();
    updateStageStatus('training', 'running', '准备中...');
    
    try {
        addLog('info', '开始训练模型...');
        addLog('info', `训练参数: epochs=${epochs}, batch_size=${batchSize}, lr=${learningRate}, dropout=${dropout}`);
        
        trainingStartTime = Date.now();
        trainingTimer = setInterval(updateTrainingTime, 1000);
        
        // 启用暂停和停止按钮
        document.getElementById('btn-pause').disabled = false;
        document.getElementById('btn-stop').disabled = false;
        
        const response = await fetch('/api/train/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                data_from: source,
                epochs: parseInt(epochs),
                batch_size: parseInt(batchSize),
                learning_rate: parseFloat(learningRate),
                dropout: parseFloat(dropout)
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addLog('success', '训练完成！');
            updateStageStatus('training', 'completed', '完成');
            document.getElementById('stat-model-status').textContent = '已训练';
            document.getElementById('btn-export').disabled = false;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        addLog('error', `训练失败: ${error.message}`);
        updateStageStatus('training', 'error', '失败');
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'inline';
        btn.querySelector('.btn-loading').style.display = 'none';
        document.getElementById('btn-pause').disabled = true;
        document.getElementById('btn-stop').disabled = true;
        if (trainingTimer) {
            clearInterval(trainingTimer);
        }
    }
}

function handlePauseTraining() {
    // TODO: 实现暂停功能
    addLog('warning', '暂停训练功能待实现');
}

function handleStopTraining() {
    if (confirm('确定要停止训练吗？')) {
        trainingSocket.emit('stop_training');
        addLog('warning', '正在停止训练...');
    }
}

async function handleExportModel() {
    try {
        const response = await fetch('/api/train/export_model');
        const blob = await response.blob();
        
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'best_model.pth';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        addLog('success', '模型导出成功！');
    } catch (error) {
        addLog('error', `模型导出失败: ${error.message}`);
    }
}

// ==================== 训练进度更新 ====================
function updateTrainingProgress(data) {
    if (data.stage) {
        updateStageStatus(data.stage, data.status, data.message);
    }
    
    if (data.epoch !== undefined) {
        document.getElementById('stat-epoch').textContent = `${data.epoch} / ${data.total_epochs}`;
    }
    
    if (data.loss !== undefined) {
        document.getElementById('stat-loss').textContent = data.loss.toFixed(4);
    }
}

function updateTrainingMetrics(data) {
    // 更新图表
    if (lossChart && data.loss !== undefined) {
        lossChart.data.labels.push(data.epoch);
        lossChart.data.datasets[0].data.push(data.loss);
        lossChart.update();
    }
    
    if (accuracyChart && data.accuracy !== undefined) {
        accuracyChart.data.labels.push(data.epoch);
        accuracyChart.data.datasets[0].data.push(data.accuracy);
        accuracyChart.update();
    }
    
    // 更新指标
    if (data.best_accuracy) {
        document.getElementById('metric-accuracy').textContent = (data.best_accuracy * 100).toFixed(1) + '%';
    }
    if (data.best_precision) {
        document.getElementById('metric-precision').textContent = data.best_precision.toFixed(4);
    }
    if (data.best_recall) {
        document.getElementById('metric-recall').textContent = data.best_recall.toFixed(4);
    }
    if (data.best_f1) {
        document.getElementById('metric-f1').textContent = data.best_f1.toFixed(4);
    }
    if (data.best_epoch) {
        document.getElementById('best-epoch-badge').textContent = `Epoch ${data.best_epoch}`;
    }
}

// ==================== 阶段状态更新 ====================
function updateStageStatus(stageId, status, message) {
    const stage = document.getElementById(`stage-${stageId}`);
    if (!stage) return;
    
    const statusEl = stage.querySelector('.stage-status');
    
    // 移除所有状态类
    stage.classList.remove('active', 'completed', 'error');
    statusEl.classList.remove('pending', 'running', 'completed', 'error');
    
    // 添加新状态类
    switch(status) {
        case 'running':
            stage.classList.add('active');
            statusEl.classList.add('running');
            statusEl.textContent = message || '进行中';
            break;
        case 'completed':
            stage.classList.add('completed');
            statusEl.classList.add('completed');
            statusEl.textContent = message || '完成';
            break;
        case 'error':
            stage.classList.add('error');
            statusEl.classList.add('error');
            statusEl.textContent = message || '失败';
            break;
        default:
            statusEl.classList.add('pending');
            statusEl.textContent = '等待中';
    }
}

// ==================== 日志管理 ====================
function addLog(level, message) {
    const container = document.getElementById('log-container');
    const entry = document.createElement('div');
    entry.className = `log-entry ${level}`;
    
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${message}`;
    
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

// ==================== 图表初始化 ====================
function initCharts() {
    // 损失曲线
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '训练损失',
                data: [],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: { display: true, text: 'Loss' }
                },
                x: {
                    title: { display: true, text: 'Epoch' }
                }
            }
        }
    });
    
    // 准确率曲线
    const accCtx = document.getElementById('accuracy-chart').getContext('2d');
    accuracyChart = new Chart(accCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '测试准确率',
                data: [],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true }
            },
            scales: {
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Accuracy' }
                },
                x: {
                    title: { display: true, text: 'Epoch' }
                }
            }
        }
    });
}

// ==================== 训练时间更新 ====================
function updateTrainingTime() {
    if (!trainingStartTime) return;
    
    const elapsed = Date.now() - trainingStartTime;
    const hours = Math.floor(elapsed / 3600000);
    const minutes = Math.floor((elapsed % 3600000) / 60000);
    const seconds = Math.floor((elapsed % 60000) / 1000);
    
    const timeStr = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    document.getElementById('stat-time').textContent = timeStr;
}

// ==================== 重置训练状态 ====================
function resetTrainingState() {
    // 重置阶段状态
    ['prepare', 'clip', 'bert', 'vgg', 'reason', 'fusion', 'training'].forEach(id => {
        updateStageStatus(id, 'pending', '等待中');
    });
    
    // 重置日志
    document.getElementById('log-container').innerHTML = '<div class="log-entry info">等待开始训练...</div>';
    
    // 重置图表
    if (lossChart) {
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.update();
    }
    if (accuracyChart) {
        accuracyChart.data.labels = [];
        accuracyChart.data.datasets[0].data = [];
        accuracyChart.update();
    }
    
    // 重置指标
    document.getElementById('metric-accuracy').textContent = '-';
    document.getElementById('metric-precision').textContent = '-';
    document.getElementById('metric-recall').textContent = '-';
    document.getElementById('metric-f1').textContent = '-';
    document.getElementById('best-epoch-badge').textContent = '-';
    
    // 重置统计
    document.getElementById('stat-epoch').textContent = '0 / 100';
    document.getElementById('stat-loss').textContent = '-';
    document.getElementById('stat-time').textContent = '00:00:00';
    
    trainingStartTime = null;
}

// ==================== 系统状态检查 ====================
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.device) {
            document.getElementById('stat-device').textContent = data.device;
        }
        
        if (data.model_loaded) {
            document.getElementById('stat-model-status').textContent = '已加载';
        }
    } catch (error) {
        console.error('系统状态检查失败:', error);
    }
}
