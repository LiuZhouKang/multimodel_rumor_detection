// ==================== 全局变量 ====================
let selectedImage = null;
let selectedImageBase64 = null;
let socket = null;

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', function() {
    initSocketIO();
    initNavigation();
    initTextCounter();
    initImageUpload();
    initButtons();
    checkHealth();
});

// ==================== Socket.IO 连接 ====================
function initSocketIO() {
    socket = io();
    
    socket.on('connect', function() {
        console.log('已连接到服务器');
    });
    
    socket.on('progress', function(data) {
        updateProgress(data.percent, data.step);
    });
    
    socket.on('disconnect', function() {
        console.log('与服务器断开连接');
    });
}

// ==================== 导航切换 ====================
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            // 只有锚点链接(以#开头)才阻止默认行为
            if (!href.startsWith('#')) {
                // 非锚点链接(如/train)，允许正常跳转
                return;
            }
            
            e.preventDefault();
            
            // 更新导航激活状态
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // 切换显示对应区块
            const targetId = href.substring(1);
            sections.forEach(section => {
                section.classList.remove('active-section');
                if (section.id === targetId) {
                    section.classList.add('active-section');
                }
            });
        });
    });
}

// ==================== 文本计数器 ====================
function initTextCounter() {
    const textarea = document.getElementById('news-text');
    const counter = document.getElementById('char-count');
    
    textarea.addEventListener('input', function() {
        counter.textContent = this.value.length;
        updateDetectButton();
    });
}

// ==================== 图片上传 ====================
function initImageUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('image-upload');
    const preview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const removeBtn = document.getElementById('remove-image');
    const uploadContent = uploadArea.querySelector('.upload-content');
    
    // 文件选择
    fileInput.addEventListener('change', function(e) {
        handleFileSelect(e.target.files[0]);
    });
    
    // 拖拽上传
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.borderColor = '#2563eb';
        this.style.background = '#eff6ff';
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.style.borderColor = '#e5e7eb';
        this.style.background = '#f3f4f6';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.borderColor = '#e5e7eb';
        this.style.background = '#f3f4f6';
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    // 移除图片
    removeBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        removeImage();
    });
}

function handleFileSelect(file) {
    if (!file) return;
    
    // 验证文件类型
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        alert('请上传有效的图像文件（JPG, PNG, GIF, BMP, WEBP）');
        return;
    }
    
    // 验证文件大小（10MB）
    if (file.size > 10 * 1024 * 1024) {
        alert('图像文件大小不能超过10MB');
        return;
    }
    
    selectedImage = file;
    
    // 预览图像
    const reader = new FileReader();
    reader.onload = function(e) {
        selectedImageBase64 = e.target.result;
        document.getElementById('preview-img').src = selectedImageBase64;
        document.getElementById('image-preview').classList.add('show');
        document.getElementById('upload-area').classList.add('has-image');
        document.getElementById('upload-area').querySelector('.upload-content').style.display = 'none';
        // 在文件读取完成后更新按钮状态
        updateDetectButton();
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedImage = null;
    selectedImageBase64 = null;
    document.getElementById('image-upload').value = '';
    document.getElementById('image-preview').classList.remove('show');
    document.getElementById('upload-area').classList.remove('has-image');
    document.getElementById('upload-area').querySelector('.upload-content').style.display = 'block';
    updateDetectButton();
}

// ==================== 按钮控制 ====================
function initButtons() {
    const detectBtn = document.getElementById('btn-detect');
    const clearBtn = document.getElementById('btn-clear');
    const newsText = document.getElementById('news-text');
    
    detectBtn.addEventListener('click', handleDetect);
    clearBtn.addEventListener('click', clearForm);
    
    // 监听文本输入，实时更新按钮状态
    newsText.addEventListener('input', updateDetectButton);
}

function updateDetectButton() {
    const text = document.getElementById('news-text').value.trim();
    const btn = document.getElementById('btn-detect');
    btn.disabled = !(text.length > 0 && selectedImageBase64);
}

// ==================== 检测功能 ====================
async function handleDetect() {
    const text = document.getElementById('news-text').value.trim();
    const btn = document.getElementById('btn-detect');
    
    if (!text || !selectedImageBase64) {
        alert('请输入文本并上传图片');
        return;
    }
    
    // 更新按钮状态
    btn.disabled = true;
    btn.querySelector('.btn-text').style.display = 'none';
    btn.querySelector('.btn-loading').style.display = 'flex';
    
    // 显示进度条
    document.getElementById('progress-section').style.display = 'block';
    updateProgress(0, '准备中...');
    
    // 隐藏结果
    document.getElementById('result-section').style.display = 'none';
    
    try {
        // 获取大模型配置
        const modelProvider = document.getElementById('model-provider').value;
        const apiKey = document.getElementById('api-key').value.trim();
        const enableImageEnhance = document.getElementById('enable-image-enhance').checked;
        
        // 发送预测请求
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                image: selectedImageBase64,
                model_provider: modelProvider,
                api_key: apiKey || undefined,
                enable_image_enhance: enableImageEnhance
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            updateProgress(100, '分析完成！');
            setTimeout(() => {
                displayResult(data.data);
            }, 500);
        } else {
            throw new Error(data.error || '检测失败');
        }
    } catch (error) {
        console.error('检测失败:', error);
        alert('检测失败：' + error.message);
        document.getElementById('progress-section').style.display = 'none';
    } finally {
        // 恢复按钮状态
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'inline';
        btn.querySelector('.btn-loading').style.display = 'none';
        updateDetectButton();
    }
}

// ==================== 进度更新 ====================
function updateProgress(percent, text) {
    document.getElementById('progress-fill').style.width = percent + '%';
    document.getElementById('progress-text').textContent = text;
}

// ==================== 结果展示 ====================
function displayResult(data) {
    const resultSection = document.getElementById('result-section');
    const resultHeader = document.getElementById('result-header');
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    
    // 设置结果类型样式
    if (data.prediction === '谣言') {
        resultHeader.className = 'result-header rumor';
        resultIcon.textContent = '⚠️';
    } else {
        resultHeader.className = 'result-header safe';
        resultIcon.textContent = '✅';
    }
    
    // 设置标题
    resultTitle.textContent = data.prediction;
    
    // 设置统计信息
    document.getElementById('stat-confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('stat-false-prob').textContent = (data.probabilities['非谣言'] * 100).toFixed(1) + '%';
    document.getElementById('stat-true-prob').textContent = (data.probabilities['谣言'] * 100).toFixed(1) + '%';
    
    // 设置分析说明
    document.getElementById('result-reasoning-text').textContent = data.reasoning;
    
    // 设置大模型推理信息
    const largeModelSection = document.getElementById('result-large-model');
    if (data.large_model_reasoning) {
        largeModelSection.style.display = 'block';
        document.getElementById('reason-text').textContent = data.large_model_reasoning.text;
        document.getElementById('reason-image').textContent = data.large_model_reasoning.image;
    } else {
        largeModelSection.style.display = 'none';
    }
    
    // 设置图表
    const falsePercent = data.probabilities['非谣言'] * 100;
    const truePercent = data.probabilities['谣言'] * 100;
    
    document.getElementById('bar-false-fill').style.width = falsePercent + '%';
    document.getElementById('bar-false-value').textContent = falsePercent.toFixed(1) + '%';
    
    document.getElementById('bar-true-fill').style.width = truePercent + '%';
    document.getElementById('bar-true-value').textContent = truePercent.toFixed(1) + '%';
    
    // 设置特征信息
    if (data.feature_stats) {
        document.getElementById('feature-bert').textContent = data.feature_stats.bert_dim;
        document.getElementById('feature-vgg').textContent = data.feature_stats.vgg_dim;
        document.getElementById('feature-mixed').textContent = data.feature_stats.mixed_dim;
    }
    document.getElementById('feature-similarity').textContent = (data.simulation_score * 100).toFixed(1) + '%';
    
    // 显示结果区块
    resultSection.style.display = 'block';
    
    // 平滑滚动到结果
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ==================== 清空表单 ====================
function clearForm() {
    document.getElementById('news-text').value = '';
    document.getElementById('char-count').textContent = '0';
    removeImage();
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('result-section').style.display = 'none';
    updateProgress(0, '准备中...');
}

// ==================== 健康检查 ====================
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.status === 'ok') {
            console.log('系统状态正常');
            console.log('模型已加载:', data.model_loaded);
            console.log('运行设备:', data.device);
            
            if (!data.model_loaded) {
                console.warn('警告：模型未加载，部分功能可能不可用');
            }
        }
    } catch (error) {
        console.error('健康检查失败:', error);
    }
}

// ==================== 工具函数 ====================
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
