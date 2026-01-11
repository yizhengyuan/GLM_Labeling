// ============ Tab 切换 ============
function switchTab(mode) {
    currentMode = mode;
    
    // 更新 tab 按钮状态
    const btns = document.querySelectorAll('.mode-btn');
    const videoInputs = document.getElementById('video-inputs');
    const imagesInputs = document.getElementById('images-inputs');
    const startBtn = document.getElementById('start-btn');
    
    if (mode === 'video') {
        btns[0].classList.add('active');
        btns[1].classList.remove('active');
        
        videoInputs.classList.remove('hidden');
        imagesInputs.classList.add('hidden');
        
        startBtn.textContent = '🚀 创建并启动流水线';
        loadVideos();
    } else {
        btns[0].classList.remove('active');
        btns[1].classList.add('active');
        
        videoInputs.classList.add('hidden');
        imagesInputs.classList.remove('hidden');
        
        startBtn.textContent = '🚀 开始标注任务';
    }
}
// 将 switchTab 暴露给全局
window.switchTab = switchTab;

// ============ 统一处理开始任务 ============
async function handleStart() {
    if (currentMode === 'video') {
        await createVideoTask();
    } else {
        await createTask();
    }
}
window.handleStart = handleStart;

// ============ 视频任务 ============
async function loadVideos() {
    try {
        const data = await apiCall('GET', '/api/videos');
        const select = document.getElementById('video-name');
        
        if (select) {
            select.innerHTML = '<option value="">选择视频源...</option>' + 
                data.videos.map(v => {
                    const name = typeof v === 'object' ? v.name : v;
                    const label = typeof v === 'object' ? `${v.name} (${v.size_str})` : v;
                    return `<option value="${name}">${label}</option>`;
                }).join('');
        }
            
    } catch (e) {
        addLog(`❌ 加载视频列表失败: ${e.message}`, 'error');
    }
}

async function createVideoTask() {
    const videoName = document.getElementById('video-name').value;
    const outputName = document.getElementById('output-name').value.trim();
    const fps = parseInt(document.getElementById('fps').value) || 3;
    const useRag = document.getElementById('video-use-rag').checked;
    
    if (!videoName || !outputName) {
        alert('请选择视频并输入输出名称');
        return;
    }
    
    try {
        const data = await apiCall('POST', '/api/tasks/video', {
            video_name: videoName,
            output_name: outputName,
            fps: fps,
            use_rag: useRag
        });
        
        state.currentTaskId = data.task.id;
        addLog(`🎬 创建视频任务: ${videoName} → ${outputName}`, 'success');
        
        // 清空表单
        document.getElementById('video-name').value = '';
        document.getElementById('output-name').value = '';
        document.getElementById('fps').value = '3';
        document.getElementById('video-use-rag').checked = false;
        
        loadTasks();
    } catch (e) {
        alert('创建任务失败: ' + e.message);
        addLog(`❌ 创建任务失败: ${e.message}`, 'error');
    }
}

// ============ 图片任务 ============
async function createTask() {
    const prefix = document.getElementById('prefix').value.trim();
    const limit = document.getElementById('limit').value ? parseInt(document.getElementById('limit').value) : null;
    const useRag = document.getElementById('use-rag').checked;
    
    if (!prefix) {
        alert('请输入图片前缀');
        return;
    }
    
    try {
        const data = await apiCall('POST', '/api/tasks', { prefix, limit, use_rag: useRag });
        state.currentTaskId = data.task.id;
        addLog(`📋 创建任务: ${prefix} (${data.task.total_frames} 张图片)`, 'success');
        
        // 清空表单
        document.getElementById('prefix').value = '';
        document.getElementById('limit').value = '';
        document.getElementById('use-rag').checked = false;
        
        // 重新加载任务列表
        loadTasks();
    } catch (e) {
        alert('创建任务失败: ' + e.message);
        addLog(`❌ 创建任务失败: ${e.message}`, 'error');
    }
}





