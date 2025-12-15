# 🖥️ 终端长时间运行脚本指南

> 本文档介绍如何在 Mac 终端中稳定运行长时间标注脚本，避免因终端关闭或网络波动导致任务中断。

---

## 📋 基础准备

### 1. 进入项目目录
```bash
cd ~/Desktop/GLM_Labeling
```

### 2. 激活虚拟环境
```bash
source venv/bin/activate
```

### 3. 设置 API Key
```bash
export ZAI_API_KEY="your_api_key_here"
```

### 4. 验证环境
```bash
python3 -c "from zai import ZaiClient; print('✅ SDK 正常')"
```

---

## 🚀 运行脚本

### 基础模式（前台运行）
```bash
# D2 数据集 + RAG 模式
python3 auto_labeling_parallel.py --prefix D2 --rag --workers 5

# 限制数量（测试用）
python3 auto_labeling_parallel.py --prefix D2 --rag --limit 20

# 不启用 RAG（只做基础检测）
python3 auto_labeling_parallel.py --prefix D2 --workers 5
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--prefix` | 图片前缀 | D1, D2, D3 |
| `--rag` | 启用 RAG 细粒度分类 | 可选 |
| `--workers` | 并行线程数 | 默认 5 |
| `--limit` | 限制处理数量 | 测试用 |
| `--images-dir` | 图片目录 | 默认 test_images/extracted_frames |

---

## 🔒 后台稳定运行（推荐）

### 方法一：nohup（最简单）

**特点**：终端关闭后任务继续运行

```bash
# 后台运行，输出重定向到日志文件
nohup python3 auto_labeling_parallel.py --prefix D2 --rag --workers 5 > d2_log.txt 2>&1 &

# 查看实时日志
tail -f d2_log.txt

# 停止监控日志（不影响任务）
Ctrl + C

# 查看后台进程
ps aux | grep python

# 杀死进程（如需停止）
kill -9 <PID>
```

### 方法二：screen（可断开重连）

**特点**：可以断开终端后重新连接查看进度

```bash
# 安装 screen（如果没有）
brew install screen

# 创建新会话
screen -S labeling

# 在 screen 里正常运行
source venv/bin/activate
export ZAI_API_KEY="your_key"
python3 auto_labeling_parallel.py --prefix D2 --rag --workers 5

# 断开会话（任务继续运行）
Ctrl + A, 然后按 D

# 重新连接
screen -r labeling

# 列出所有会话
screen -ls

# 结束会话
exit
```

### 方法三：tmux（功能最强）

**特点**：分屏、多窗口、可保存布局

```bash
# 安装 tmux
brew install tmux

# 创建新会话
tmux new -s labeling

# 运行命令
source venv/bin/activate
export ZAI_API_KEY="your_key"
python3 auto_labeling_parallel.py --prefix D2 --rag --workers 5

# 断开会话
Ctrl + B, 然后按 D

# 重新连接
tmux attach -t labeling

# 列出所有会话
tmux ls

# 杀死会话
tmux kill-session -t labeling
```

---

## 📊 完整工作流示例

### 示例 1：处理 D3 数据集

```bash
# 1. 打开终端
# 2. 进入项目目录
cd ~/Desktop/GLM_Labeling

# 3. 激活虚拟环境
source venv/bin/activate

# 4. 设置 API Key
export ZAI_API_KEY="your_api_key"

# 5. 后台运行（推荐）
nohup python3 auto_labeling_parallel.py --prefix D3 --rag --workers 5 > d3_log.txt 2>&1 &

# 6. 监控进度
tail -f d3_log.txt

# 7. 任务完成后查看输出
ls output/d3_annotations_rag/
```

### 示例 2：一行命令搞定

```bash
cd ~/Desktop/GLM_Labeling && source venv/bin/activate && export ZAI_API_KEY="your_key" && python3 auto_labeling_parallel.py --prefix D2 --rag --workers 5
```

---

## ⚠️ 常见问题

### 1. 终端关闭后任务终止
**解决**：使用 `nohup` 或 `screen` 后台运行

### 2. API 超时或 429 错误
**解决**：脚本已内置指数退避重试（2s, 4s, 6s）

### 3. 磁盘空间不足
**解决**：脚本已内置临时文件自动清理

### 4. 查看进程状态
```bash
ps aux | grep python
```

### 5. 强制停止任务
```bash
# 找到 PID
ps aux | grep auto_labeling

# 杀死进程
kill -9 <PID>
```

---

## 📁 输出文件位置

```
output/
├── d2_annotations_rag/     # JSON 标注文件
└── d2_visualized_rag/      # 可视化图片
```

---

## 🔗 相关文档

- [RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md) - RAG 技术方案
- [D2_annotation_report.md](D2_annotation_report.md) - D2 标注报告
- [README.md](README.md) - 项目总览

---

<p align="center">
  <b>🚀 Happy Labeling!</b>
</p>
