#!/bin/bash

# TriModalFusion Environment Setup Script
# 自动化解决部署过程中的常见问题

set -e  # Exit on any error

echo "开始设置 TriModalFusion 环境..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python版本
check_python() {
    echo -e "${YELLOW}检查Python版本...${NC}"
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo -e "${GREEN}找到 python3${NC}"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        echo -e "${GREEN}✓ 找到 python${NC}"
    else
        echo -e "${RED}❌ 未找到Python，请先安装Python 3.8+${NC}"
        exit 1
    fi
    
    # 检查Python版本
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Python版本: $PYTHON_VERSION"
    
    if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 0 ]]; then
        echo -e "${RED}❌ Python版本需要3.8或更高，当前版本: $PYTHON_VERSION${NC}"
        exit 1
    fi
}

# 检查并初始化conda
setup_conda() {
    echo -e "${YELLOW}检查conda环境...${NC}"
    
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}❌ 未找到conda，请先安装Anaconda或Miniconda${NC}"
        echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # 初始化conda（如果未初始化）
    if [[ ! -f ~/.bashrc ]] || ! grep -q "conda initialize" ~/.bashrc; then
        echo -e "${YELLOW}初始化conda...${NC}"
        conda init bash
        echo -e "${YELLOW}请重新启动终端并重新运行此脚本${NC}"
        exit 0
    fi
    
    # 确保conda已激活
    source ~/.bashrc 2>/dev/null || true
    
    echo -e "${GREEN}✓ conda已就绪${NC}"
}

# 创建或激活conda环境
setup_conda_env() {
    local env_name="TriModalFusion"
    
    echo -e "${YELLOW}设置conda环境: $env_name${NC}"
    
    # 检查环境是否已存在
    if conda env list | grep -q "^$env_name "; then
        echo -e "${YELLOW}环境 $env_name 已存在，激活现有环境${NC}"
        source activate $env_name || conda activate $env_name
    else
        echo -e "${YELLOW}创建新的conda环境: $env_name${NC}"
        conda create -n $env_name python=3.9 -y
        source activate $env_name || conda activate $env_name
    fi
    
    echo -e "${GREEN}✓ conda环境已激活${NC}"
}

# 升级基础包
upgrade_base_packages() {
    echo -e "${YELLOW}升级基础包...${NC}"
    
    pip install --upgrade pip setuptools wheel
    
    # 解决已知的版本冲突
    pip install --upgrade \
        "setuptools>=50.0" \
        "requests>=2.30.0" \
        "fsspec>=2023.1.0,<=2024.6.1" \
        "wandb>=0.15.0,<=0.19.0"
    
    echo -e "${GREEN}✓ 基础包升级完成${NC}"
}

# 安装PyTorch（优先使用conda）
install_pytorch() {
    echo -e "${YELLOW}安装PyTorch...${NC}"
    
    # 检查是否已安装
    if $PYTHON_CMD -c "import torch; print('PyTorch已安装，版本:', torch.__version__)" 2>/dev/null; then
        echo -e "${GREEN}✓ PyTorch已安装${NC}"
        return
    fi
    
    echo "安装PyTorch（使用conda）..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    
    # 验证安装
    if $PYTHON_CMD -c "import torch; print('PyTorch安装成功，版本:', torch.__version__)"; then
        echo -e "${GREEN}✓ PyTorch安装成功${NC}"
    else
        echo -e "${RED}❌ PyTorch安装失败${NC}"
        exit 1
    fi
}

# 安装项目依赖
install_dependencies() {
    echo -e "${YELLOW}安装项目依赖...${NC}"
    
    if [[ -f "requirements.txt" ]]; then
        # 分批安装以避免冲突
        echo "安装深度学习框架..."
        pip install lightning transformers timm
        
        echo "安装计算机视觉库..."
        pip install opencv-python Pillow albumentations mediapipe
        
        echo "安装音频处理库..."
        pip install librosa soundfile audioread
        
        echo "安装科学计算库..."
        pip install numpy scipy scikit-learn
        
        echo "安装其他依赖..."
        pip install -r requirements.txt --ignore-installed
        
        echo -e "${GREEN}✓ 依赖安装完成${NC}"
    else
        echo -e "${RED}❌ 未找到 requirements.txt 文件${NC}"
        exit 1
    fi
}

# 修复导入路径问题
fix_import_paths() {
    echo -e "${YELLOW}修复导入路径...${NC}"
    
    # 创建根目录__init__.py（如果不存在）
    if [[ ! -f "__init__.py" ]]; then
        echo "# TriModalFusion package" > __init__.py
        echo -e "${GREEN}✓ 创建根目录 __init__.py${NC}"
    fi
    
    # 设置PYTHONPATH
    export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"
    
    echo -e "${GREEN}✓ 导入路径已修复${NC}"
}

# 运行测试
run_tests() {
    echo -e "${YELLOW}运行基本测试...${NC}"
    
    # 测试基本导入
    echo "测试基本模块导入..."
    $PYTHON_CMD -c "
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

try:
    import torch
    print('✓ PyTorch导入成功')
    
    import src
    print('✓ src包导入成功')
    
    from src.utils.config import load_config
    print('✓ 配置模块导入成功')
    
    print('✅ 所有基本模块导入测试通过')
except ImportError as e:
    print(f'❌ 导入错误: {e}')
    exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ 基本测试通过${NC}"
    else
        echo -e "${RED}❌ 基本测试失败${NC}"
        exit 1
    fi
}

# 创建激活脚本
create_activation_script() {
    echo -e "${YELLOW}创建环境激活脚本...${NC}"
    
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# TriModalFusion 环境激活脚本

# 激活conda环境
if command -v conda &> /dev/null; then
    source activate TriModalFusion 2>/dev/null || conda activate TriModalFusion
    echo "✓ conda环境已激活"
else
    echo "❌ 未找到conda"
fi

# 设置PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"
echo "✓ PYTHONPATH已设置"

# 显示环境信息
echo "🐍 Python路径: $(which python)"
echo "📦 当前环境: $CONDA_DEFAULT_ENV"
echo "🚀 TriModalFusion环境已就绪！"

# 可选：运行示例
echo ""
echo "运行基本示例："
echo "  python examples/basic_usage.py"
echo ""
echo "或者使用模块方式："
echo "  python -m examples.basic_usage"
EOF
    
    chmod +x activate_env.sh
    echo -e "${GREEN}✓ 激活脚本已创建: activate_env.sh${NC}"
}

# 主函数
main() {
    echo "=================================================="
    echo "     TriModalFusion 自动环境设置脚本"
    echo "=================================================="
    echo ""
    
    # 检查是否在项目根目录
    if [[ ! -f "requirements.txt" ]] || [[ ! -d "src" ]]; then
        echo -e "${RED}❌ 请在TriModalFusion项目根目录运行此脚本${NC}"
        exit 1
    fi
    
    check_python
    setup_conda
    setup_conda_env
    upgrade_base_packages
    install_pytorch
    install_dependencies
    fix_import_paths
    run_tests
    create_activation_script
    
    echo ""
    echo "=================================================="
    echo -e "${GREEN}🎉 环境设置完成！${NC}"
    echo "=================================================="
    echo ""
    echo "下次使用时，运行以下命令激活环境："
    echo -e "${YELLOW}  source activate_env.sh${NC}"
    echo ""
    echo "然后运行示例："
    echo -e "${YELLOW}  python examples/basic_usage.py${NC}"
    echo ""
    echo -e "${GREEN}祝您使用愉快！${NC}"
}

# 运行主函数
main "$@"