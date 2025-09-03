#!/bin/bash

# TriModalFusion Conda 环境安装脚本
# 使用 environment.yml 文件创建完整的 conda 环境

set -e  # Exit on any error

echo "🚀 使用 Conda 安装 TriModalFusion 环境..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 环境配置
ENV_NAME="TriModalFusion"
PYTHON_VERSION="3.9"

# 打印环境信息
print_info() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  TriModalFusion Conda 环境安装${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "环境名称: ${YELLOW}$ENV_NAME${NC}"
    echo -e "Python版本: ${YELLOW}$PYTHON_VERSION${NC}"
    echo -e "配置文件: ${YELLOW}environment.yml${NC}"
    echo ""
}

# 检查conda是否可用
check_conda() {
    echo -e "${YELLOW}检查 conda 安装状态...${NC}"
    
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}❌ 未找到 conda 命令${NC}"
        echo -e "${YELLOW}请先安装 Anaconda 或 Miniconda:${NC}"
        echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        echo "  - Anaconda: https://www.anaconda.com/products/distribution"
        exit 1
    fi
    
    echo -e "${GREEN}✓ conda 已安装${NC}"
    echo "conda 版本: $(conda --version)"
}

# 检查environment.yml文件
check_environment_file() {
    echo -e "${YELLOW}检查环境配置文件...${NC}"
    
    if [[ ! -f "environment.yml" ]]; then
        echo -e "${RED}❌ 未找到 environment.yml 文件${NC}"
        echo "请确保在项目根目录运行此脚本"
        exit 1
    fi
    
    echo -e "${GREEN}✓ 找到 environment.yml${NC}"
}

# 初始化conda（如果需要）
init_conda() {
    echo -e "${YELLOW}检查 conda 初始化状态...${NC}"
    
    # 检查是否已经初始化
    if [[ -f ~/.bashrc ]] && grep -q "conda initialize" ~/.bashrc; then
        echo -e "${GREEN}✓ conda 已初始化${NC}"
    else
        echo -e "${YELLOW}初始化 conda...${NC}"
        conda init bash
        echo -e "${YELLOW}conda 已初始化。请重新启动终端并重新运行此脚本。${NC}"
        exit 0
    fi
    
    # 确保conda base环境可用
    source ~/.bashrc 2>/dev/null || true
}

# 删除现有环境（如果存在）
remove_existing_env() {
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}发现现有环境 '$ENV_NAME'${NC}"
        read -p "是否删除现有环境并重新创建？[y/N]: " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}删除现有环境...${NC}"
            conda env remove -n $ENV_NAME -y
            echo -e "${GREEN}✓ 现有环境已删除${NC}"
        else
            echo -e "${YELLOW}取消安装。${NC}"
            exit 0
        fi
    fi
}

# 创建conda环境
create_environment() {
    echo -e "${YELLOW}从 environment.yml 创建 conda 环境...${NC}"
    echo "这可能需要几分钟时间，请耐心等待..."
    echo ""
    
    # 使用environment.yml创建环境
    conda env create -f environment.yml
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ conda 环境创建成功${NC}"
    else
        echo -e "${RED}❌ conda 环境创建失败${NC}"
        exit 1
    fi
}

# 验证环境
verify_installation() {
    echo -e "${YELLOW}验证环境安装...${NC}"
    
    # 激活环境
    source activate $ENV_NAME 2>/dev/null || conda activate $ENV_NAME
    
    # 验证Python版本
    INSTALLED_PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "安装的Python版本: $INSTALLED_PYTHON_VERSION"
    
    # 验证关键包
    echo -e "${YELLOW}验证关键依赖包...${NC}"
    
    python -c "
import sys
import importlib

packages_to_check = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('torchaudio', 'TorchAudio'),
    ('lightning', 'Lightning'),
    ('transformers', 'Transformers'),
    ('cv2', 'OpenCV'),
    ('mediapipe', 'MediaPipe'),
    ('librosa', 'Librosa'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas')
]

failed_imports = []

for package, name in packages_to_check:
    try:
        module = importlib.import_module(package)
        version = getattr(module, '__version__', 'unknown')
        print(f'✓ {name}: {version}')
    except ImportError:
        print(f'❌ {name}: 导入失败')
        failed_imports.append(name)

if failed_imports:
    print(f'\\n❌ 以下包导入失败: {', '.join(failed_imports)}')
    sys.exit(1)
else:
    print('\\n✅ 所有关键依赖包验证通过')
"
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ 环境验证成功${NC}"
    else
        echo -e "${RED}❌ 环境验证失败${NC}"
        exit 1
    fi
}

# 设置项目路径
setup_project_paths() {
    echo -e "${YELLOW}设置项目路径...${NC}"
    
    # 创建激活脚本
    cat > activate_trimodal.sh << EOF
#!/bin/bash
# TriModalFusion 环境激活脚本

echo "🚀 激活 TriModalFusion 环境..."

# 激活conda环境
source activate $ENV_NAME 2>/dev/null || conda activate $ENV_NAME

# 设置Python路径
export PYTHONPATH="\$(pwd):\$(pwd)/src:\$PYTHONPATH"

# 显示环境信息
echo "✓ 环境已激活: \$CONDA_DEFAULT_ENV"
echo "✓ Python版本: \$(python --version)"
echo "✓ Python路径: \$(which python)"
echo "✓ 工作目录: \$(pwd)"
echo ""
echo "🎉 TriModalFusion 环境已就绪！"
echo ""
echo "运行示例："
echo "  python examples/basic_usage.py"
echo ""
echo "或使用模块方式："
echo "  python -m examples.basic_usage"
EOF
    
    chmod +x activate_trimodal.sh
    echo -e "${GREEN}✓ 激活脚本已创建: activate_trimodal.sh${NC}"
}

# 运行基础测试
run_basic_test() {
    echo -e "${YELLOW}运行基础测试...${NC}"
    
    # 确保环境已激活
    source activate $ENV_NAME 2>/dev/null || conda activate $ENV_NAME
    
    # 设置Python路径
    export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"
    
    # 测试基本导入
    python -c "
import sys
import os

# 添加项目路径
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

print('Python路径已设置')

try:
    # 测试基本导入
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    
    import lightning as L
    print(f'✓ Lightning {L.__version__}')
    
    # 测试项目导入
    from src.utils import load_config
    print('✓ 项目配置模块')
    
    print('\\n🎉 所有测试通过！环境配置成功！')
    
except Exception as e:
    print(f'❌ 测试失败: {e}')
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ 基础测试通过${NC}"
    else
        echo -e "${RED}❌ 基础测试失败${NC}"
        return 1
    fi
}

# 显示使用说明
show_usage_instructions() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}🎉 安装完成！${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    echo -e "${YELLOW}环境使用说明:${NC}"
    echo ""
    echo "1. 激活环境:"
    echo -e "   ${YELLOW}source activate_trimodal.sh${NC}"
    echo ""
    echo "2. 手动激活环境:"
    echo -e "   ${YELLOW}conda activate $ENV_NAME${NC}"
    echo ""
    echo "3. 运行示例:"
    echo -e "   ${YELLOW}python examples/basic_usage.py${NC}"
    echo ""
    echo "4. 查看环境信息:"
    echo -e "   ${YELLOW}conda list${NC}"
    echo ""
    echo "5. 导出环境配置:"
    echo -e "   ${YELLOW}conda env export > my_environment.yml${NC}"
    echo ""
    echo -e "${GREEN}祝您使用愉快！${NC}"
}

# 主函数
main() {
    print_info
    check_conda
    check_environment_file
    init_conda
    remove_existing_env
    create_environment
    verify_installation
    setup_project_paths
    
    if run_basic_test; then
        show_usage_instructions
    else
        echo -e "${YELLOW}⚠️  环境已创建，但基础测试失败。请检查配置。${NC}"
        echo -e "${YELLOW}您仍然可以手动激活环境: conda activate $ENV_NAME${NC}"
    fi
}

# 检查是否在正确目录
if [[ ! -f "environment.yml" ]] && [[ ! -f "requirements.txt" ]]; then
    echo -e "${RED}❌ 请在 TriModalFusion 项目根目录运行此脚本${NC}"
    exit 1
fi

# 运行主函数
main "$@"