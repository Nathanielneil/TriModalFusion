#!/bin/bash

# TriModalFusion 一键部署脚本
# ================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
TriModalFusion 部署脚本

使用方法:
  $0 [OPTIONS] DEPLOYMENT_TYPE

部署类型:
  local       本地开发部署
  docker      单容器Docker部署
  compose     Docker Compose生产部署
  k8s         Kubernetes集群部署

选项:
  -h, --help          显示此帮助信息
  -m, --model PATH    指定模型文件路径
  -c, --config PATH   指定配置文件路径
  -p, --port PORT     指定服务端口 (默认: 8000)
  -g, --gpu           启用GPU支持
  --no-cache          不使用Docker构建缓存
  --dry-run           只显示将要执行的命令，不实际执行

示例:
  $0 local                                # 本地部署
  $0 docker -m models/best_model.pth     # Docker部署，指定模型
  $0 compose -g                          # Docker Compose部署，启用GPU
  $0 k8s --dry-run                       # Kubernetes部署预览

EOF
}

# 默认参数
DEPLOYMENT_TYPE=""
MODEL_PATH=""
CONFIG_PATH=""
PORT="8000"
GPU_ENABLED=false
NO_CACHE=false
DRY_RUN=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ENABLED=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        local|docker|compose|k8s)
            DEPLOYMENT_TYPE="$1"
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查部署类型
if [[ -z "$DEPLOYMENT_TYPE" ]]; then
    log_error "请指定部署类型"
    show_help
    exit 1
fi

# 执行命令函数
execute_command() {
    local cmd="$1"
    local description="$2"
    
    log_info "$description"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  > $cmd"
        return 0
    fi
    
    if eval "$cmd"; then
        log_success "$description 完成"
    else
        log_error "$description 失败"
        exit 1
    fi
}

# 检查系统依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    local required_tools=()
    
    case $DEPLOYMENT_TYPE in
        local)
            required_tools=("python3" "pip")
            ;;
        docker)
            required_tools=("docker")
            ;;
        compose)
            required_tools=("docker" "docker-compose")
            ;;
        k8s)
            required_tools=("kubectl" "helm")
            ;;
    esac
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "缺少依赖工具: $tool"
            exit 1
        fi
    done
    
    # 检查GPU支持
    if [[ "$GPU_ENABLED" == "true" ]]; then
        if [[ "$DEPLOYMENT_TYPE" != "local" ]]; then
            if ! command -v nvidia-docker &> /dev/null && ! docker info 2>/dev/null | grep -q "nvidia"; then
                log_warning "未检测到NVIDIA Docker支持，GPU功能可能无法使用"
            fi
        else
            if ! command -v nvidia-smi &> /dev/null; then
                log_warning "未检测到NVIDIA驱动，GPU功能可能无法使用"
            fi
        fi
    fi
    
    log_success "依赖检查通过"
}

# 验证文件
validate_files() {
    log_info "验证必要文件..."
    
    # 检查模型文件
    if [[ -n "$MODEL_PATH" ]]; then
        if [[ ! -f "$MODEL_PATH" ]]; then
            log_error "模型文件不存在: $MODEL_PATH"
            exit 1
        fi
        log_info "使用模型文件: $MODEL_PATH"
    else
        log_warning "未指定模型文件，将使用默认路径"
    fi
    
    # 检查配置文件
    if [[ -n "$CONFIG_PATH" ]]; then
        if [[ ! -f "$CONFIG_PATH" ]]; then
            log_error "配置文件不存在: $CONFIG_PATH"
            exit 1
        fi
        log_info "使用配置文件: $CONFIG_PATH"
    else
        log_info "使用默认配置文件"
    fi
}

# 本地部署
deploy_local() {
    log_info "开始本地部署..."
    
    # 创建虚拟环境
    execute_command "python3 -m venv venv" "创建虚拟环境"
    execute_command "source venv/bin/activate" "激活虚拟环境"
    
    # 安装依赖
    execute_command "pip install --upgrade pip" "升级pip"
    execute_command "pip install -r requirements.txt" "安装基础依赖"
    execute_command "pip install -r requirements-prod.txt" "安装生产依赖"
    execute_command "pip install -e ." "安装项目包"
    
    # 准备启动命令
    local start_cmd="python deployment/serve.py --host 0.0.0.0 --port $PORT"
    
    if [[ -n "$MODEL_PATH" ]]; then
        export TRIMODAL_MODEL_PATH="$MODEL_PATH"
    fi
    
    if [[ -n "$CONFIG_PATH" ]]; then
        export TRIMODAL_CONFIG_PATH="$CONFIG_PATH"
    fi
    
    if [[ "$GPU_ENABLED" == "true" ]]; then
        export CUDA_VISIBLE_DEVICES="0"
    fi
    
    log_success "本地部署准备完成"
    log_info "启动服务: $start_cmd"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        exec $start_cmd
    fi
}

# Docker部署
deploy_docker() {
    log_info "开始Docker部署..."
    
    local image_name="trimodal-fusion:latest"
    local container_name="trimodal-api"
    
    # 构建镜像
    local build_cmd="docker build -t $image_name"
    if [[ "$NO_CACHE" == "true" ]]; then
        build_cmd="$build_cmd --no-cache"
    fi
    build_cmd="$build_cmd -f deployment/docker/Dockerfile ."
    
    execute_command "$build_cmd" "构建Docker镜像"
    
    # 停止并删除已有容器
    execute_command "docker stop $container_name 2>/dev/null || true" "停止已有容器"
    execute_command "docker rm $container_name 2>/dev/null || true" "删除已有容器"
    
    # 准备运行命令
    local run_cmd="docker run -d --name $container_name"
    
    if [[ "$GPU_ENABLED" == "true" ]]; then
        run_cmd="$run_cmd --gpus all"
    fi
    
    run_cmd="$run_cmd -p $PORT:8000"
    
    # 挂载卷
    if [[ -n "$MODEL_PATH" ]]; then
        local model_dir=$(dirname "$MODEL_PATH")
        run_cmd="$run_cmd -v $model_dir:/app/models:ro"
    fi
    
    if [[ -n "$CONFIG_PATH" ]]; then
        local config_dir=$(dirname "$CONFIG_PATH")
        run_cmd="$run_cmd -v $config_dir:/app/configs:ro"
    fi
    
    run_cmd="$run_cmd $image_name"
    
    execute_command "$run_cmd" "启动Docker容器"
    
    log_success "Docker部署完成"
    log_info "服务地址: http://localhost:$PORT"
    log_info "API文档: http://localhost:$PORT/docs"
}

# Docker Compose部署
deploy_compose() {
    log_info "开始Docker Compose部署..."
    
    cd deployment/docker
    
    # 设置环境变量
    if [[ -n "$MODEL_PATH" ]]; then
        export TRIMODAL_MODEL_PATH="$MODEL_PATH"
    fi
    
    if [[ -n "$CONFIG_PATH" ]]; then
        export TRIMODAL_CONFIG_PATH="$CONFIG_PATH"
    fi
    
    if [[ "$GPU_ENABLED" == "true" ]]; then
        export COMPOSE_FILE="docker-compose.yml:docker-compose.gpu.yml"
    fi
    
    # 构建镜像
    local compose_build_cmd="docker-compose build"
    if [[ "$NO_CACHE" == "true" ]]; then
        compose_build_cmd="$compose_build_cmd --no-cache"
    fi
    
    execute_command "$compose_build_cmd" "构建Docker Compose服务"
    
    # 启动服务
    execute_command "docker-compose up -d" "启动所有服务"
    
    # 等待服务就绪
    log_info "等待服务启动完成..."
    sleep 30
    
    execute_command "docker-compose ps" "检查服务状态"
    
    cd ../..
    
    log_success "Docker Compose部署完成"
    log_info "服务地址: http://localhost"
    log_info "API文档: http://localhost/docs"
    log_info "监控面板: http://localhost:3000"
    log_info "任务监控: http://localhost:5555"
}

# Kubernetes部署
deploy_k8s() {
    log_info "开始Kubernetes部署..."
    
    # 检查集群连接
    execute_command "kubectl cluster-info" "检查Kubernetes集群"
    
    # 创建命名空间
    execute_command "kubectl create namespace trimodal --dry-run=client -o yaml | kubectl apply -f -" "创建命名空间"
    
    # 部署应用
    execute_command "kubectl apply -f deployment/kubernetes/" "部署应用到Kubernetes"
    
    # 等待部署完成
    log_info "等待Pod启动完成..."
    execute_command "kubectl wait --for=condition=ready pod -l app=trimodal-api -n trimodal --timeout=300s" "等待Pod就绪"
    
    # 检查部署状态
    execute_command "kubectl get pods -n trimodal" "检查Pod状态"
    execute_command "kubectl get services -n trimodal" "检查服务状态"
    
    log_success "Kubernetes部署完成"
    log_info "使用以下命令进行端口转发："
    log_info "kubectl port-forward service/trimodal-api-service 8000:80 -n trimodal"
}

# 部署后检查
post_deploy_check() {
    log_info "执行部署后检查..."
    
    local health_url=""
    
    case $DEPLOYMENT_TYPE in
        local|docker)
            health_url="http://localhost:$PORT/health"
            ;;
        compose)
            health_url="http://localhost/health"
            ;;
        k8s)
            log_info "请先执行端口转发后再检查服务健康状态"
            return 0
            ;;
    esac
    
    if [[ "$DRY_RUN" == "false" && -n "$health_url" ]]; then
        log_info "等待服务启动..."
        sleep 10
        
        for i in {1..10}; do
            if curl -s "$health_url" > /dev/null; then
                log_success "服务健康检查通过"
                break
            else
                if [[ $i -eq 10 ]]; then
                    log_warning "服务健康检查失败，请手动检查服务状态"
                else
                    log_info "重试健康检查... ($i/10)"
                    sleep 5
                fi
            fi
        done
    fi
}

# 显示部署信息
show_deploy_info() {
    log_success "部署完成！"
    
    echo ""
    echo "=== 部署信息 ==="
    echo "部署类型: $DEPLOYMENT_TYPE"
    echo "GPU支持: $GPU_ENABLED"
    echo "服务端口: $PORT"
    
    if [[ -n "$MODEL_PATH" ]]; then
        echo "模型文件: $MODEL_PATH"
    fi
    
    if [[ -n "$CONFIG_PATH" ]]; then
        echo "配置文件: $CONFIG_PATH"
    fi
    
    echo ""
    echo "=== 服务地址 ==="
    case $DEPLOYMENT_TYPE in
        local|docker)
            echo "API服务: http://localhost:$PORT"
            echo "API文档: http://localhost:$PORT/docs"
            echo "健康检查: http://localhost:$PORT/health"
            ;;
        compose)
            echo "API服务: http://localhost"
            echo "API文档: http://localhost/docs"
            echo "健康检查: http://localhost/health"
            echo "监控面板: http://localhost:3000 (admin/admin)"
            echo "任务监控: http://localhost:5555"
            echo "指标监控: http://localhost:9090"
            ;;
        k8s)
            echo "执行端口转发后访问:"
            echo "kubectl port-forward service/trimodal-api-service 8000:80 -n trimodal"
            echo "然后访问 http://localhost:8000"
            ;;
    esac
    
    echo ""
    echo "=== 管理命令 ==="
    case $DEPLOYMENT_TYPE in
        local)
            echo "停止服务: Ctrl+C"
            ;;
        docker)
            echo "停止服务: docker stop trimodal-api"
            echo "查看日志: docker logs trimodal-api"
            echo "删除容器: docker rm trimodal-api"
            ;;
        compose)
            echo "停止服务: docker-compose down (在deployment/docker目录下)"
            echo "查看日志: docker-compose logs trimodal-api"
            echo "重启服务: docker-compose restart trimodal-api"
            ;;
        k8s)
            echo "删除部署: kubectl delete -f deployment/kubernetes/"
            echo "查看日志: kubectl logs -f deployment/trimodal-api -n trimodal"
            echo "扩展服务: kubectl scale deployment trimodal-api --replicas=5 -n trimodal"
            ;;
    esac
}

# 主执行流程
main() {
    log_info "开始TriModalFusion部署 ($DEPLOYMENT_TYPE)"
    
    check_dependencies
    validate_files
    
    case $DEPLOYMENT_TYPE in
        local)
            deploy_local
            ;;
        docker)
            deploy_docker
            ;;
        compose)
            deploy_compose
            ;;
        k8s)
            deploy_k8s
            ;;
        *)
            log_error "不支持的部署类型: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    post_deploy_check
    show_deploy_info
}

# 执行主函数
main