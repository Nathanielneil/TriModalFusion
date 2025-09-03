#!/usr/bin/env python3
"""
TriModalFusion Web演示启动脚本
启动带有Web界面的实时检测服务
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'jinja2',
        'python-multipart',
        'websockets',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_environment():
    """设置环境变量和路径"""
    project_root = Path(__file__).parent
    
    # 添加项目路径到Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = f"{project_root}:{project_root / 'src'}:{os.environ.get('PYTHONPATH', '')}"
    
    return project_root

def check_web_files(project_root):
    """检查Web文件是否存在"""
    web_files = [
        'web/templates/index.html',
        'web/static/style.css', 
        'web/static/app.js'
    ]
    
    missing_files = []
    for file_path in web_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("缺少以下Web文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def start_server(host='127.0.0.1', port=8000, dev_mode=False):
    """启动Web服务器"""
    project_root = setup_environment()
    
    print("="*60)
    print("🚀 TriModalFusion Web演示系统")
    print("="*60)
    
    # 检查依赖
    print("📦 检查依赖...")
    if not check_dependencies():
        return 1
    
    # 检查Web文件
    print("🔍 检查Web文件...")
    if not check_web_files(project_root):
        print("❌ Web文件缺失，请确保已创建Web界面文件")
        return 1
    
    print("✅ 环境检查完成")
    
    # 启动服务器
    print(f"🌐 启动Web服务器在 http://{host}:{port}")
    print("💡 功能说明:")
    print("  • 实时摄像头 + 麦克风检测")
    print("  • 文件上传检测")  
    print("  • 三模态特征可视化")
    print("  • 注意力权重热力图")
    print("  • 检测历史记录")
    print()
    
    try:
        import uvicorn
        
        # 切换到项目根目录
        os.chdir(project_root)
        
        # 启动参数
        uvicorn_kwargs = {
            'app': 'deployment.serve:app',
            'host': host,
            'port': port,
            'reload': dev_mode,
            'log_level': 'info'
        }
        
        if dev_mode:
            print("🔧 开发模式：启用热重载")
        
        print(f"⚡ 服务器启动中...")
        print(f"📱 Web界面: http://{host}:{port}")
        print(f"📚 API文档: http://{host}:{port}/docs")
        print(f"🛠️  健康检查: http://{host}:{port}/health")
        print()
        print("按 Ctrl+C 停止服务器")
        print("-"*60)
        
        uvicorn.run(**uvicorn_kwargs)
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
        return 0
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="TriModalFusion Web演示系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python start_web_demo.py                    # 默认启动 (localhost:8000)
  python start_web_demo.py --host 0.0.0.0    # 允许外部访问
  python start_web_demo.py --port 8080       # 自定义端口
  python start_web_demo.py --dev             # 开发模式
        """
    )
    
    parser.add_argument(
        '--host', 
        default='127.0.0.1', 
        help='服务器地址 (默认: 127.0.0.1)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000, 
        help='服务器端口 (默认: 8000)'
    )
    parser.add_argument(
        '--dev', 
        action='store_true', 
        help='开发模式 (启用热重载)'
    )
    parser.add_argument(
        '--check-only', 
        action='store_true', 
        help='仅检查环境，不启动服务器'
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        project_root = setup_environment()
        print("🔍 环境检查中...")
        
        dep_ok = check_dependencies()
        web_ok = check_web_files(project_root)
        
        if dep_ok and web_ok:
            print("✅ 环境检查通过，可以启动服务器")
            return 0
        else:
            print("❌ 环境检查失败")
            return 1
    
    return start_server(args.host, args.port, args.dev)

if __name__ == '__main__':
    exit(main())