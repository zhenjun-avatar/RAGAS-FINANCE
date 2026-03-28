#!/usr/bin/env python3
"""直接启动API服务器"""

import sys
import os

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import uvicorn
from api.server import app
from core.config import config

if __name__ == "__main__":
    print("启动 Node-centric RAG API 服务器")
    print("=" * 50)
    print(f"📍 主机: {config.host}")
    print(f"📍 端口: {config.port}")
    print(f"📍 API文档: http://{config.host}:{config.port}/docs")
    print(f"📍 健康检查: http://{config.host}:{config.port}/health")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower()
    )
