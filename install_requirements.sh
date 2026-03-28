#!/bin/bash
# 安装FastAPI服务所需的依赖项

# 确保我们在正确的目录
cd "$(dirname "$0")"

echo "安装FastAPI服务所需的依赖项..."

# 检查Python版本
python --version || python3 --version

# 安装依赖项
pip install fastapi uvicorn websockets pydantic langchain langchain-openai openai deepseek-ai langgraph || python -m pip install fastapi uvicorn websockets pydantic langchain langchain-openai openai deepseek-ai langgraph

# 如果使用虚拟环境
if [ -d "venv" ] || [ -d ".venv" ]; then
    echo "检测到虚拟环境，在虚拟环境中安装依赖项..."
    if [ -d "venv" ]; then
        source venv/bin/activate || source venv/Scripts/activate
    else
        source .venv/bin/activate || source .venv/Scripts/activate
    fi
    pip install fastapi uvicorn websockets pydantic langchain langchain-openai openai deepseek-ai langgraph
fi

# 安装当前包
pip install -e . || python -m pip install -e .

echo "依赖项安装完成！" 