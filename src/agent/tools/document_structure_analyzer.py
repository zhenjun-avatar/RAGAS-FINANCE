"""
文档结构图生成器 - 极简智能版
LLM 自动决定图表类型，无硬编码
"""

from typing import Dict, Any
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
import re


class DocumentStructureAnalyzer:
    """文档结构分析器 - 极简版"""
    
    def __init__(self):
        from .llm import get_llm

        self.llm = get_llm(temperature=0.3)
    
    async def generate_diagram(
        self,
        content: str,
        user_prompt: str,
        title: str = ""
    ) -> Dict[str, Any]:
        """
        生成文档结构图 - 核心方法
        
        Args:
            content: 文档内容
            user_prompt: 用户的分析需求（如："分析人物关系"）
            title: 文档标题（可选）
        
        Returns:
            {
                'success': bool,
                'mermaid_definition': str,
                'description': str
            }
        """
        try:
            logger.info(f"[DocDiagram] Generating diagram with prompt: {user_prompt}")
            
            # 调用 LLM 生成 Mermaid 图表
            mermaid_def = await self._call_llm(content, user_prompt, title)
            
            logger.info(f"[DocDiagram] Successfully generated diagram")
            
            return {
                'success': True,
                'mermaid_definition': mermaid_def,
                'description': user_prompt
            }
            
        except Exception as e:
            logger.error(f"[DocDiagram] Generation failed: {e}")
            return {
                'success': False,
                'mermaid_definition': '',
                'description': ''
            }
    
    async def _call_llm(
        self,
        content: str,
        user_prompt: str,
        title: str
    ) -> str:
        """
        调用 LLM 生成 Mermaid 图表
        这是唯一的核心方法，LLM 自动决定一切
        """
        # 限制内容长度（避免 token 超限）
        max_content_length = 4000
        content_preview = content[:max_content_length]
        if len(content) > max_content_length:
            content_preview += "\n\n...(内容已截断)"
        
        # 构建 system prompt
        system_prompt = """你是一个文档分析专家。根据用户的需求和文档内容，生成合适的 Mermaid 图表。

你可以使用以下 Mermaid 图表类型（根据需求自动选择）：

1. mindmap - 思维导图（适合：人物关系、概念关系、主题分析）
   mindmap
     root((中心))
       分支1
         子分支
       分支2

2. flowchart - 流程图（适合：流程、步骤、逻辑）
   flowchart LR
     A[开始] --> B[步骤]
     B --> C[结束]

3. graph TD - 层级图（适合：分类、层级、结构）
   graph TD
     A[根] --> B[子1]
     A --> C[子2]

4. sequenceDiagram - 时序图（适合：交互、对话）
   sequenceDiagram
     A->>B: 消息

5. classDiagram - 类图（适合：系统架构、组件关系）
   classDiagram
     Class1 --> Class2

**重要**：
- 只返回 Mermaid 代码，不要任何解释
- 根据用户需求自动选择最合适的图表类型
- 保持简洁，节点数量控制在 10-20 个
- 使用中文标签
"""
        
        # 构建 user prompt
        user_message = f"""
文档标题：{title if title else "无标题"}

用户需求：{user_prompt}

文档内容：
{content_preview}

请根据用户需求生成 Mermaid 图表（只返回 Mermaid 代码）：
"""
        
        # 调用 LLM
        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        # 提取 Mermaid 代码
        return self._extract_mermaid(response.content)
    
    def _extract_mermaid(self, response_text: str) -> str:
        """从 LLM 响应中提取 Mermaid 定义"""
        # 尝试提取 ```mermaid ... ``` 块
        match = re.search(r'```mermaid\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 尝试提取 ``` ... ``` 块
        match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 直接返回（假设整个响应就是 Mermaid 定义）
        return response_text.strip()


# ============ 导出函数（简洁接口）============

async def generate_document_diagram(
    content: str,
    user_prompt: str,
    title: str = ""
) -> Dict[str, Any]:
    """
    生成文档结构图 - 极简接口
    
    Args:
        content: 文档内容
        user_prompt: 用户需求（如："分析人物关系"）
        title: 文档标题（可选）
    
    Returns:
        {
            'success': bool,
            'mermaid_definition': str,
            'description': str
        }
    """
    analyzer = DocumentStructureAnalyzer()
    return await analyzer.generate_diagram(content, user_prompt, title)

