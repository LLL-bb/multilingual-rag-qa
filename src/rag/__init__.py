"""中英混合文档 RAG 问答系统 —— 语言感知检索的核心方法复现。

子模块按职责划分,可单独 import:

    rag.language   语言检测
    rag.loaders    PDF / DOCX / TXT / MD 解析
    rag.chunking   分块
    rag.embedders  语言感知嵌入
    rag.index      FAISS 双索引
    rag.pipeline   端到端流程
"""

__version__ = "0.1.0"
