"""BGE Reranker — 使用 CrossEncoder 对检索结果重排序，召回 20 → 精排 5。

BAAI/bge-reranker-base 基于 XLMRobertaForSequenceClassification，
输入 (query, document) 对，输出相关性分数，分数越高越相关。
"""

from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class Reranker:
    """延迟加载 BGE CrossEncoder，避免阻塞应用启动。"""

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_path)
        return self._model

    def rerank(
        self, query: str, docs: List[Document], top_k: int = 5
    ) -> List[Document]:
        """对文档列表按相关性重排序，返回 top_k 个最相关文档。"""
        if len(docs) <= top_k:
            return docs

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # scores 是 float 列表，高分 = 更相关
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


class RerankingRetriever(BaseRetriever):
    """检索器包装：先召回较多候选，再用 CrossEncoder 精排取 top k_final。

    base_retriever 应配置为 k=20（宽召回），
    RerankingRetriever 将结果重排序后返回 k_final=5 个。
    """

    base_retriever: BaseRetriever
    """被包装的底层检索器（召回阶段）。"""

    reranker: Reranker
    """CrossEncoder 重排序器。"""

    k_final: int = 5
    """重排序后返回的文档数量。"""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if len(docs) <= self.k_final:
            return docs
        return self.reranker.rerank(query, docs, top_k=self.k_final)
