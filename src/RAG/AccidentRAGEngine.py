import os
import logging
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from sqlalchemy import create_engine

class AccidentRAGEngine:
    def __init__(self, similarity_threshold=0.7):
        """
        기본의 초기화 로직 및 setuo_rag_chain의 기능을 클래스 생성 시 수행합니다.
        """
        load_dotenv()
        self.threshold = similarity_threshold

        # 1. 벡터 스토어 로드 
        self.vector_store = self._get_vector_store

        # 2. LLM 설정
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 3. RAG 체인 설정
        self.rag_chain = self._initialize_rag_chain

    def _get_vector_store(self):
        pass

    def _initialize_rag_chain(self):
        pass
