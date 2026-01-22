
import os
import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수 로드
dotenv.load_dotenv()

def main():
    """
    RAG 챗봇의 메인 실행 함수입니다.
    """
    # 1. FAISS 벡터 DB 로드
    db_path = "vectorDB/faiss_index_samsung_fire"
    if not os.path.exists(db_path):
        print(f"오류: 벡터 DB 경로를 찾을 수 없습니다 - {db_path}")
        return

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local(db_path, embeddings_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # 2. 프롬프트 템플릿 설정
    template = """
    당신은 삼성화재 자동차보험 약관에 대해 상세히 답변하는 AI 어시스턴트입니다.
    사용자의 질문에 대해 아래 제공된 '문맥' 정보를 바탕으로, 명확하고 친절하게 설명해주세요.
    답변은 반드시 한국어로 작성해야 합니다. 문맥에 없는 내용은 답변하지 마세요.

    ---
    [문맥]
    {context}
    ---
    [질문]
    {question}
    ---
    [답변]
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. LLM 모델 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 4. RAG 체인 구성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("안녕하세요! 삼성화재 자동차보험 약관 챗봇입니다.")
    print("궁금한 점을 물어보세요. (종료하려면 'exit' 또는 'q'를 입력하세요)")

    while True:
        user_question = input("\n질문: ")
        if user_question.lower() in ["exit", "q"]:
            print("챗봇을 종료합니다.")
            break

        if user_question.strip():
            response = rag_chain.invoke(user_question)
            print("\n답변:", response)

if __name__ == "__main__":
    main()
