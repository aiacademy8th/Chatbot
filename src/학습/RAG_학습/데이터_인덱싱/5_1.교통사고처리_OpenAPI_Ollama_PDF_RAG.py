import os

from dotenv import load_dotenv
print(load_dotenv()) # 환경변수 값 로딩
#os.environ['OPENAI_API_KEY']  = os.getenv('OPENAI_API_KEY')

# OLLAMA LLM Or OpenAI API
OLLAMA_LLM = False
# OpenAI Embedding
OpenAI_Embedding = True

from langchain_community.document_loaders.pdf import PyPDFLoader
# 1) 분석할 PDF 파일 로딩

# 현재 실행 중인 파이썬 파일의 디렉토리 경로를 가져옵니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
# 해당 경로와 파일명을 결합하여 절대 경로를 만듭니다.
file_path = os.path.join(current_dir, "훈령_교통사고조사규칙 전문_202110.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()

#print(documents)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # text 분할
# 단계 2: 문서 분할(Split Documents) / 청크로 나누고
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs_split = text_splitter.split_documents(documents)
#print(docs_split)

if OpenAI_Embedding:
    print("using openai embedding!!")
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
else:
    print("using huggingface embedding!!")
    # HuggingFaceEmbeddings 모델 활용 임베딩
    #from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    # 사전 학습된 sentence-transformers 지원 모델
    # https://www.sbert.net/docs/pretrained_models.html 참조
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

from langchain_openai import  OpenAIEmbeddings

# 단계 3: 분할 Documents 임베딩화 및 vectorstore 저장
from langchain_community.vectorstores import Chroma
# from_documents() : Document항목 list를 임베딩과 동시 vectorstore에 저장 메서드
# Document항목 list가 아닌 text항목 list 경우 from_texts()메서드 활용
vector_store = Chroma.from_documents(docs_split, embedding=embeddings)

# 단계 4: 임베딩벡터를 검색(retriever)하기 위한 객체 생성
# chain 실행시 질문 쿼리와 유사한 문장 검색
retriever = vector_store.as_retriever()


# 단계 5: 프롬프트 생성(Create Prompt)
# llm 에 전달할 프롬프트 생성 / context + 질문 qurey
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
template="""You are an assistant for question-answering tasks.
You are an assistant for question-answering tasks.
[출력 규칙]
- 아래 주어진 context를 분석한 후 필히 사고자 입장에서 단계별 준비해야될 내용을 출력
- 만약 답변하기에 정보가 부족하면 정보가 부족합니다 를 출력

\nQuestion: {question} \nContext: {context} \nAnswer:  """,
input_variables=['context', 'question']
)
#  concise : 간결한

# 단계 6: 질문에 응답할 LLM 모델 구성
# OpenAPI 모델 Or Ollama local 모델
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


if OLLAMA_LLM:
    # from langchain_community.llms import Ollama
    print("unsing Ollama LLM !!")
    from langchain_ollama import OllamaLLM

    chat_llm = OllamaLLM(
        base_url='http://192.168.0.130:11434',
        # model = 'llama2' # 명령 메시지에 따라 약 2~5분 정도 추론 소요
        model="llama3.1"  # 한국어보다 영어로 질문 입력
    )
else:
    print("unsing OpenAI LLM !!")
    chat_llm = ChatOpenAI(model_name="gpt-4o-mini",
                     temperature=0)  # Modify model_name if you have access to GPT-4

import re
def format_ret_docs(ret_docs):
    # 질문 query를 바탕으로 retriever가 유사 문장을 검색한 문서 결과(page_content)를
    # 하나의 문단으로 합쳐주는 함수 구현
    text_to_docs= "\n\n".join(doc.page_content for doc in ret_docs)
    text_to_docs = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text_to_docs)
    print(" context text ".center(80,'='))
    print(text_to_docs)
    return text_to_docs

# 단계 7: 구성요소를 결합하는 체인 생성 / LCEL방식
# RunnablePassthrough() : 'question' 프롬프트 항목에는 질문 쿼리 그대로 전달하는 기능
# 'context'프롬프트 항목에는 retriever를 활용해 질문 쿼리를 바탕으로 검색한
# 유사문장을 합쳐서 채움
# 1. prompt 를 구성할 내용 생성 -> 2. llm에 전달할 prompt 구성
# 3. llm에 응답 요청 -> 4. llm 응답 내용을 문자열 parser 통해 출력
rag_chain = (
    {'context':retriever | format_ret_docs, 'question': RunnablePassthrough()}
    | prompt
    | chat_llm
    | StrOutputParser()
)
question = "초동조치에 대해 알려줘"
reponse_data = rag_chain.invoke(question)
print(' 출력 결과 '.center(80,"="))
print(reponse_data)

#
# # 6) gradio 웹 페이지를 활용한 ChatGPT Demo App 구성 및 실행
# import gradio as gr
#
# def respond(message, chat_history):  # 채팅봇의 응답을 처리하는 함수를 정의합니다.
#
#     # query = "network design of yolo"
#     result = rag_chain.invoke(message)
#     bot_message = result
#     chat_history.append((message, bot_message))  # 채팅 기록에 사용자의 메시지와 봇의 응답을 추가합니다.
#
#     return "", chat_history  # 수정된 채팅 기록을 반환합니다.
#
# with gr.Blocks() as demo:  # gr.Blocks()를 사용하여 인터페이스를 생성합니다.
#     chatbot = gr.Chatbot(label="채팅창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
#     msg = gr.Textbox(label="입력")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
#     clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.
#
#     msg.submit(respond, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
#     clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.
#
# demo.launch(share=True)
