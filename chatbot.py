
# chạy trên gpu 10+ VRAM

import torch
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print (f"Đang chạy trên thiết bị: {device}")

    # tải địa chỉ tuyệt đối của file pdf
    file_path = "YOLOv10_Tutorials.pdf"

    # tải file pdf lên RAM
    loader = PyPDFLoader(file_path)
    document = loader.load()

    # tải model chuyển văn bản thành các vector
    embedding = HuggingFaceEmbeddings(
        model_name = "bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs = {'device': device},
        encode_kwargs = {'batch_size': 64}
    )

    # khởi tạo bộ tách văn bản
    semantic_splitter = SemanticChunker(
        embeddings = embedding,
        buffer_size = 1,
        breakpoint_threshold_type = "percentile",
        breakpoint_threshold_amount = 95,
        min_chunk_size = 500,
        add_start_index = True
    )
    docs = semantic_splitter.split_documents(document)

    # tạo vector database với chroma
    vector_db = Chroma.from_documents(documents = docs, embedding = embedding)

    # tạo đối tượng truy vấn database, ở dây là chroma
    retriever = vector_db.as_retriever()

    int8_config = BitsAndBytesConfig(
        load_in_8bit = True
    )

    # sử dụng model vicuna 7b v1.5
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map = 'auto',
        quantization_config = int8_config,
        low_cpu_mem_usage = True
    )
    # khởi tạo tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # tích hợp tokenizer và model thành một pipeline để tiện sử dụng
    model_pipeline = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_new_tokens = 512,
        max_length = None,
        return_full_text = False,
        pad_token_id = tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline = model_pipeline)

    # chạy thử ...
    template = """USER: Bạn là một trợ lý ảo chuyên trả lời câu hỏi dựa trên tài liệu. Hãy sử dụng các đoạn ngữ cảnh (Context) dưới đây để trả lời câu hỏi (Question). Nếu thông tin không có trong tài liệu, hãy nói rằng bạn không biết, tuyệt đối không bịa đặt thông tin.
    Context: {context}
    Question: {question}
    ASSISTANT:"""

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": retriever | format_docs, "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    USER_QUESTION = "YOLOv10 là gì?"
    output = rag_chain.invoke(USER_QUESTION)
    answer = output.strip()
    print(answer)

if __name__ == "__main__":
    main()