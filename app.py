import streamlit as st
import tempfile
import os
from time import time
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# khởi tạo các biến quản lí phiên
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# decorator cache để không lại tải lại khi tương tác với web 
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs = {'device': 'cuda'},
        encode_kwargs = {'batch_size': 128}
    )

# decorator cache để không lại tải lại khi tương tác với web 
@st.cache_resource
def load_llm():
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    int8_config = BitsAndBytesConfig(load_in_8bit = True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = int8_config,
        low_cpu_mem_usage = True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_pipeline = pipeline(
        task = "text-generation",
        model = model,
        tokenizer = tokenizer,
        device_map = 'auto',
        max_new_tokens = 512,
        pad_token_id = tokenizer.eos_token_id
    )

    return HuggingFacePipeline(pipeline = model_pipeline)

def process_pdf(uploaded_file):

    # tạo temp file để lưu trữ
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp_file:
        tmp_file.write (uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # tải file pdf lên RAM
    loader = PyPDFLoader(tmp_file_path)
    document = loader.load()

    # bộ tách đoạn văn bản theo ngữ cảnh
    semantic_splitter = SemanticChunker(
        embeddings = st.session_state.embeddings,
        buffer_size = 1,
        min_chunk_size = 500,
        add_start_index = True,
        breakpoint_threshold_type = "percentile",
        breakpoint_threshold_amount = 95
    )

    # tách văn bản
    docs = semantic_splitter.split_documents(document)

    # lưu trữ các đoạn văn bản đã được tách và chuyển sang vector bằng vector database
    vector_db = Chroma.from_documents(documents = docs, embedding = st.session_state.embeddings)

    # tạo đối tượng truy vấn database
    retriever = vector_db.as_retriever()

    # template cho llm hiểu ngữ cảnh nhiệm vụ và quy tắc trả lời
    template = """USER: Bạn là một trợ lý chuyên trả lời câu hỏi dựa trên tài liệu. Hãy sử dụng các đoạn ngữ cảnh (Context) dưới đây để trả lời câu hỏi (Question). Nếu thông tin không có trong tài liệu, hãy trả lời rằng bạn không biết. Tuyệt đối không được bịa đặt thông tin.
    Context: {context}
    Question: {question}
    Answer:"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    # dọn file tạm
    os.unlink(tmp_file_path)
    return rag_chain, len(docs)

def add_message(role, content):
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "time_stamp": time
    })

def display_chat():
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write("Xin chào! Tôi là AI assistant. Hãy upload file PDF và bắt đầu đặt câu hỏi về nội dung tài liệu nhé! 😊")

# xóa lịch sử chat
def clear_chat():
    st.session_state.chat_history = []

# giao diện (UI)
def main(): 

    st.set_page_config(page_title = "PDF RAG Assistant", layout = "wide")
    st.title("PDF RAG Assistant")
    st.logo("./face_id01.jpg", size = "large")

    with st.sidebar:
        st.title("⚙️ Cài đặt")

        # tải model một lần duy nhất khi ứng dụng được khởi chạy
        if not st.session_state.model_loaded:
            st.warning("⏳ Đang tải models...")

            with st.spinner("Đang tải AI models..."):
                st.session_state.embeddings = load_embeddings()
                st.session_state.llm = load_llm()
                st.session_state.model_loaded = True
            st.success("✅ Models đã sẵn sàng!")
            st.rerun()
        else:
            st.success("✅ Models đã sẵn sàng!")

        st.markdown("---")
        st.subheader("📄 Upload tài liệu")

        # upload file pdf và tiến hành xử lí, tạo RAG chain
        uploaded_file = st.file_uploader("Chọn file PDF", type = "pdf")

        # upload file pdf và xử lý
        if uploaded_file and st.button("🔄 Xử lý PDF", use_container_width = True):
            with st.spinner("Đang xử lý..."):

                # tạo RAG chain
                st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name

                # nếu tải lại file PDF hoặc tải file PDF khác
                clear_chat()
                add_message("assistant", f"✅ Đã xử lí thành công file {st.session_state.pdf_name}!\n\n📊 Tài liệu được chia thành {num_chunks} phần. Bạn có thể bắt đầu đặt câu hỏi vè nội dung tài liệu.")

            st.rerun()

        if st.session_state.pdf_processed:
            st.success(f"📄 Đã tải: {st.session_state.pdf_name}")
        else:
            st.info("📄 Chưa tải file PDF")

        st.markdown("---")

        st.subheader("💬 Điều khiển Chat")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width = True):
            clear_chat()

        st.markdown ("---")

        st.subheader("📋 Hướng dẫn")
        st.markdown("""
        **Cách sử dụng:**
        1. **Upload PDF** - Chọn file và nhấn "Xử lý PDF"
        2. **Đặt câu hỏi** - Nhập câu hỏi trong ô chat
        3. **Nhận trả lời** - AI sẽ trả lời dựa trên nội dung PDF
        """)

    st.markdown("*Trò chuyện với Chatbot để trao đổi về nội dung tài liệu PDF của bạn*")

    # giao diện hỏi đáp
    chat_container = st.container()
    # in ra lịch sử chat
    with chat_container:
        display_chat()

    if st.session_state.model_loaded:
        if st.session_state.pdf_processed:
            user_input = st.chat_input("Nhập câu hỏi của bạn...")

            if user_input:
                add_message("user", user_input)
                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        try:
                            output = st.session_state.rag_chain.invoke(user_input)
                            answer = output.split("Answer:")[1].strip() if "Answer:" in output else output.strip()

                            st.write (answer)
                            add_message("assistant", answer)

                        except Exception as e:
                            error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
        else:
            st.info("🔄 Vui lòng upload và xử lý file PDF trước khi bắt đầu chat!")
            st.chat_input("Nhập câu hỏi của bạn...", disabled = True)
    else:
        st.info("⏳ Đang tải AI models, vui lòng đợi...")
        st.chat_input("Nhập câu hỏi của bạn...", disabled = True)
            
if __name__ == "__main__":
    main()