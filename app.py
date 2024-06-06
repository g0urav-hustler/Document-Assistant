import streamlit as st
# from streamlit_chat import message
import time 
import os
import base64
import random
# from chromadb.config import Settings

st.set_page_config(layout="wide")

def generate_answer():
    answer = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    return answer

# Streamed response emulator
def response_generator(response):
    
    for word in response.split():
        yield word + " "
        time.sleep(0.2)

MODEL_NAME = "MBZUAI/LaMini-T5-738M"
device = "cpu"
print(f"Checkpoint path: {MODEL_NAME}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map=device,
    torch_dtype=torch.float32
)

 
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='data_base',
        anonymized_telemetry=False
)
persist_directory = "data_base"

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root,file))

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 200)
    texts = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name = MODEL_NAME)
    db = Croma.from_documents(texts, embeddings, ersist_directory=persist_directory, client_settings=CHROMA_SETTINGS)   
    db.persist()
    db=None

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        "text2text-generation",
        model= base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p = 0.95,
        device = device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm
@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embedding = SentenceTransformersEmbeddings(model_name = MODEL_NAME)
    db = Chroma(persist_directory = "data_base",embedding_function = embeddings, client_settings=CHROMA_SETTINGS )
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents = True
    )

    return qa



def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# def display_conversation(history):
#     for i in range(len(history["generated"])):
#         message(history["past"][i], is_user=True, key=str(i) + "_user")
#         message(history["generated"][i],key=str(i))

def main():
    st.title("Document Assistant ")
    # st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    
    # st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        os.makedirs("docs", exist_ok = True)

        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

        col1, col2= st.columns([1,2])
        with col1:
            # st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.markdown("File Details")
            st.json(file_details)
            # st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            st.markdown("File Format")
            pdf_view = displayPDF(filepath)

        with st.spinner('Embeddings are in process...'):
                time.sleep(5)
                # ingested_data = data_ingestion()
        st.success('Embeddings are created successfully!')

        with col2:
            
            # st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)
            st.subheader("Chat with your document using LLama Model")
            # st.markdown("Chat With your document using LLama model")

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("What is up?"):
                # Add user message to chat history

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.spinner("finding the answer"):
                    time.sleep(5)
                    answer = generate_answer()
                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator(answer))

                st.session_state.messages.append({"role": "user", "content": prompt})
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()