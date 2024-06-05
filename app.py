import streamlit as st
# from streamlit_chat import message
import time 
import os
import base64
import random

st.set_page_config(layout="wide")


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

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

        with col2:
            with st.spinner('Embeddings are in process...'):
                time.sleep(5)
                # ingested_data = data_ingestion()
            st.success('Embeddings are created successfully!')
            # st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)
            st.markdown("Chat With your document using LLama model")


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
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator())
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})





            # # Initialize session state for generated responses and past messages
            # if "generated" not in st.session_state:
            #     st.session_state["generated"] = ["I am ready to help you"]
            # if "past" not in st.session_state:
            #     st.session_state["past"] = ["Hey there!"]


            # user_input = st.text_input("", key="input")

            # # Search the database for a response based on user input and update session state
            # if user_input:
            #     # answer = process_answer({'query': user_input})
            #     answer = "Anwers for the promts "
            #     st.session_state["past"].append(user_input)
            #     response = answer
            #     st.session_state["generated"].append(response)

            # # Display conversation history using Streamlit messages
            # if st.session_state["generated"]:
            #     display_conversation(st.session_state)
        


if __name__ == "__main__":
    main()