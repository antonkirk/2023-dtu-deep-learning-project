from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from huggingface_hub import login
import streamlit as st


st.set_page_config(
        page_title="BookChat",
)

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    st.write("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")

st.title("📖 BookChat")
st.caption("Ask your textbook questions using GPT-3.5 Turbo.")


uploaded_file = st.file_uploader("Upload a chapter from your textbook, or a course description", type=("txt"))

question = st.text_input(
    "Ask something about the document",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

if uploaded_file and question and openai_api_key:
    document_name = uploaded_file.name
    chapter = uploaded_file.read().decode()
    with open(f'documents/{document_name}', 'w') as f:
        f.write(chapter)

    loader = TextLoader(f'documents/{document_name}')

    text_splitter = RecursiveCharacterTextSplitter(separators= ["\n\n", "\n", "\t"], chunk_size=1500, chunk_overlap=500)
    splits = text_splitter.split_documents(loader.load())

    #for split in splits:
    #    print(f"Split: {split}")

    #login('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')

    # model_name = "dtu-deep-learning-course-f2023/msmarco-rag-finetune"
    # model_kwargs = {'device': 'cpu'}
    # encode_kwargs = {'normalize_embeddings': False}
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    TEMPLATE = """ \
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise, but thorough.
    Question: {question}
    Context: {context}
    Answer:
    """

    rag_prompt = PromptTemplate.from_template(template=TEMPLATE)

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, openai_api_key=openai_api_key)

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()

    docs = vectorstore.similarity_search(question)

    for doc in docs:
        print(doc)

    st.write(rag_chain.invoke(question))
    
    st.write("Sources:")
    for i, doc in enumerate(docs):
        source = doc.metadata["source"]
        st.write(f"{i}: {source}")
