from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
import streamlit as st

FILE_PATH = "documents/neural_networks_and_deep_learning.pdf"
TEXT = """
Nineteen Eighty-Four (also published as 1984) is a dystopian novel and cautionary tale by English writer George Orwell. It was published on 8 June 1949 by Secker & Warburg as Orwell's ninth and final book completed in his lifetime. Thematically, it centres on the consequences of totalitarianism, mass surveillance and repressive regimentation of people and behaviours within society.[2][3] Orwell, a democratic socialist, modelled the authoritarian state in the novel on the Soviet Union in the era of Stalinism, and Nazi Germany.[4] More broadly, the novel examines the role of truth and facts within societies and the ways in which they can be manipulated.

The story takes place in an imagined future in an unspecified year believed to be 1984, when much of the world is in perpetual war. Great Britain, now known as Airstrip One, has become a province of the totalitarian superstate Oceania, which is led by Big Brother, a dictatorial leader supported by an intense cult of personality manufactured by the Party's Thought Police. The Party engages in omnipresent government surveillance and, through the Ministry of Truth, historical negationism and constant propaganda to persecute individuality and independent thinking.[5]

The protagonist, Winston Smith, is a diligent mid-level worker at the Ministry of Truth who secretly hates the Party and dreams of rebellion. Smith keeps a forbidden diary. He begins a relationship with a colleague, Julia, and they learn about a shadowy resistance group called the Brotherhood. However, their contact within the Brotherhood turns out to be a Party agent, and Smith and Julia are arrested. He is subjected to months of psychological manipulation and torture by the Ministry of Love and is released once he has come to love Big Brother.

Nineteen Eighty-Four has become a classic literary example of political and dystopian fiction. It also popularised the term "Orwellian" as an adjective, with many terms used in the novel entering common usage, including "Big Brother", "doublethink", "Thought Police", "thoughtcrime", "Newspeak", and "2 + 2 = 5". Parallels have been drawn between the novel's subject matter and real life instances of totalitarianism, mass surveillance, and violations of freedom of expression among other themes.[6][7][8] Orwell described his book as a "satire",[9] and a display of the "perversions to which a centralised economy is liable," while also stating he believed "that something resembling it could arrive."[9] Time included the novel on its list of the 100 best English-language novels published from 1923 to 2005,[10] and it was placed on the Modern Library's 100 Best Novels list, reaching number 13 on the editors' list and number 6 on the readers' list.[11] In 2003, it was listed at number eight on The Big Read survey by the BBC.
"""

def load_text_from_file(path: str):
    loader = PyPDFLoader(path)
    pages = loader.load()

    # Combine the pages, and replace the tabs with spaces
    text = ""

    for page in pages:
        text += page.page_content
        
    text = text.replace('\t', ' ')
    return text

def embed_text(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"],
                                               chunk_size=8000,
                                               chunk_overlap=1500)

    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors = embeddings.embed_documents([x.page_content for x in docs])

    return vectors

def generate_questions(llm, text):

    # Define the prompt
    prompt = f"Please provide 3 short comprehension questions about the following technical text: {text}"
    
    # Use the model to generate questions
    response = llm.invoke(prompt)

    answers = response.content
    print(answers)
    
    return answers

def generate_answers(llm, questions, context):
    # Define the prompt
    template = """ \
        Please provide concise answers to the following questions using the context: 
        questions: {questions}        
        context: {context}
        Correct answers:
    """

    prompt = PromptTemplate.from_template(template=template)
    
    chain = prompt | llm | StrOutputParser()
    
    # Use the model to generate answers 
    response = chain.invoke({'questions': questions, 'context': context})

    return response

st.set_page_config(
        page_title="Q&A Chatbot",
)

if not "is_questions_generated" in st.session_state:
    st.session_state["is_questions_generated"] = False

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    st.write("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")

st.title("💬 Q&A Chatbot")
st.caption("Generate comprehension questions for your textbook using GPT-3.5 Turbo.")

# uploaded_file = st.file_uploader("Upload a chapter from your textbook", type=("pdf"))

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Generate questions to begin the learning session."}]

if st.button("Generate Questions", disabled=st.session_state["is_questions_generated"]):
    if openai_api_key:
        llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.7, openai_api_key=openai_api_key)
        # text = load_text_from_file(FILE_PATH)
        questions = generate_questions(llm, TEXT)
        correct_answers = generate_answers(llm, questions, TEXT)
        st.session_state["questions"] = questions
        st.session_state["correct_answers"] = correct_answers
        st.session_state.messages.append({"role": "assistant", "content": questions})
        st.session_state["is_questions_generated"] = True
    else:
        st.info("Please add your OpenAI API key to continue.")

for msg in st.session_state.messages:
   st.chat_message(msg["role"]).write(msg["content"])

if answer := st.chat_input("Your answer", disabled=not st.session_state["is_questions_generated"]):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    else:
        TEMPLATE = """ \
        You are a tutor for a student in the middle of a learning session with a chat history. 
        Help them by grading their answers and providing feedback. Do not give out the correct answer, but rather guide the student.
        You are allowed to answer related questions, but you should only give out the answer when the student answers the questions.
        If you see in the chat history that the session is complete, you are allowed to discuss other topics.
        Chat history: {history}
        Student answer: {answer}
        Correct answers: {correct_answers}
        Feedback:
        """

        prompt = PromptTemplate.from_template(template=TEMPLATE)

        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, openai_api_key=openai_api_key)
        
        st.session_state.messages.append({"role": "user", "content": answer})
        st.chat_message("user").write(answer)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({'history': st.session_state.messages, 'questions': st.session_state["questions"], 'answer': answer, 'correct_answers': st.session_state["correct_answers"]})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
