from flask import Flask, render_template, request, jsonify
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
#from langchain.llms import HuggingFaceHub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
import gradio as gr
from googletrans import Translator, LANGUAGES

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'

#######################
embeddings = HuggingFaceInstructEmbeddings()
    ## For Quran and Ahadith Docs
umrah_db = FAISS.load_local("E:/D/Faiza/PhD/Umrah Competition/db", embeddings, allow_dangerous_deserialization=True)
#######################

############################
# get a token: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token

#from getpass import getpass

#HUGGINGFACEHUB_API_TOKEN = getpass()

#os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
#################################3
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.2)
#llm = HuggingFaceEndpoint(repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_new_tokens": 1024})
##########################

##########################
def initialize_again():
    print("in initilization")
    system_prompt = """
        You are a Muslim Scholar who helps Muslims with their queries related to Umrah and their daily life matters.
        If you don't know the answer to any question from the documents provided to you, then apologize.
        Prepare your answer keeping in focus the Context and chat history of the user questions."""
    B_INST, E_INST = "<s>[INST] ", " [/INST]"
    template = (
                B_INST
                + system_prompt
                + """

            Context: {context} / {chat_history}
            User: {question}"""
                + E_INST
            )
    prompt = PromptTemplate(input_variables=["context", "chat_history","question"], template=template)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    retriever = umrah_db.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_generated_question=False,
    rephrase_question=False,
    return_source_documents=False ,
    #max_tokens_limit=500,
    combine_docs_chain_kwargs={"prompt": prompt}
    )


################################
system_prompt = """
        You are a Muslim Scholar who helps Muslims with their queries related to Umrah and their daily life matters.
        If you don't know the answer to any question from the documents provided to you, then apologize.
        Prepare your answer keeping in focus the Context and chat history of the user questions."""
B_INST, E_INST = "<s>[INST] ", " [/INST]"
template = (
                B_INST
                + system_prompt
                + """

            Context: {context} / {chat_history}
            User: {question}"""
                + E_INST
            )
prompt = PromptTemplate(input_variables=["context", "chat_history","question"], template=template)

############################

############################
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
retriever = umrah_db.as_retriever()

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_generated_question=False,
    rephrase_question=False,
    return_source_documents=False ,
    #max_tokens_limit=500,
    combine_docs_chain_kwargs={"prompt": prompt}
    )

##############################################

##############################################
def translate_input_to_english(input_text):
    # Initialize the translator
    translator = Translator()

    # Detect the language of the input text
    detected_language = translator.detect(input_text).lang
    print(detected_language)
    translated_to_english = translator.translate(input_text, src=detected_language, dest='en').text
    return detected_language, translated_to_english

def translate_output_from_english(input_text, lang):
    translator = Translator()
    translated_from_english = translator.translate(input_text, src='en', dest=lang).text
    return translated_from_english
    
##############################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    lang, msg = translate_input_to_english(user_message)
    bot_response = get_bot_response(msg)
    response= translate_output_from_english(bot_response, lang)
    return jsonify({'bot_response': response})

def get_bot_response(user_message):
    try:
        result=chain(user_message)
        result = (result['answer'].split('[/INST]')[-1])
    except:
        print("in exception")
        initialize_again()
        result=chain(user_message)
        result = (result['answer'].split('[/INST]')[-1])
    return result

if __name__ == "__main__":
    app.run(debug=True)
