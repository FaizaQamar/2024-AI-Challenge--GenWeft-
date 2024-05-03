from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from googletrans import Translator, LANGUAGES

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'

#######################
embeddings = HuggingFaceInstructEmbeddings()
## Load For Quran and Ahadith Vector Databse
db = FAISS.load_local("./db", embeddings, allow_dangerous_deserialization=True)

#################################
#set huggingface api token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "" #YOUR TOKEN HERE
repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.1)

################################
system_prompt = """
        You are a language model who will think and make decisions like a Muslim Scholar. Your job is to help Muslims with their queries related to their daily life matters.
        The answer should be according to Quran and Ahadith provided to you as documents. If you don't find the answer to any question from the documents provided to you, then apologize.
        Prepare your answer keeping in focus the Context and chat history of the user questions."""
B_INST, E_INST = "<s>[INST] ", " [/INST]"

#Prompt template to has chat histroy as context as well
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

#Memory type from langchain to store the chat history
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

#Initializing the vectorized database as retriever
retriever = db.as_retriever()

#Initializing the Conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_generated_question=False,
    rephrase_question=False,
    return_source_documents=False ,
    combine_docs_chain_kwargs={"prompt": prompt}
    )

##############################################

#Translation module to support all languages
def translate_input_to_english(input_text):
    # Initialize the translator
    translator = Translator()

    # Detect the language of the input text
    detected_language = translator.detect(input_text).lang
    print(detected_language)
    try:
        translated_to_english = translator.translate(input_text, src=detected_language, dest='en').text
    except:
        translated_to_english = input_text
        detected_language = 'en'
    return detected_language, translated_to_english

#Translate the generated answer in user's language
def translate_output_from_english(input_text, lang):
    translator = Translator()
    translated_from_english = translator.translate(input_text, src='en', dest=lang).text
    return translated_from_english
    
##############################################
#Routing Layer
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
        result = user_message
    return result

if __name__ == "__main__":
    app.run(debug=True)
