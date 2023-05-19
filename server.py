import os
import http.server
import socketserver
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, \
    ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
from dotenv import load_dotenv
import sys

from http import HTTPStatus

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("KEY")

def createVectorIndex(path):
    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    #define LLM
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))

    #load data
    docs = SimpleDirectoryReader(path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=prompt_helper)

    #create vector index
    vectorIndex = GPTVectorStoreIndex.from_documents(
        docs, service_context=service_context
    )
    vectorIndex.storage_context.persist(persist_dir="storage")

    return vectorIndex

def display_first_line(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            print(first_line)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")

def answerMe():
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

    # Configure the parameters for the query engine
    query_engine = index.as_query_engine(
        temperature=0.8,
        max_tokens=500,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.5
    )

    q = "Here is a context for your reponses:" \
        "Talk as an AI bot who provides legal advice as a response and also provide specific examples of cases within knowledge you have of previous cases as example." \
        "Below is your question you have to respond to:" \
        "Can you talk about an interesting kidnapping case and what happenend in that trial?"

    print(q)
    print('------------')
    response = query_engine.query(q)
    print(response)
    return response


# createVectorIndex('knowledge')
# answerMe()

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/respond':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            response = answerMe()  # Generate the response using the answerMe() function
            msg = 'Python is running on LOL! You requested %s' % (response)
            self.wfile.write(msg.encode())
        else:
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            msg = 'Python is running on Qoddi! You requested %s' % (self.path)
            self.wfile.write(msg.encode())

port = int(os.getenv('PORT', 8080))
print('Listening on port %s' % (port))
httpd = socketserver.TCPServer(('', port), Handler)
httpd.serve_forever()