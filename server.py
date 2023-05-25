import os
import http.server
import socketserver
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, \
    ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
from dotenv import load_dotenv
import json
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA, FAISS
from langchain.document_loaders import UnstructuredFileLoader

from http import HTTPStatus

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("KEY")


def createVectorIndex(path):
    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    # define LLM
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))

    # load data
    docs = SimpleDirectoryReader(path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=prompt_helper)

    # create vector index
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


def answerMe(prompt, language):
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

    # Configure the parameters for the query engine
    query_engine = index.as_query_engine(
        temperature=0.6,
        max_tokens=500,
        top_p=0.9,
        frequency_penalty=0.3,
        presence_penalty=0.2
    )

    q = "Here is a context for your responses:" \
        "Respond as an AI helper who provides legal advice as a response and also provide relevant examples of previous cases and then advises if the query is a personal situation and not a question" \
        "Below is your question you have to respond to:"

    # Concatenate the prompt and language
    q = q + "\n" + prompt + "\nTranslate response to " + language + ".\nGenerate a complete response."

    print(q)
    print('------------')
    response = query_engine.query(q)
    return response


# createVectorIndex('knowledge')
# answerMe()

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        # Initialize the chain variable as None
        chain = None
        docsearch = None

        if self.path == '/respond':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            prompt = data.get('prompt')
            language = data.get('language')

            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # generated_response = "random stuff for now"
            # Generate the response using the prompt and language
            generated_response = answerMe(prompt, language)

            # convert response to json format manually
            json_response = f'{{"advice": "{generated_response}"}}'

            # # Return the generated response as JSON
            # response_data = {'advice': generated_response}
            # answer = json.dumps(response_data)
            self.wfile.write(json_response.encode())
        elif self.path == '/setupPDF':
            print("setup pdf request received")
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            pdfData = data.get('data')
            print("error getting pdf data")

            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=900, chunk_overlap=200,
                                                  length_function=len)
            texts = text_splitter.split_text(pdfData)

            print("error splitting data")
            # print(texts[0])
            # print("\n\n\n")
            # print(texts[1])

            embeddings = OpenAIEmbeddings()
            print("error initiazliging openAI embeddings")

            docsearch = FAISS.from_texts(texts, embeddings)
            print("error setting up docsearch")
            chain = load_qa_chain(OpenAI(temperature=0.5), chain_type="stuff")

            print("error setting up qa chain")
            # convert response to json format manually
            json_response = f'{{"response": "{"Chain setup succesfully"}"}}'

            self.wfile.write(json_response.encode())
            print("sent response for confirmation")

        elif self.path == '/askQuestion' :
            print("ask question request received")
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            ques = data.get('question')
            print("error getting question")

            docs = docsearch.similarity_search(ques)
            print("error searching doc for siimlar chunk")

            response = chain.run(input_documents=docs, question=ques)
            print("error running chain")
            # convert response to json format manually
            json_response = f'{{"response": "{response}"}}'
            self.wfile.write(json_response.encode())
            print("sent reponse for question")

        else:
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            msg = 'Python is running on Qoddi! You requested %s' % (self.path)
            self.wfile.write(msg.encode())


port = int(os.getenv('PORT', 8080))
print('Listening on port %s' % (port))
httpd = socketserver.TCPServer(('', port), Handler)
httpd.serve_forever()