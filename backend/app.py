
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import traceback

# Load FAISS index and text mapping
faiss_index = faiss.read_index("knowledge_base.index")
with open("text_mapping.pkl", "rb") as f:
    texts = pickle.load(f)

# Initialize the FastAPI app
app = FastAPI()

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Stable and free

# Allow CORS for all domains (for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (for development purposes)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Pydantic model to handle user query input
class UserInput(BaseModel):
    query: str


# Function to query the knowledge base
def query_knowledge_base(query_text, n_results=3):
    try:
        # Step 1: Generate embedding for query
        query_embedding = embedding_model.encode([query_text])

        # Step 2: Search in FAISS index
        distances, indices = faiss_index.search(query_embedding, n_results)

        # Step 3: Retrieve and display results
        response_content = []
        for i, idx in enumerate(indices[0]):
            response_content.append({"page_content": texts[idx]})

        return response_content
    except Exception as e:
        print(f"Error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint for chatbot interaction
@app.post("/query")
async def chat_response(user_input: UserInput):
    # Get results from the knowledge base based on user input
    responses = query_knowledge_base(user_input.query)

    ## consider additional user inputs, for ex : beaches, nature walks
    # prior activites, (based on hitsory)
    #

    print("response recivied:", type(responses))

    str = ""
    for response in responses:
        str += " " + response["page_content"]  # Concatenate all responses

    print(str[:200])  # Displaying the first 200 characters of the response
    return {"response": str}

"""





# with llm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import traceback

# Load FAISS index and text mapping
faiss_index = faiss.read_index("knowledge_base.index")
with open("knowledge_base/text_mapping.pkl", "rb") as f:
    texts = pickle.load(f)

# Initialize the FastAPI app
app = FastAPI()

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Stable and free

# Initialize Ollama model
llm = ChatOllama(
    model="gemma2:2b",
    temperature=0,
)

# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""You are an assistant for travel guide tasks.
    Use the following documents to answer the question.
    Be Polite to the user, dont mention anywhere that external websites should be referred,don't mention documents are used to train the engine  
    Answer in 2 paragraphs of 6 sentences in each at maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

# FAISS Retriever
class FAISSRetriever:
    def __init__(self, faiss_index, texts):
        self.faiss_index = faiss_index
        self.texts = texts

    def invoke(self, query, n_results=3):
        # Generate embeddings for the query
        query_embedding = embedding_model.encode([query])

        # Search in FAISS index
        distances, indices = self.faiss_index.search(query_embedding, n_results)

        # Retrieve the corresponding documents
        documents = []
        for idx in indices[0]:
            if idx < len(self.texts):
                documents.append(Document(page_content=self.texts[idx]))
        return documents

retriever = FAISSRetriever(faiss_index, texts)

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        try:
            # Retrieve relevant documents
            documents = self.retriever.invoke(question)
            # Extract content from retrieved documents
            doc_texts = "\n".join([doc.page_content for doc in documents])
            # Generate the answer using the LLM and RAG chain
            answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
            return answer
        except Exception as e:
            print(f"Error in RAG application: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing query.")

# Allow CORS for all domains (for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (for development purposes)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Pydantic model to handle user query input
class UserInput(BaseModel):
    query: str

# API endpoint for chatbot interaction
@app.post("/query")
async def chat_response(user_input: UserInput):
    try:
        # Initialize RAG application
        rag_application = RAGApplication(retriever, rag_chain)

        print("reached ckpts 1")

        # Run the RAG application pipeline with the user's query
        answer = rag_application.run(user_input.query)

        print(answer)



        # Return the final answer to the front end
        return {"response": answer}

    except Exception as e:
        print(f"Error handling request: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")


"""
async def chat_response(user_input: UserInput):
    # Get results from the knowledge base based on user input
    responses = query_knowledge_base(user_input.query)

    ## consider additional user inputs, for ex : beaches, nature walks
    # prior activites, (based on hitsory)
    #

    print("response recivied:", type(responses))

    str = ""
    for response in responses:
        str += " " + response["page_content"]  # Concatenate all responses

    print(str[:200])  # Displaying the first 200 characters of the response
    return {"response": str}

"""