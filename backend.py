from flask import Flask, send_from_directory,request,jsonify,url_for
from flask_cors import CORS
import fitz 
import os,re

import joblib
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS,VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import shutil
import json
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai._exceptions import OpenAIError
import re

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import logging

"""
logging.basicConfig(filename="debug_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",  # Overwrites file on each run
)
"""
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
vectorStore=None
total_pdf=0
BASE_URL = "https://sajjadalinoor.vercel.app";

UPLOADS_FOLDER = "uploads" 
openai_api_key = os.getenv("OPENAI_API_KEY_S")
print(app.url_map)



class Document:
    def __init__(self, page_content, metadata=None, doc_id=None):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.id = doc_id or str(id(self))  # Generate a unique ID if not provided

    def set_page_number(self, page_number):
        """Set the page number in the metadata."""
        self.metadata["page_number"] = page_number

    def get_page_number(self):
        """Get the page number from the metadata."""
        return self.metadata.get("page_number", "Page number not available")
def separate_sentences(text):
    # Split by newline characters and filter out any empty strings
    sentences = [sentence for sentence in text.split("\n") if sentence.strip()]
    return sentences

def save_context_in_file(context_data,context,question,awnser):
    new_context = [item for item in context if len(item) > 30][:5]
    if not context_data:
        return "No data to save"

    try:
        with open("context.txt", "a", encoding="utf-8") as file:  # Open file in append mode
            for entry in context_data:
                pdf_url = entry.get("pdf_url", "Unknown PDF URL")
                page_number = entry.get("page_number", "Unknown Page")
                search_string = entry.get("searchString", "No relevant text found")

                # Format the content as per the requirement
                formatted_content = f"""
The model came up with the answer:\n"---{awnser}---"\n\n to your question:\n "---{question}---"\n\n after reading the following lines in the page numbers of the mentioned PDF:

Strings that help the model generate the response:\n
{chr(10).join(new_context)}

PDF Name:
{pdf_url}

PDF page number:
{page_number}

-------------------------
"""

                file.write(formatted_content)  # Write the formatted content to the file

    except Exception as e:
        error_message = f"Error while saving context data: {str(e)}"
        print(error_message)
        return error_message  # Return the error message
    
    return "Data saved successfully"  # Optionally return success message

def clean_extracted_text(text):
    if text is None:
        return ""
    
    # Step 1: Replace newline characters with spaces
    cleaned_text = text.replace('\n', ' ')
    
    # Step 2: Add space between lowercase and uppercase letters
    # This pattern detects transitions from lowercase to uppercase, commonly used in concatenated words
    cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)
    
    
    # Step 4: Remove any extra spaces (multiple spaces or spaces at the beginning/end)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def extract_text_from_pdf(pdf_path, target_page=None):
    pdf_text = {}
    try:
        pdf_name = pdf_path.split('/')[-1]  # Extract the PDF file name from the path
        pdf_text = {pdf_name: {}}  # Initialize the nested dictionary with the PDF name as the key
        
        # Open the PDF with PyMuPDF (fitz)
        doc = fitz.open(pdf_path)
        
        # If a specific target page is provided
        if target_page:
            try:
                if 1 <= target_page <= len(doc):  # Ensure the page exists
                    page = doc.load_page(target_page - 1)  # Zero-based indexing
                    text = page.get_text()
                    text = clean_extracted_text(text)  # Clean the extracted text
                    pdf_text[pdf_name][target_page] = text
                else:
                    print(f"Page {target_page} is out of range. This PDF has {len(doc)} pages.")
            except Exception as e:
                print(f"Error processing target page {target_page} in {pdf_name}: {e}")
        
        else:
            # Read a specific range of pages (e.g., pages 4 to 6)
            try:
                for page_number in range(3, 10):  # pages 4 to 6 (0-indexed, so pages 3,4,5)
                    page = doc.load_page(page_number)
                    page_text = page.get_text()  # Extract text for this page only
                    page_text = clean_extracted_text(page_text)  # Clean the extracted text
                    pdf_text[pdf_name][page_number + 1] = page_text  # Use 1-indexed page numbers
            except Exception as e:
                print(f"Error reading pages 4 to 6 in {pdf_name}: {e}")
        
        return pdf_text
    
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return {}    

def create_or_load_vectorstore(pdf_text, vectorstore_path="vectorstore.faiss", embeddings_path="embeddings.pkl"):

    try:
        if os.path.exists(vectorstore_path) and os.path.exists(embeddings_path):
            print("Loading existing vector store and embeddings...")

            # Load the saved embeddings parameters
            embeddings_params = joblib.load(embeddings_path)

            # Initialize embeddings using the saved API key
            embeddings = OpenAIEmbeddings(openai_api_key=embeddings_params[openai_api_key])

            # Load the saved vector store (FAISS index)
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

        else:
            print("Creating new vector store and embeddings...")
            with open("extracted_pages.txt", "w", encoding="utf-8") as f:
                for pdf_name, pages in pdf_text.items():
                    for page_number, text in pages.items():
                        f.write(f"PDF: {pdf_name} | Page: {page_number} | Text (first 200 chars): {text[:200]}\n\n")


            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

            # Create a list to store chunks with their metadata
            documents_with_page_info = []

            for pdf_name, page_content in pdf_text.items():
                # Use the loop variable 'text' directly rather than re-accessing page_content by key
                for page_number, text in page_content.items():
                    if text.strip():  # Skip empty pages
                        for chunk in text_splitter.split_text(text):
                            # Create a Document with metadata containing the page number and PDF name
                            document_with_page_info = Document(
                                page_content=chunk,
                                metadata={"pdf_name": pdf_name, "page_number": page_number}
                            )
                            documents_with_page_info.append(document_with_page_info)

            # Generate embeddings and create vector store
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.from_documents(documents_with_page_info, embeddings)

            # Save vector store and embeddings for future use
            vectorstore.save_local(vectorstore_path)  # Saving the FAISS index
            embeddings_params = {openai_api_key: embeddings.openai_api_key}
            joblib.dump(embeddings_params, embeddings_path)

        return vectorstore, embeddings

    except Exception as e:
        print("Error:", e)
"""
def create_or_load_vectorstore(pdf_text, vectorstore_path="vectorstore.faiss", embeddings_path="embeddings.pkl"):

    try:
        if os.path.exists(vectorstore_path) and os.path.exists(embeddings_path):
            print("Loading existing vector store and embeddings...")

            # Load the saved embeddings parameters
            embeddings_params = joblib.load(embeddings_path)

            # Initialize embeddings using the saved API key
            embeddings = OpenAIEmbeddings(openai_api_key=embeddings_params[openai_api_key])

            # Load the saved vector store (FAISS index)
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

        else:
            logging.info("Creating new vector store and embeddings...")

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            # Create a list to store chunks with their metadata
            documents_with_page_info = []

            for pdf_name, page_content in pdf_text.items():
                for page_number, content in page_content.items():
                    logging.debug(f"Checking content for PDF: {pdf_name}, Page: {page_number}")
                    logging.debug(f"Page Content (first 500 chars): {content[:500]}")  # Show first 500 chars

                    if content.strip():  # Skip empty pages
                        for chunk in text_splitter.split_text(content):
                            logging.debug(f"Chunk from PDF: {pdf_name}, Page: {page_number} -> {chunk[:200]}")  # Show first 200 chars
                            document_with_page_info = Document(
                                page_content=chunk,
                                metadata={"pdf_name": pdf_name, "page_number": page_number}
                            )
                            documents_with_page_info.append(document_with_page_info)

            logging.info(f"Total Chunks Created: {len(documents_with_page_info)}")

            # Generate embeddings and create vector store
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            logging.info("Generating embeddings for document chunks...")

            # Check for duplicate chunks before passing to FAISS
            unique_chunks = set(doc.page_content for doc in documents_with_page_info)
            logging.debug(f"Unique chunks count before FAISS: {len(unique_chunks)}")

            # Create FAISS vector store
            vectorstore = FAISS.from_documents(documents_with_page_info, embeddings)
            logging.info("FAISS vector store created successfully.")

            # Save vector store and embeddings for future use
            vectorstore.save_local(vectorstore_path)  # Saving the FAISS index
            embeddings_params = {"openai_api_key": openai_api_key}
            joblib.dump(embeddings_params, embeddings_path)
            logging.info("FAISS index and embeddings saved successfully.")

            return vectorstore, embeddings

    except Exception as e:
        # Handle any exceptions and display them in a messagebox
        error_message = f"An error occurred while loading or creating the vector store: {str(e)}"
        print(error_message)
"""
def get_embedding(text, model="text-embedding-ada-002"):
    """Function to get an embedding for a given text using OpenAI's API"""
    client = openai.OpenAI(api_key=openai_api_key)  # Replace with your actual API key

    response = client.embeddings.create(
        input=text,
        model=model
    )
    embedding = response.data[0].embedding

     
    return embedding
def extract_relevant_context(relevant_text, query, num_sentences=2):
    """
    Refines the relevant context by selecting the top N sentences
    that are most similar to the query.
    """
    query_embedding = get_embedding(query, model="text-embedding-ada-002")

    # Split text into sentences based on multiple delimiters
    sentences = re.split(r'[.?!]', relevant_text)

    cleaned_sentences = []
    sentence_embeddings = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # Ignore empty sentences
            # Remove non-ASCII characters
            sentence = re.sub(r'[^\x00-\x7F]+', '', sentence)
            cleaned_sentences.append(sentence)

            # Generate embedding for the sentence
            try:
                embedding = get_embedding(sentence, model="text-embedding-ada-002")
                sentence_embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for sentence: {sentence}. Skipping.")
                continue  # Skip problematic sentences

    if not sentence_embeddings:
        return "No relevant context could be extracted."

    # Convert list to NumPy array for efficient similarity computation
    sentence_embeddings = np.array(sentence_embeddings)

    # Compute cosine similarity scores
    similarity_scores = cosine_similarity([query_embedding], sentence_embeddings)[0]

    # Pair sentences with their similarity scores
    scored_sentences = list(zip(cleaned_sentences, similarity_scores))

    # Sort sentences by similarity score in descending order
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Select the top N most relevant sentences
    top_relevant_sentences = [sentence for sentence, _ in scored_sentences[:num_sentences]]

    # Format final output
    final_context = '\n'.join([f"{sentence}" for sentence in top_relevant_sentences])


    return final_context
def create_qa_chain(vectorstore, question, top_n=5):
    client = openai.OpenAI(api_key=openai_api_key)

    # Define a prompt template
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Use the following retrieved context to answer the user's query:
        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=["context", "question"]
    )

    # Set up the retrieval-augmented QA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_n})
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    try:
        # Retrieve the top N relevant documents
        search_results = retriever.get_relevant_documents(question)

        if not search_results:
            raise ValueError("No documents found for the given query.")

        # Extract document texts, filenames, and page numbers
        document_texts = []
        document_metadata = []
        unique_pdf_names = set()

        for doc in search_results:
            text = doc.page_content
            metadata = doc.metadata
            pdf_name = metadata.get("pdf_name", "Unknown PDF")
            page_number = metadata.get("page_number", "Unknown Page")
            
            document_texts.append(text)
            document_metadata.append((pdf_name, page_number))
            unique_pdf_names.add(pdf_name)
        
        # Determine the number of unique PDFs to extract context from
        num_relevant_docs = len(unique_pdf_names)
        
        # Generate embeddings for the documents
        document_embeddings = []
        for text in document_texts:
            try:
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                document_embeddings.append(embedding)
            except openai.OpenAI.OpenAIError as e:
                print(f"Error generating embedding for document: {e}")
                document_embeddings.append(None)

        # Generate the embedding for the question
        try:
            response = client.embeddings.create(
                input=question,
                model="text-embedding-ada-002"
            )
            query_embedding = response.data[0].embedding
        except openai.OpenAI.OpenAIError as e:
            print(f"Error generating embedding for query: {e}")
            query_embedding = None

        # Ensure embeddings were successfully generated
        if query_embedding is None or any(embedding is None for embedding in document_embeddings):
            raise ValueError("Error in generating embeddings. Please check the API responses.")

        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], document_embeddings)[0]

        # Find indices of the top `num_relevant_docs` documents
        #sorted_indices = similarities.argsort()[::-1]

        sorted_indices = similarities.argsort()[::-1]
        
        # Extract relevant text and metadata from multiple PDFs
        #relevant_texts = [extract_relevant_context(document_texts[i], question) for i in sorted_indices]
        relevant_texts = [extract_relevant_context(document_texts[sorted_indices[0]], question)]

        most_relevant_pdf = sorted_indices[0]
        relevant_metadata = [document_metadata[most_relevant_pdf]]  

        # Merge the contexts from multiple PDFs
        merged_relevant_text = "\n\n".join(relevant_texts)

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

        return qa_chain, merged_relevant_text, relevant_metadata

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None, None, None

    except OpenAI.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return None, None, None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None

@app.route("/api/")
def home():
    return jsonify({"message": "API is running!"})


#after loading index.html, the browser requests additional resources and these requests contain paths that does not match wthe root route("/")
# so it will be handled by the second route ("/<path:path>")
@app.route("/api/<path:path>", methods=["GET", "POST"])
def static_files(path):
    return send_from_directory("build", path)

@app.route("/api/uploads/<path:filename>")
def serve_pdf(filename):
    return send_from_directory(UPLOADS_FOLDER, filename)
"""
@app.route('/api/DeleteVS', methods=['POST']) 
def delete_vectorstore():
    # Define paths for the folders and files
    vectorstore_folder_path = "vectorstore.faiss"
    embeddings_file_path = "embeddings.pkl"
    uploads_folder_path = "uploads"  # New folder to delete

    # Deleting the vectorstore folder if it exists
    deleted_anything = False  # Flag to check if anything was deleted

    try:
        # Deleting the vectorstore folder if it exists
        if os.path.exists(vectorstore_folder_path):
            shutil.rmtree(vectorstore_folder_path)  # Remove non-empty directory
            print(f"Folder {vectorstore_folder_path} deleted successfully")
            deleted_anything = True  # Mark as deleted

        # Deleting the vectorstore file if it exists
        if os.path.exists(embeddings_file_path):
            os.remove(embeddings_file_path)
            print(f"File {embeddings_file_path} deleted successfully")
            deleted_anything = True  # Mark as deleted

        # Deleting the uploads folder if it exists
        if os.path.exists(uploads_folder_path):
            shutil.rmtree(uploads_folder_path)  # Remove non-empty directory
            print(f"Folder {uploads_folder_path} deleted successfully")
            os.makedirs(uploads_folder_path, exist_ok=True)  # Make a new upload folder
            deleted_anything = True  # Mark as deleted

        # Return success message only if something was deleted
        if deleted_anything:
            return jsonify({"message": "VectorStore and/or uploads folder deleted successfully"}), 200
        
        return jsonify({"message": "Nothing to delete"}), 204  # No Content response if nothing was deleted    

    except Exception as e:
        # Handle any exceptions that may occur
        return jsonify({"detail": f"Error deleting VectorStore or uploads folder: {str(e)}"}), 500
@app.route('/api/checkVS', methods=['GET'])  # for printing the name of PDFs in the uploads folder
def checkVS():
    current_directory = os.getcwd()
    folder_name = "vectorstore.faiss"
    file_name = "embeddings.pkl"
    
    folder_exists = os.path.isdir(os.path.join(current_directory, folder_name))
    file_exists = os.path.isfile(os.path.join(current_directory, file_name))

    if folder_exists and file_exists:
        return jsonify({"result": "yes"})
    else:
        return jsonify({"result": "no"})

    
@app.route('/api/pdf-files', methods=['GET']) #for printing the name of pdfs in the uploads folder
def get_pdf_files():
    # Path to the 'uploads' folder
    uploads_dir = "uploads"
    
    try:
        # Get a list of all files in the uploads folder
        files = os.listdir(uploads_dir)
        
        # Filter files that end with .pdf
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        
        # Return the list of PDF files as a JSON response
        return jsonify(pdf_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""
@app.route("/api/upload", methods=["GET", "POST"])
def upload_files():
    if request.method == "GET":
        return jsonify({"message": "Upload endpoint is working. Use POST to upload files."})

    if "pdfs" not in request.files:
        return jsonify({"message": "No file part in the request"}), 400
    
    files = request.files.getlist("pdfs")  # Get multiple files
    if not files:
        return jsonify({"message": "No files uploaded"}), 400

    saved_files = []
    for file in files:
        if file.filename == "":
            continue
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)  # Save file
        saved_files.append(file.filename)

    return jsonify({"message": "Files uploaded successfully", "files": saved_files})

@app.route('/api/train_model', methods=['POST'])
def train_model():
    global vectorStore
    global total_pdf
    try:    
        #step 1 collect the text 
        combined_text={}
        for filename in os.listdir("uploads"):
            if filename.lower().endswith(".pdf"):
                total_pdf+=1
                file_path=os.path.join("uploads",filename)
                try:
                     pdf_data = extract_text_from_pdf(file_path)
                     combined_text.update(pdf_data) 
                     print()
                except Exception as e:
                        print(f"Error extracting text from {file_path}: {e}")
                        continue  
        #step 2 create the embedding and the vector store
        vectorStore, embeddings = create_or_load_vectorstore(combined_text)  
        print(vectorStore) 
        return jsonify({"message": "Model trained successfully!"}), 200
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "Model could not be trained!"}), 500  # HTTP 500 for server error
    

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json  # Get JSON data from the request
    user_message = data.get('message', '')  # Extract user message
    

    #create the QA chain
    if vectorStore is None:
        train_model()  # In case if the user starts chatting without pressing the train button
        
        qa_chain, relevant_texts, document_metadata = create_qa_chain(vectorStore, user_message)  

        # Ask the question
        answer = qa_chain.run(user_message)    

        # Prepare context data for all relevant documents
        context_data = []
        for metadata in document_metadata:
            pdf_name = metadata[0]
            pdf_page = metadata[1]

            pdf_url = f"{BASE_URL}/{UPLOADS_FOLDER}/{os.path.basename(pdf_name)}"
            context_data.append({
                "page_number": pdf_page,
                "pdf_url": pdf_url,
                
            })

        # Simple bot logic (Modify this to include actual processing)
        search_strings=separate_sentences(relevant_texts)
        bot_reply = f"ChatBot: {answer}"

        response = {
            "reply": bot_reply,
            "context": context_data,
            "search_strings":search_strings
        }

        print(response)
        save_context_in_file(context_data,search_strings,user_message,bot_reply)
        return jsonify(response)
    else:   
        qa_chain, relevant_texts, document_metadata = create_qa_chain(vectorStore, user_message)  

        # Ask the question
        answer = qa_chain.run(user_message)    

        # Prepare context data for all relevant documents
        context_data = []
        for metadata in document_metadata:
            pdf_name = metadata[0]
            pdf_page = metadata[1]

            pdf_url = f"{BASE_URL}/{UPLOADS_FOLDER}/{os.path.basename(pdf_name)}"
            context_data.append({
                "page_number": pdf_page,
                "pdf_url": pdf_url,
                
            })

        # Simple bot logic (Modify this to include actual processing)
        search_strings=separate_sentences(relevant_texts)
        bot_reply = f"ChatBot: {answer}"

        response = {
            "reply": bot_reply,
            "context": context_data,
            "search_strings":search_strings
        }

        print(response)
        save_context_in_file(context_data,search_strings,user_message,bot_reply)
        return jsonify(response) 
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)

