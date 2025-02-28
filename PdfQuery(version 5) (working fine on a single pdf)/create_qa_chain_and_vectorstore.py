import joblib
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import PyPDF2
import pdfplumber
import re
from PySide6.QtGui import QPixmap, QImage, QColor


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


def create_qa_chain(vectorstore, question):
    # Define a prompt template
    prompt = PromptTemplate(
        template="""You are an AI-powered internal audit assistant designed to assist internal auditors at the Punjab Pension Fund. Your role is to provide expert guidance on all aspects of internal auditing and deliver professional support throughout the internal audit lifecycle, processes, actions, duties, and skills. You are also a domain expert in pension fund auditing and governance.

You are trained on a Retrieval-Augmented Generation (RAG) model that utilizes a knowledge base of uploaded documents to respond to user queries. When answering a query:
1. If the answer can be found in the knowledge base of uploaded documents, provide the response under a section titled **"Answer based on the knowledge base."** Ensure the response is accurate, professional, and directly relevant to the query.
2. If the answer cannot be found in the knowledge base, provide a response using your general knowledge and expertise under a section titled **"Answer based on AI model."** Clearly indicate that this response is generated based on your AI expertise and not the uploaded documents.
3. If the response requires both document-based information and AI model-based expertise, structure the response into two sections:
   - **"Answer based on the knowledge base."** Include information retrieved from the knowledge base here.
   - **"Answer based on AI model."** Include AI-generated insights or additional context here.

Your responses should:
- Be professional, concise, and actionable.
- Include detailed and step-by-step guidance for internal audit tasks when required.
- Clearly segregate information derived from the knowledge base and the AI model to ensure transparency.
- Maintain focus on improving productivity, enhancing decision-making, and ensuring accuracy in audit operations.

Your goal is to act as a reliable and expert assistant for internal auditors, helping them navigate their daily tasks, solve challenges, and perform their duties effectively. Stay consistent, professional, and user-focused in all interactions. Use the following retrieved context to answer the user's query:
        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=["context", "question"]
    )

    # Set up the retrieval-augmented QA chain 
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key="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")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    # Check available methods on the retriever object
    print(dir(retriever))

    # Retrieve the context based on the question
    #context = retriever.retrieve(question)
    # Retrieve the context using the correct method for your retriever
    context = retriever.get_relevant_documents(question)


    # Return both the qa_chain and the context
    return qa_chain, context
def view_pdf(pdf_path,page_number,search_string):
        # Get PDF file path and page number
        """
        pdf_path = self.pdf_path_input.text()
        page_number = int(self.pdf_page_number.text())   # Convert to 0-based index
        search_string = self.enter_string_to_search.text()
        """
        current_directory = os.getcwd()
        pdf_path=os.path.join(current_directory,pdf_path)
        search_string=search_string[100:400]
        page_number=page_number-1

        # Render PDF page using PyMuPDF
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_number)  # Load the specified page
        print(page)
        # Search for the string in the page
        search_instances = page.search_for(search_string)

        # If there are matches, highlight them
        for inst in search_instances:
            # Highlight matching text by drawing a rectangle around the found string
            page.draw_rect(inst, color=(1, 1, 0), width=1)  # Yellow rectangle with 1px width

        # Convert the page to a pixmap after highlighting
        pix = page.get_pixmap()

        # Convert Pixmap to QImage
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        
        # Convert the QImage to QPixmap for display
        pixmap = QPixmap.fromImage(img)
        
        
        return pixmap


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
    
    pdf_name = pdf_path.split('/')[-1]  # Extract the PDF file name from the path
    pdf_text = {pdf_name: {}}  # Initialize the nested dictionary with the PDF name as the key
    text=""
    # Open the PDF with pdfplumber
    doc=fitz.open(pdf_path)
        # If a specific target page is provided
    if target_page:
            if 1 <= target_page <= len(doc):  # Ensure the page exists
                page = doc.load_page(target_page - 1)  # Zero-based indexing
                text+=page.get_text()
                text=clean_extracted_text(text)
                
                pdf_text[pdf_name][target_page] =  text
            else:
                print(f"Page {target_page} is out of range. This PDF has {len(doc)} pages.")
    else:
            # Read a specific range of pages (e.g., pages 4 to 6)
            
        
        for page_number in range(3, 6):  # pages 4 to 6 (0-indexed, so 3:6)
            page=doc.load_page(page_number)
            text+=page.get_text()
            pdf_text[pdf_name][page_number + 1] = text  # 1-indexed page numbers
    return pdf_text

def create_or_load_vectorstore(pdf_text, vectorstore_path="vectorstore.faiss", embeddings_path="embeddings.pkl"):
    
    if os.path.exists(vectorstore_path) and os.path.exists(embeddings_path):
        print("Loading existing vector store and embeddings...")

        # Load the saved embeddings parameters
        embeddings_params = joblib.load(embeddings_path)

        # Initialize embeddings using the saved API key
        embeddings = OpenAIEmbeddings(openai_api_key=embeddings_params["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"])

        # Load the saved vector store (FAISS index)
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    else:
        print("Creating new vector store and embeddings...")
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        # Create a list to store chunks with their metadata
        documents_with_page_info = []

        for pdf_name,page_content in pdf_text.items():
            for page_number,content in page_content.items():
                text=page_content[page_number]
                if text.strip():  # Skip empty pages
                    for chunk in text_splitter.split_text(text):
                    # Create a Document with metadata containing the page number and PDF name
                        document_with_page_info = Document(
                            page_content=chunk,
                            metadata={"pdf_name": pdf_name, "page_number": page_number}
                        )
                        documents_with_page_info.append(document_with_page_info)

        # Generate embeddings and create vector store
        embeddings = OpenAIEmbeddings(openai_api_key="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")
        vectorstore = FAISS.from_documents(documents_with_page_info, embeddings)

        # Save vector store and embeddings for future use
        vectorstore.save_local(vectorstore_path)  # Saving the FAISS index
        embeddings_params = {"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": embeddings.openai_api_key}
        joblib.dump(embeddings_params, embeddings_path)

    return vectorstore, embeddings

