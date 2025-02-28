import sys
import os
import webbrowser
from PySide6.QtWidgets import QApplication,QScrollArea, QMainWindow, QVBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog, QLineEdit, QWidget, QHBoxLayout, QFrame, QListWidget, QMessageBox
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMessageBox
import fitz  # PyMuPDF
import pickle  # For saving and loading the vector store
from create_qa_chain_and_vectorstore import create_qa_chain,create_or_load_vectorstore,extract_text_from_pdf,clean_extracted_text,view_pdf
import shutil
#from PyQt5 import QtGui
import datetime
from langchain.chat_models import ChatOpenAI

# include regular expression in the code... make sure that you identify the page number of the pdf you are reading and add that to the meta data so you can later print the context with the page number where the context exists.... 




class ChatbotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Agent Chatbot")
        self.setGeometry(100, 100, 1200, 600)
        self.file_paths = {}
        self.expiry_date = datetime.date(2026, 2, 15)  # Set your expiry date here (YYYY, MM, DD)
        self.check_expiry()  
        self.init_ui()

    def show_info(self, message):
        """Show an informational message box."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Information")
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()
    
    def show_error(self, message):
        """Show an error message box."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()


    def check_expiry(self):
        """Check how many days are left until the expiry date."""
        today = datetime.date.today()
        remaining_days = (self.expiry_date - today).days

        if remaining_days > 0:
            # Show a message to let the user know how many days are left
            message = f"Your application is valid for {remaining_days} day(s)."
            self.show_info(message)
        elif remaining_days == 0:
            # Warn the user that today is the last valid day
            message = "Warning: Your application expires today!"
            self.show_warning(message)
        else:
            # Warn the user that the application has expired
            message = "Error: Your application has expired!"
            self.show_error(message)
            sys.exit()

    def show_warning(self, message):
    # Create a warning message box
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)  # Set the icon to warning
        msg.setWindowTitle("Warning")  # Set the title of the message box
        msg.setText(message)  # Set the warning message
        msg.setStandardButtons(QMessageBox.Ok)  # Add an OK button to close the message box
        msg.exec()  # Display the message box

    def init_ui(self):
    # Main layout container
        main_layout = QVBoxLayout()  # A layout manager provided by PyQt or PySide. It arranges child widgets in a vertical (top-to-bottom) stack.

    # Header Section
        header_frame = QFrame(self)
        header_frame.setStyleSheet("background-color: #4285f4; height: 50px;") 
        header_layout = QHBoxLayout(header_frame)
        header_label = QLabel("Internal Audit Assistant", self)
        header_label.setStyleSheet("color: #D3D3D3; font: bold 16px;")
        header_layout.addWidget(header_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(header_frame)

    # Chat and File List Section
        content_frame = QWidget(self)
        content_layout = QHBoxLayout(content_frame)

    # File list section (1)
        file_frame = QFrame(self)
        file_frame.setStyleSheet("background-color: #A9A9A9; padding: 10px;")
        file_list_layout = QVBoxLayout(file_frame)
        self.file_list_widget = QListWidget(file_frame)
        self.file_list_widget.setStyleSheet("background-color: #FFFFFF; font: 12px; color: black;")
        self.file_list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.file_list_widget.itemDoubleClicked.connect(self.open_file)
        self.file_list_widget.setFixedWidth(400)
        self.file_list_widget.setFixedHeight(450)
        file_list_layout.addWidget(self.file_list_widget)

        self.populate_file_list() # to add the current pdfs in the current directory to file list

        upload_button = QPushButton("Upload", self)
        upload_button.setStyleSheet("background-color: #60819C; color: white; padding: 10px;")
        upload_button.clicked.connect(self.upload_file)
        file_list_layout.addWidget(upload_button)

        remove_button = QPushButton("Remove", self)
        remove_button.setStyleSheet("background-color: #92756C; color: white; padding: 10px;")
        remove_button.clicked.connect(self.remove_file)
        file_list_layout.addWidget(remove_button)

        train_button = QPushButton("Train Model", self)
        train_button.setStyleSheet("background-color: #748875; color: white; padding: 10px;")
        

        train_button.clicked.connect(self.train_model)
        file_list_layout.addWidget(train_button)

        content_layout.addWidget(file_frame)

    # Chat window section
        chat_frame = QFrame(self)
        chat_frame.setStyleSheet("background-color:#A9A9A9; padding: 10px;")
        chat_layout = QVBoxLayout(chat_frame)
        self.chat_window = QTextEdit(self)
        self.chat_window.setStyleSheet("background-color: #FFFFFF; font: 12px; color: black;")
        self.chat_window.setReadOnly(True) 
        self.chat_window.setFixedWidth(350)
        chat_layout.addWidget(self.chat_window)

         # Create the chat input field (QTextEdit)
        self.chat_input = QTextEdit(self)
        self.chat_input.setFixedHeight(90)  # Set a fixed height
        self.chat_input.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show scrollbar when needed

        # Set placeholder text, font, and custom styling
        self.chat_input.setPlaceholderText("Train the model...")
        self.chat_input.setEnabled(False)
        self.chat_input.setStyleSheet("background-color: #D3D3D3; font: 12px; color: black; padding: 5px;")

        # Create a layout and add the chat input to it
        layout = QVBoxLayout(self)
        layout.addWidget(self.chat_input)

        # Set the layout of the window
        self.setLayout(layout)

        # Set the layout of the window
        self.setLayout(layout)

        send_button = QPushButton(self)
        send_button.setIcon(QIcon("play_icon2.webp"))  # Ensure the play icon exists
        send_button.setIconSize(QSize(30, 30))  # Resize the icon to fit button
        send_button.setStyleSheet("background-color: transparent; border: none; padding: 10px;")
        send_button.clicked.connect(self.send_message)

    

    # Layout for input and buttons
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)
        chat_layout.addLayout(input_layout)
        content_layout.addWidget(chat_frame)

        main_layout.addWidget(content_frame)

    
        # pdf view section
        pdf_frame = QFrame(self)
        pdf_frame.setStyleSheet("background-color: #A9A9A9; padding: 10px;")
        self.pdf_layout = QVBoxLayout(pdf_frame)
        self.pdf_display_label = QLabel(self)
       # self.pdf_display_label.setStyleSheet("background-color:#D2B48C  ; font: 12px; color: black;")
        self.pdf_display_label.setFixedWidth(350)
        self.pdf_layout.addWidget(self.pdf_display_label)
        
        scroll_area = QScrollArea(self)  # Create a scroll area with the current widget as parent
        scroll_area.setWidget(pdf_frame)  # Set the pdf_frame inside the scroll area
        scroll_area.setWidgetResizable(True) 

        # Add scroll area to the content layout, not the frame directly
        content_layout.addWidget(scroll_area)
        
    # Add pdf_frame to the layout (important)
        

    # Set main layout to central widget
        container = QWidget(self)
        container.setLayout(main_layout)
        self.setCentralWidget(container)    
    
    def populate_file_list(self):
        # Get the current directory
        current_directory = os.getcwd()
        
        # List all files in the current directory
        for file_name in os.listdir(current_directory):
            # Check if the file is a PDF
            if file_name.lower().endswith('.pdf'):
                # Add the PDF file to the QListWidget
                self.file_list_widget.addItem(file_name)
                file_path=os.path.join(current_directory,file_name)
                self.file_paths[file_name] = file_path
    
    def append_message(self, user_message, align="left"):
        # Add a new message to the chat window
        if align == "right":
            self.chat_window.append(f"{user_message}\n")
        else:
            self.chat_window.append(f"\n{user_message}")
         
    def clear_pdf_layout(self):
        if self.pdf_layout is not None:
            # Loop through all items in the layout and remove them
            while self.pdf_layout.count():
                item = self.pdf_layout.takeAt(0)  # Take the first item
                widget = item.widget()  # Get the widget from the layout item
            
                if widget:
                    widget.deleteLater()  # Delete the widget
                else:
                # Remove spacers or other non-widget items
                    self.pdf_layout.removeItem(item)

        self.pdf_layout.update()  # Update the layout after clearing


    def send_message(self):
        user_message = self.chat_input.toPlainText()
        if user_message.strip() == "":
            return
        self.clear_pdf_layout()  # Clear all content in the layout

        self.chat_window.append(f'<p style="text-align: left;">'
                       # f'<span style="background-color: #D2B48C; border-radius: 20px; padding: 20px 25px; display: inline-block;">'
                        f'<b>You:</b> {user_message}</span></p>')   
        # invoking the model to repsond to the user
        # Step 1: saving the user message as a question for the model
        question=user_message
        #Step 2: # Create the QA chain        
        qa_chain,context = create_qa_chain(self.vectorstore,question)
        print(context)
        #step 3: asking the question from the mode
        answer=qa_chain.run(question)
        
        context_info = ""
        for doc in context:
    # Access the page number from the metadata
            page_number = doc.metadata.get("page_number", "Page number not available")
            pdf_name=  doc.metadata.get("pdf_name", "pdf name not available")
            
    
    # Construct the context info string
            print(doc.page_content)
            page_content=doc.page_content
            page_content_cleaned=clean_extracted_text(page_content)
            pixmap=view_pdf(pdf_name,page_number,page_content_cleaned)            
            image_label = QLabel(self)
            image_label.setPixmap(pixmap)
            self.pdf_layout.addWidget(image_label)

        
           # context_info += f"Page Number: {page_number}\nContent: {page_content_cleaned}\n\n"

# Append the context and answer for the user
        
        #answer = "AI Agent: " + answer + "\n\n" + "Following is the context used by the model to generate the reply:\n\n" + context_info
        answer = "AI Agent: " + answer + "\n\n" 

        #answer= answer + " (following is the context which was used by the model to generate the reply): " + context 
        print(answer)
        formatted_answer = answer.replace("\n\n", "<br><br>")
        #appedning the chatframe with model's answer
        #self.chat_window.append(f'<p style="text-align: left;">'
         #               f'<span style="background-color: #d3f9d8; border-radius: 20px; padding: 20px 25px; display: inline-block;">'
          #              f'<b>AI Agent:</b> {formatted_answer}</span></p>')
        self.chat_window.append(f'<p style="text-align: left;">'
                        f'<span style="border-radius: 20px; padding: 20px 25px; display: inline-block;">'
                        f'<b>AI Agent:</b> {formatted_answer}</span></p>')


        # Scroll to the bottom
        self.chat_window.verticalScrollBar().setValue(self.chat_window.verticalScrollBar().maximum())

        # Clear the input field
        self.chat_input.clear()

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF File", "", "PDF Files (*.pdf)")
        if file_path:
            file_name = os.path.basename(file_path)
            self.file_paths[file_name] = file_path
            self.update_file_list()
            current_directory=os.getcwd()
            destination_path=os.path.join(current_directory,os.path.basename(file_path))
            shutil.copy(file_path,destination_path)
        #delete the existing vector store
        folder_to_delete = "vectorstore.faiss"
        file_to_delete = "embeddings.pkl"

        self.delete_folder_and_file(folder_to_delete,file_to_delete)
        self.chat_input.setEnabled(False)
        self.chat_input.setPlaceholderText("Train the Model")

    def delete_folder_and_file(self,folder_name, file_name): # to delete the currnet vector store
    # Delete the folder if it exists
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            try:
                shutil.rmtree(folder_name)
                print(f"Folder '{folder_name}' deleted successfully.")
            except Exception as e:
                print(f"Error deleting folder '{folder_name}': {e}")
        else:
            print(f"Folder '{folder_name}' does not exist or is not a directory.")

    # Delete the file if it exists
        if os.path.exists(file_name) and os.path.isfile(file_name):
            try:
                os.remove(file_name)
                print(f"File '{file_name}' deleted successfully.")
            except Exception as e:
                print(f"Error deleting file '{file_name}': {e}")
        else:
            print(f"File '{file_name}' does not exist or is not a file.")





    def remove_file(self):
        selected_item = self.file_list_widget.currentItem()
        if selected_item:
            selected_file = selected_item.text()
            if selected_file in self.file_paths:
                del self.file_paths[selected_file]
                self.update_file_list()
        #delete the vector store as user is deleting a file
        folder_to_delete = "vectorstore.faiss"
        file_to_delete = "embeddings.pkl"

        self.delete_folder_and_file(folder_to_delete,file_to_delete)
        self.chat_input.setEnabled(False)
        self.chat_input.setPlaceholderText("Train the Model")




    def update_file_list(self):
        # Update file list UI
        self.file_list_widget.clear()
        for file_name in self.file_paths.keys():
            self.file_list_widget.addItem(file_name)

    def open_file(self):
        selected_item = self.file_list_widget.currentItem()
        if selected_item:
            selected_file = selected_item.text()
            if selected_file in self.file_paths:
                file_path = self.file_paths[selected_file]
                if file_path.lower().endswith(".pdf"):
                    webbrowser.open(f'file://{file_path}')
                else:
                    QMessageBox.critical(self, "Error", "Only PDF files can be opened in the viewer.")

    def train_model(self):
        # step1 : Collect the text for creating the embedding and the vector store
        combined_text={}
        if self.file_paths:
            for file_path in self.file_paths:
                file_path = self.file_paths[file_path]
                pdf_data = extract_text_from_pdf(file_path,4)
                combined_text.update(pdf_data)
                print(combined_text)
        else:
            self.show_warning("you have not added any files yet")
        #Step 2: Create the embedding and the vector store
        vectorstore, embeddings = create_or_load_vectorstore(combined_text)
        self.vectorstore=vectorstore
        self.chat_input.setEnabled(True)
        self.chat_input.setPlaceholderText("type your message...")

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotApp()
    window.show()
    sys.exit(app.exec())
