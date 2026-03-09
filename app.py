from flask import Flask, request, session, redirect, send_file, render_template_string, g, jsonify
import sqlite3
import os
import threading
from email.mime.text import MIMEText
import smtplib
from werkzeug.security import generate_password_hash, check_password_hash
from sqlite3 import IntegrityError
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import traceback
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import openai
import json
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from wordcloud import WordCloud

# Load environment variables
load_dotenv()

# ===========================
# LangChain Imports
# ===========================
try:
    from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain.chains import ConversationalRetrievalChain
    LANGCHAIN_AVAILABLE = True
    print("LangChain imported successfully")
except ImportError as e:
    print(f" LangChain import error: {e}")
    LANGCHAIN_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")

# ================= CONFIGURATION =================
UPLOAD_FOLDER = "uploads"
VECTORSTORE_FOLDER = "./vectorstore"
DATABASE = "users.db"
REPORTS_FOLDER = "reports"

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(VECTORSTORE_FOLDER, 'db_chroma_Job'), exist_ok=True)
os.makedirs(os.path.join(VECTORSTORE_FOLDER, 'db_chroma_Resume'), exist_ok=True)

# ================= AZURE OPENAI CONFIGURATION =================
api_base = os.getenv("API_BASE")
api_type = os.getenv("API_TYPE", "azure")
api_version = os.getenv("API_VERSION", "2023-05-15")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
embeddings_deployment = os.getenv("embeddings_deployment")
chat_completion_deployment = os.getenv("chat_completion_deployment")
temperature = float(os.getenv("TEMPERATURE", 0.7))
embeddings_chunk_size = int(os.getenv("EMBEDDINGS_CHUNK_SIZE", 16))

# Print configuration status for debugging
print("=" * 50)
print("🔧 AZURE OPENAI CONFIGURATION CHECK:")
print(f"API_BASE: {' Set' if api_base else ' Missing'}")
print(f"API_KEY: {' Set' if api_key else ' Missing'}")
print(f"API_TYPE: {api_type}")
print(f"API_VERSION: {api_version}")
print(f"embeddings_deployment: {' Set' if embeddings_deployment else ' Missing'}")
print(f"chat_completion_deployment: {' Set' if chat_completion_deployment else ' Missing'}")
print("=" * 50)

# Check if chatbot should be enabled
CHATBOT_ENABLED = False
if all([api_key, api_base, embeddings_deployment, chat_completion_deployment]) and LANGCHAIN_AVAILABLE:
    CHATBOT_ENABLED = os.getenv("CHATBOT_ENABLED", "False").lower() == "true"
    if CHATBOT_ENABLED:
        print(" Chatbot will be enabled")
    else:
        print(" Chatbot is disabled in .env (CHATBOT_ENABLED=False)")
else:
    print(" Chatbot cannot be enabled due to missing configuration")

# Set OpenAI configuration
if all([api_key, api_base, api_version]):
    openai.api_type = api_type
    openai.api_version = api_version
    openai.api_base = api_base
    openai.api_key = api_key
    print(" OpenAI configuration set")

# Vector store paths
DB_CHROMA_JOB_PATH = os.path.join(VECTORSTORE_FOLDER, 'db_chroma_Job')
DB_CHROMA_RESUME_PATH = os.path.join(VECTORSTORE_FOLDER, 'db_chroma_Resume')

# ================= EMAIL CONFIG =================
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "your_email@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your_app_password")
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "False").lower() == "true"

def send_email(to_email, subject, body):
    """Send email notification"""
    if not EMAIL_ENABLED:
        print(f"Email not sent (disabled): To: {to_email}, Subject: {subject}")
        return
    try:
        msg = MIMEText(body)
        msg["From"] = EMAIL_SENDER
        msg["To"] = to_email
        msg["Subject"] = subject
        
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f" Email sent to {to_email}")
    except Exception as e:
        print(f" Failed to send email: {e}")

# ================= DATABASE MANAGEMENT =================
def get_db():
    """Get database connection for current thread"""
    if not hasattr(g, 'database'):
        g.database = sqlite3.connect(DATABASE, check_same_thread=False)
        g.database.row_factory = sqlite3.Row
    return g.database

def init_db():
    """Initialize database with proper connection"""
    db = sqlite3.connect(DATABASE, check_same_thread=False)
    c = db.cursor()
    
    # Create tables
    c.execute("""
    CREATE TABLE IF NOT EXISTS candidates(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        qualification TEXT,
        skills TEXT,
        resume_path TEXT,
        notify_email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS recruiters(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        company TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS jobs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recruiter_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        location TEXT NOT NULL,
        salary TEXT,
        skills TEXT,
        company TEXT NOT NULL,
        views INTEGER DEFAULT 0,
        applications_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recruiter_id) REFERENCES recruiters(id) ON DELETE CASCADE
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS applications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        candidate_id INTEGER NOT NULL,
        similarity_score REAL DEFAULT 0,
        resume_score REAL DEFAULT 0,
        status TEXT DEFAULT 'pending',
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(job_id, candidate_id),
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
        FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS notifications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id INTEGER NOT NULL,
        message TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_read INTEGER DEFAULT 0,
        FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
    )""")
    
    c.execute("""
    CREATE TABLE IF NOT EXISTS analytics(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recruiter_id INTEGER NOT NULL,
        report_type TEXT NOT NULL,
        report_data TEXT,
        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recruiter_id) REFERENCES recruiters(id) ON DELETE CASCADE
    )""")
    
    # Add rejection_reason column if not exists
    try:
        c.execute("ALTER TABLE applications ADD COLUMN rejection_reason TEXT")
    except sqlite3.OperationalError:
        pass
    
    # Add columns to jobs if not exists
    try:
        c.execute("ALTER TABLE jobs ADD COLUMN views INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    
    try:
        c.execute("ALTER TABLE jobs ADD COLUMN applications_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    
    # Create indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_applications_job ON applications(job_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_applications_candidate ON applications(candidate_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_notifications_candidate ON notifications(candidate_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_analytics_recruiter ON analytics(recruiter_id)")
    
    db.commit()
    db.close()

@app.teardown_appcontext
def close_db(error):
    """Close database connection at the end of request"""
    if hasattr(g, 'database'):
        g.database.close()

# Initialize database on startup
with app.app_context():
    init_db()

# ================= HELPER FUNCTIONS =================
def is_logged_in(user_type=None):
    """Check if user is logged in, optionally check type"""
    if 'user' not in session or 'type' not in session:
        return False
    if user_type and session['type'] != user_type:
        return False
    return True

def similarity(a, b):
    """Calculate text similarity between two strings"""
    if not a or not b:
        return 0.0
    try:
        tfidf = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    except:
        return 0.0

def resume_similarity(resume_path, job_skills):
    """Calculate similarity between resume and job skills"""
    if not resume_path or not os.path.exists(resume_path):
        return 0.0
    try:
        reader = PdfReader(resume_path)
        text = " ".join(p.extract_text() or "" for p in reader.pages)
        return similarity(text, job_skills)
    except Exception as e:
        print(f"Error reading resume: {e}")
        return 0.0

def add_notification(candidate_id, message):
    """Add notification for candidate"""
    db = get_db()
    try:
        db.execute("""
        INSERT INTO notifications(candidate_id, message)
        VALUES(?, ?)
        """, (candidate_id, message))
        
        # Get candidate's notification email
        cand = db.execute("""
        SELECT email, notify_email FROM candidates WHERE id=?
        """, (candidate_id,)).fetchone()
        
        if cand and cand['notify_email']:
            send_email(cand['notify_email'], "Job Portal Notification", message)
        
        db.commit()
    except Exception as e:
        print(f"Error adding notification: {e}")
        db.rollback()

def increment_job_views(job_id):
    """Increment view count for a job"""
    db = get_db()
    try:
        db.execute("UPDATE jobs SET views = views + 1 WHERE id = ?", (job_id,))
        db.commit()
    except:
        pass

def check_vector_store(path, name):
    """Check if vector store exists and has content"""
    if not os.path.exists(path):
        print(f" {name} path does not exist: {path}")
        return False
    
    chroma_db = os.path.join(path, "chroma.sqlite3")
    if not os.path.exists(chroma_db):
        print(f" {name} chroma.sqlite3 not found")
        return False
    
    try:
        if CHATBOT_ENABLED and LANGCHAIN_AVAILABLE and all([api_key, api_base, embeddings_deployment]):
            embeddings = AzureOpenAIEmbeddings(
                deployment=embeddings_deployment,
                api_key=api_key,
                azure_endpoint=api_base,
                openai_api_version=api_version,
                openai_api_type=api_type
            )
            db = Chroma(persist_directory=path, embedding_function=embeddings)
            count = len(db.get()['ids']) if db.get() else 0
            print(f" {name} has {count} documents")
            return count > 0
        else:
            print(f" Cannot check {name} - Chatbot not fully configured")
            return False
    except Exception as e:
        print(f" Error checking {name}: {e}")
        return False

# Initialize vector stores on startup
print("\n Initializing vector stores...")
job_exists = check_vector_store(DB_CHROMA_JOB_PATH, "Job vector store")
resume_exists = check_vector_store(DB_CHROMA_RESUME_PATH, "Resume vector store")

# ================= CHATBOT FUNCTIONS =================
def chatbot(query, chat_history, DB_Chroma_PATH, user_context=None):
    """Process chat queries using RAG (Retrieval Augmented Generation)
    
    Args:
        query: User's question
        chat_history: List of previous exchanges
        DB_Chroma_PATH: Path to vector store (jobs or resumes)
        user_context: Optional context about the user
    
    Returns:
        Response string from the AI
    """
    
    # Check if chatbot is enabled
    if not CHATBOT_ENABLED or not LANGCHAIN_AVAILABLE:
        return " Chatbot is not configured. Please check your Azure OpenAI settings and .env file."
    
    # Check if vector store exists
    if not os.path.exists(DB_Chroma_PATH):
        return f" Vector store not found at {DB_Chroma_PATH}. Please add some data first."
    
    chroma_db_path = os.path.join(DB_Chroma_PATH, "chroma.sqlite3")
    if not os.path.exists(chroma_db_path):
        return f" Vector store database not found. Please add some data first."
    
    try:
        print(f"\n Processing query: '{query}'")
        print(f" Using vector store: {DB_Chroma_PATH}")
        
        # Get environment variables with defaults
        temperature = float(os.getenv("TEMPERATURE", 0.7))
        chat_completion_deployment = os.getenv("chat_completion_deployment")
        embeddings_chunk_size = int(os.getenv("EMBEDDINGS_CHUNK_SIZE", 16))
        
        # Validate required variables
        if not all([api_key, api_base, embeddings_deployment, chat_completion_deployment]):
            return " Chatbot configuration error: Missing Azure OpenAI credentials"
        
        # Initialize embeddings with direct API key
        embeddings = AzureOpenAIEmbeddings(
            deployment=embeddings_deployment,
            api_key=api_key,
            azure_endpoint=api_base,
            openai_api_version=api_version,
            openai_api_type=api_type,
            chunk_size=embeddings_chunk_size
        )
        
        # Load vector store
        db = Chroma(persist_directory=DB_Chroma_PATH, embedding_function=embeddings)
        
        # Check if vector store has any documents
        try:
            all_docs = db.get()
            doc_count = len(all_docs['ids']) if all_docs and 'ids' in all_docs else 0
            print(f" Vector store has {doc_count} total chunks")
            
            if doc_count == 0:
                return " The vector store is empty. Please add some data first."
        except Exception as e:
            print(f" Could not get document count: {e}")
        
        # Initialize LLM
        llm = AzureChatOpenAI(
            temperature=temperature,
            api_key=api_key,
            azure_endpoint=api_base,
            openai_api_version=api_version,
            openai_api_type=api_type,
            deployment_name=chat_completion_deployment
        )
        
        # Determine if we're in job or resume context
        is_job_context = 'job' in DB_Chroma_PATH.lower()
        is_resume_context = 'resume' in DB_Chroma_PATH.lower()
        
        query_lower = query.lower()
        
        # ========== SPECIAL HANDLING FOR LIST QUERIES ==========
        # Handle "all jobs" type queries
        if is_job_context and any(word in query_lower for word in 
            ['all jobs', 'list jobs', 'show jobs', 'available jobs', 'what jobs', 'show me jobs']):
            try:
                all_docs = db.get()
                if all_docs and 'documents' in all_docs and all_docs['documents']:
                    # Use a set to track unique jobs
                    unique_jobs = {}
                    for i, doc in enumerate(all_docs['documents']):
                        metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                        title = metadata.get('title', 'Unknown')
                        company = metadata.get('company', 'Unknown')
                        location = metadata.get('location', 'Unknown')
                        
                        # Create a unique key
                        job_key = f"{title}_{company}"
                        
                        # Only add if not seen before
                        if job_key not in unique_jobs:
                            unique_jobs[job_key] = {
                                'title': title,
                                'company': company,
                                'location': location
                            }
                    
                    if unique_jobs:
                        jobs_list = []
                        for job in unique_jobs.values():
                            jobs_list.append(f"• **{job['title']}** at **{job['company']}** ({job['location']})")
                        
                        response = "##  Available Jobs\n\n" + "\n".join(jobs_list)
                        
                        total_unique = len(unique_jobs)
                        total_all = len(all_docs['documents'])
                        if total_all > total_unique:
                            response += f"\n\n*({total_unique} unique jobs shown - {total_all - total_unique} duplicates filtered)*"
                        
                        return response
            except Exception as e:
                print(f"Error getting all jobs: {e}")
        
        # Handle "all candidates" type queries
        if is_resume_context and any(word in query_lower for word in 
            ['all candidates', 'list candidates', 'show candidates', 'available candidates', 'who applied']):
            try:
                all_docs = db.get()
                if all_docs and 'documents' in all_docs and all_docs['documents']:
                    # Track unique candidates
                    unique_candidates = {}
                    for i, metadata in enumerate(all_docs['metadatas']):
                        if metadata:
                            name = metadata.get('candidate_name', None)
                            source = metadata.get('source_document', 'Unknown')
                            cand_id = metadata.get('candidate_id', 'Unknown')
                            
                            # Extract name from filename if not in metadata
                            if not name or name == 'Unknown':
                                name = os.path.splitext(source)[0].replace('_', ' ').replace('-', ' ')
                            
                            if name not in unique_candidates:
                                unique_candidates[name] = {
                                    'source': source,
                                    'id': cand_id
                                }
                    
                    if unique_candidates:
                        candidates_list = []
                        for name, info in unique_candidates.items():
                            candidates_list.append(f"• **{name}** (Resume: {info['source']})")
                        
                        response = "## 👥 Candidates Who Have Applied\n\n" + "\n".join(candidates_list)
                        response += f"\n\n*Total unique candidates: {len(unique_candidates)}*"
                        
                        return response
            except Exception as e:
                print(f"Error getting all candidates: {e}")
        
        # ========== REGULAR QUERY PROCESSING ==========
        
        # Create retriever with appropriate number of documents
        retriever = db.as_retriever(
            search_kwargs={'k': 15}  # Get enough documents for context
        )
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=False,  # Set to True for debugging
            return_source_documents=True
        )
        
        # Format chat history correctly
        formatted_history = []
        for chat in chat_history[-5:]:  # Last 5 exchanges
            if isinstance(chat, tuple) and len(chat) == 2:
                formatted_history.append(chat)
            elif isinstance(chat, dict) and 'question' in chat and 'response' in chat:
                formatted_history.append((chat['question'], chat['response']))
        
        # Enhance query with context if provided
        enhanced_query = query
        if user_context:
            enhanced_query = f"{user_context}\n\nQuestion: {query}"
        
        # Get response from chain
        result = chain({
            "question": enhanced_query,
            "chat_history": formatted_history
        })
        
        # ========== ENHANCE RESPONSE WITH SOURCES ==========
        answer = result["answer"]
        
        if result.get("source_documents"):
            # Collect different types of sources
            job_sources = set()
            candidate_sources = set()
            file_sources = set()
            
            for doc in result["source_documents"]:
                metadata = doc.metadata
                
                # Job-related metadata
                if metadata.get("title"):
                    company = metadata.get('company', 'Unknown Company')
                    job_sources.add(f"**{metadata['title']}** at {company}")
                
                # Candidate-related metadata  
                elif metadata.get("candidate_name"):
                    name = metadata['candidate_name']
                    candidate_sources.add(f"**{name}**")
                
                # File-based metadata
                elif metadata.get("source_document"):
                    source = metadata["source_document"]
                    # Try to extract a readable name
                    if is_resume_context:
                        # For resumes, try to get candidate name
                        name = metadata.get('candidate_name', None)
                        if not name:
                            # Extract from filename
                            name = os.path.splitext(source)[0].replace('_', ' ').replace('-', ' ')
                        candidate_sources.add(f"**{name}**")
                    else:
                        file_sources.add(source)
            
            # Build sources section
            sources_text = []
            
            if candidate_sources:
                sources_text.append(f" **Candidates:** {', '.join(sorted(candidate_sources)[:5])}")
                if len(candidate_sources) > 5:
                    sources_text[-1] += f" and {len(candidate_sources)-5} more"
            
            if job_sources:
                sources_text.append(f" **Jobs:** {', '.join(sorted(job_sources)[:3])}")
                if len(job_sources) > 3:
                    sources_text[-1] += f" and {len(job_sources)-3} more"
            
            if file_sources and not (candidate_sources or job_sources):
                sources_text.append(f" **Sources:** {', '.join(file_sources[:3])}")
            
            if sources_text:
                answer += "\n\n---\n" + "\n".join(sources_text)
        
        print(f" Response generated ({len(answer)} characters)")
        return answer
        
    except Exception as e:
        print(f" Chatbot error: {str(e)}")
        traceback.print_exc()
        
        # Return user-friendly error message
        error_msg = f" Sorry, I encountered an error: {str(e)[:100]}"
        if "Authentication" in str(e) or "authorization" in str(e).lower():
            error_msg = " Authentication error. Please check your Azure OpenAI API key."
        elif "deployment" in str(e).lower():
            error_msg = " Deployment error. Please check your Azure OpenAI deployment names."
        elif "rate limit" in str(e).lower():
            error_msg = " Rate limit exceeded. Please try again in a moment."
        
        return error_msg
def extractpdf(pdffile, chunksize, overlap, candidate_name=None, candidate_id=None):
    """Extract text from PDF and add to vector store with candidate info"""
    if not CHATBOT_ENABLED or not LANGCHAIN_AVAILABLE:
        print("Chatbot disabled, skipping vector store update")
        return None
    
    try:
        DB_CHROMA_PATH = os.path.join(VECTORSTORE_FOLDER, 'db_chroma_Resume')
        os.makedirs(DB_CHROMA_PATH, exist_ok=True)
        
        currentfilename = os.path.basename(pdffile)
        
        # If candidate_name not provided, try to extract from filename
        if not candidate_name:
            # Remove extension and underscores/hyphens
            candidate_name = os.path.splitext(currentfilename)[0].replace('_', ' ').replace('-', ' ')
        
        docswithmeta = []
        
        # Read PDF
        reader = PdfReader(pdffile)
        full_text = []
        for i in range(len(reader.pages)):
            text = reader.pages[i].extract_text()
            if text and text.strip():
                full_text.append(text)
                docswithmeta.append(Document(
                    page_content=text,
                    metadata={
                        "source_document": currentfilename,
                        "candidate_name": candidate_name,
                        "candidate_id": str(candidate_id) if candidate_id else "unknown",
                        "pageno": i + 1,
                        "type": "resume"
                    }
                ))
        
        if not docswithmeta:
            print(f"No text extracted from {pdffile}")
            return None
        
        # Also try to extract name from resume content (first few lines)
        if candidate_name == os.path.splitext(currentfilename)[0].replace('_', ' ').replace('-', ' '):
            # Try to find name in first page
            first_page = full_text[0] if full_text else ""
            lines = first_page.split('\n')[:5]  # First 5 lines
            for line in lines:
                line = line.strip()
                if line and len(line.split()) <= 4 and not any(word in line.lower() for word in ['resume', 'cv', 'curriculum', 'vitae', 'email', 'phone']):
                    # Looks like a name
                    candidate_name = line
                    break
        
        # Split text with appropriate chunk size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunksize,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = text_splitter.split_documents(docswithmeta)
        
        # Update all chunks with the candidate name
        for doc in docs:
            doc.metadata['candidate_name'] = candidate_name
            doc.metadata['candidate_id'] = str(candidate_id) if candidate_id else "unknown"
        
        print(f"Split into {len(docs)} chunks for candidate: {candidate_name}")
        
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            deployment=embeddings_deployment,
            api_key=api_key,
            azure_endpoint=api_base,
            openai_api_version=api_version,
            openai_api_type=api_type,
            chunk_size=chunksize
        )
        
        # Create or update vector store
        if os.path.exists(os.path.join(DB_CHROMA_PATH, "chroma.sqlite3")):
            db = Chroma(
                persist_directory=DB_CHROMA_PATH, 
                embedding_function=embeddings
            )
            db.add_documents(docs)
            print(f" Added {len(docs)} chunks to existing vector store")
        else:
            db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=DB_CHROMA_PATH
            )
            print(f" Created new vector store with {len(docs)} chunks")
        
        db.persist()
        
        # Verify the data was added
        try:
            test_results = db.similarity_search(candidate_name, k=1)
            print(f" Verification: Found {len(test_results)} documents for candidate {candidate_name}")
        except:
            pass
        
        return db
        
    except Exception as e:
        print(f" Error extracting PDF: {e}")
        traceback.print_exc()
        return None
def add_job_to_vectorstore(job_data):
    """Add job posting to vector store"""
    if not CHATBOT_ENABLED or not LANGCHAIN_AVAILABLE:
        print("Chatbot disabled, skipping job vector store update")
        return None
    
    try:
        DB_CHROMA_PATH = os.path.join(VECTORSTORE_FOLDER, 'db_chroma_Job')
        os.makedirs(DB_CHROMA_PATH, exist_ok=True)
        
        # Create rich job description text with clear formatting
        text = f"""
JOB TITLE: {job_data['title']}
COMPANY: {job_data['company']}
LOCATION: {job_data['location']}
SALARY: {job_data.get('salary', 'Not specified')}
REQUIRED SKILLS: {job_data.get('skills', 'Not specified')}

JOB DESCRIPTION:
{job_data['description']}

This is a job posting for a {job_data['title']} position at {job_data['company']}.
"""
        
        # Create document with comprehensive metadata
        doc = Document(
            page_content=text,
            metadata={
                "job_id": str(job_data.get('job_id', '')),
                "recruiter_id": str(job_data['recruiter_id']),
                "title": job_data['title'],
                "company": job_data['company'],
                "location": job_data['location'],
                "skills": job_data.get('skills', ''),
                "salary": job_data.get('salary', ''),
                "type": "job_posting"
            }
        )
        
        # Split into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = text_splitter.split_documents([doc])
        
        # Initialize embeddings with direct API key
        embeddings = AzureOpenAIEmbeddings(
            deployment=embeddings_deployment,
            api_key=api_key,
            azure_endpoint=api_base,
            openai_api_version=api_version,
            openai_api_type=api_type,
            chunk_size=500
        )
        
        # Create or update vector store
        if os.path.exists(os.path.join(DB_CHROMA_PATH, "chroma.sqlite3")):
            db = Chroma(
                persist_directory=DB_CHROMA_PATH, 
                embedding_function=embeddings
            )
            db.add_documents(docs)
            print(f" Added job '{job_data['title']}' to existing vector store")
        else:
            db = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=DB_CHROMA_PATH
            )
            print(f"Created new job vector store with job '{job_data['title']}'")
        
        db.persist()
        
        # Verify the data was added
        try:
            test_results = db.similarity_search(job_data['title'], k=1)
            print(f" Verification: Found {len(test_results)} documents matching '{job_data['title']}'")
        except:
            pass
        
        return db
        
    except Exception as e:
        print(f" Error adding job to vector store: {e}")
        traceback.print_exc()
        return None

# ================= ANALYSIS FUNCTIONS =================
def generate_applications_chart(recruiter_id):
    """Generate chart of applications over time"""
    db = get_db()
    
    # Get applications for last 30 days
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    apps = db.execute("""
    SELECT DATE(a.applied_at) as date, COUNT(*) as count
    FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ? AND a.applied_at >= ?
    GROUP BY DATE(a.applied_at)
    ORDER BY date
    """, (recruiter_id, thirty_days_ago)).fetchall()
    
    if not apps:
        return None
    
    dates = [app['date'] for app in apps]
    counts = [app['count'] for app in apps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dates, counts, marker='o', linestyle='-', color='#10b981')
    plt.title('Applications Over Time (Last 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Number of Applications')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def generate_status_pie_chart(recruiter_id):
    """Generate pie chart of application statuses"""
    db = get_db()
    
    stats = db.execute("""
    SELECT 
        COUNT(CASE WHEN a.status = 'pending' THEN 1 END) as pending,
        COUNT(CASE WHEN a.status = 'shortlisted' THEN 1 END) as shortlisted,
        COUNT(CASE WHEN a.status = 'rejected' THEN 1 END) as rejected
    FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ?
    """, (recruiter_id,)).fetchone()
    
    if not stats or (stats['pending'] == 0 and stats['shortlisted'] == 0 and stats['rejected'] == 0):
        return None
    
    labels = ['Pending', 'Shortlisted', 'Rejected']
    sizes = [stats['pending'], stats['shortlisted'], stats['rejected']]
    colors = ['#fbbf24', '#10b981', '#ef4444']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Application Status Distribution')
    plt.axis('equal')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def generate_skills_wordcloud(recruiter_id):
    """Generate word cloud of skills from candidates"""
    db = get_db()
    
    # Get all skills from candidates who applied
    skills_data = db.execute("""
    SELECT c.skills
    FROM applications a
    JOIN candidates c ON a.candidate_id = c.id
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ? AND c.skills IS NOT NULL
    """, (recruiter_id,)).fetchall()
    
    if not skills_data:
        return None
    
    # Combine all skills
    all_skills = []
    for row in skills_data:
        if row['skills']:
            skills = [s.strip() for s in row['skills'].split(',')]
            all_skills.extend(skills)
    
    if not all_skills:
        return None
    
    # Create word cloud
    text = ' '.join(all_skills)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Skills from Applicants')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def generate_job_performance_chart(recruiter_id):
    """Generate bar chart of job performance"""
    db = get_db()
    
    jobs = db.execute("""
    SELECT title, views, applications_count
    FROM jobs
    WHERE recruiter_id = ?
    ORDER BY applications_count DESC
    LIMIT 10
    """, (recruiter_id,)).fetchall()
    
    if not jobs:
        return None
    
    titles = [job['title'][:20] + '...' if len(job['title']) > 20 else job['title'] for job in jobs]
    views = [job['views'] for job in jobs]
    apps = [job['applications_count'] for job in jobs]
    
    x = range(len(titles))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], views, width, label='Views', color='#3b82f6')
    plt.bar([i + width/2 for i in x], apps, width, label='Applications', color='#10b981')
    
    plt.xlabel('Jobs')
    plt.ylabel('Count')
    plt.title('Job Performance: Views vs Applications')
    plt.xticks(x, titles, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def generate_score_distribution(recruiter_id):
    """Generate histogram of similarity scores"""
    db = get_db()
    
    scores = db.execute("""
    SELECT similarity_score
    FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ? AND similarity_score > 0
    """, (recruiter_id,)).fetchall()
    
    if not scores:
        return None
    
    score_values = [s['similarity_score'] * 100 for s in scores]
    
    plt.figure(figsize=(10, 6))
    plt.hist(score_values, bins=20, color='#8b5cf6', edgecolor='white', alpha=0.7)
    plt.title('Distribution of Candidate Match Scores')
    plt.xlabel('Match Score (%)')
    plt.ylabel('Number of Candidates')
    plt.grid(True, alpha=0.3)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def generate_analysis_report(recruiter_id):
    """Generate comprehensive analysis report"""
    db = get_db()
    
    # Get basic stats
    total_jobs = db.execute("SELECT COUNT(*) FROM jobs WHERE recruiter_id = ?", (recruiter_id,)).fetchone()[0]
    total_apps = db.execute("""
    SELECT COUNT(*) FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ?
    """, (recruiter_id,)).fetchone()[0]
    
    # Get average scores
    avg_scores = db.execute("""
    SELECT AVG(similarity_score) as avg_sim, AVG(resume_score) as avg_res
    FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ?
    """, (recruiter_id,)).fetchone()
    
    avg_sim = avg_scores['avg_sim'] * 100 if avg_scores['avg_sim'] else 0
    avg_res = avg_scores['avg_res'] * 100 if avg_scores['avg_res'] else 0
    
    # Get top skills
    skills_data = db.execute("""
    SELECT c.skills
    FROM applications a
    JOIN candidates c ON a.candidate_id = c.id
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ? AND c.skills IS NOT NULL
    """, (recruiter_id,)).fetchall()
    
    all_skills = []
    for row in skills_data:
        if row['skills']:
            skills = [s.strip() for s in row['skills'].split(',')]
            all_skills.extend(skills)
    
    top_skills = Counter(all_skills).most_common(10) if all_skills else []
    
    # Get application trends by day of week
    day_stats = db.execute("""
    SELECT strftime('%w', applied_at) as day, COUNT(*) as count
    FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ?
    GROUP BY day
    ORDER BY day
    """, (recruiter_id,)).fetchall()
    
    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    day_counts = [0] * 7
    for stat in day_stats:
        day_counts[int(stat['day'])] = stat['count']
    
    report = {
        'total_jobs': total_jobs,
        'total_applications': total_apps,
        'avg_similarity': round(avg_sim, 2),
        'avg_resume_score': round(avg_res, 2),
        'top_skills': top_skills,
        'applications_by_day': list(zip(days, day_counts))
    }
    
    return report

# ================= DEBUG ROUTES =================
@app.route('/debug/config')
def debug_config():
    """Debug endpoint to check configuration"""
    return f"""
    <h2>Configuration Check:</h2>
    <ul>
        <li>API_BASE: {' Set' if api_base else ' Missing'}</li>
        <li>API_KEY: {' Set' if api_key else ' Missing'}</li>
        <li>API_TYPE: {api_type}</li>
        <li>API_VERSION: {api_version}</li>
        <li>embeddings_deployment: {' Set' if embeddings_deployment else ' Missing'}</li>
        <li>chat_completion_deployment: {' Set' if chat_completion_deployment else ' Missing'}</li>
        <li>CHATBOT_ENABLED: {CHATBOT_ENABLED}</li>
        <li>LANGCHAIN_AVAILABLE: {LANGCHAIN_AVAILABLE}</li>
    </ul>
    <h2>Vector Stores:</h2>
    <ul>
        <li>Job vector store exists: {os.path.exists(DB_CHROMA_JOB_PATH)}</li>
        <li>Job chroma.sqlite3 exists: {os.path.exists(os.path.join(DB_CHROMA_JOB_PATH, "chroma.sqlite3"))}</li>
        <li>Resume vector store exists: {os.path.exists(DB_CHROMA_RESUME_PATH)}</li>
        <li>Resume chroma.sqlite3 exists: {os.path.exists(os.path.join(DB_CHROMA_RESUME_PATH, "chroma.sqlite3"))}</li>
    </ul>
    """

@app.route('/debug/jobs')
def debug_jobs():
    """Debug endpoint to check job vector store"""
    if not CHATBOT_ENABLED:
        return "Chatbot disabled"
    
    try:
        embeddings = AzureOpenAIEmbeddings(
            deployment=embeddings_deployment,
            api_key=api_key,
            azure_endpoint=api_base,
            openai_api_version=api_version,
            openai_api_type=api_type
        )
        
        db = Chroma(
            persist_directory=DB_CHROMA_JOB_PATH, 
            embedding_function=embeddings
        )
        
        # Get all documents
        all_docs = db.get()
        
        if not all_docs or not all_docs['documents']:
            return "No jobs found in vector store"
        
        html = "<h2>Jobs in Vector Store:</h2>"
        html += f"<p>Total documents: {len(all_docs['documents'])}</p>"
        html += "<ul>"
        
        for i, doc in enumerate(all_docs['documents']):
            metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
            title = metadata.get('title', 'Unknown')
            company = metadata.get('company', 'Unknown')
            location = metadata.get('location', 'Unknown')
            content_preview = doc[:100] + "..." if len(doc) > 100 else doc
            html += f"<li><b>{title}</b> at {company} ({location})<br><small>{content_preview}</small></li>"
        
        html += "</ul>"
        return html
        
    except Exception as e:
        return f"Error: {str(e)}"
    

@app.route('/debug/resumes-with-names')
@app.route('/debug/resumes-with-names')
def debug_resumes_with_names():
    """Enhanced debug endpoint to check resume metadata"""
    if not CHATBOT_ENABLED:
        return "Chatbot disabled"
    
    try:
        embeddings = AzureOpenAIEmbeddings(
            deployment=embeddings_deployment,
            api_key=api_key,
            azure_endpoint=api_base,
            openai_api_version=api_version,
            openai_api_type=api_type
        )
        
        db = Chroma(
            persist_directory=DB_CHROMA_RESUME_PATH, 
            embedding_function=embeddings
        )
        
        # Get all documents
        all_docs = db.get()
        
        if not all_docs or not all_docs['documents']:
            return "No resumes found in vector store"
        
        # Also get database candidates for comparison
        sql_db = get_db()
        db_candidates = sql_db.execute("SELECT id, name, resume_path FROM candidates WHERE resume_path IS NOT NULL").fetchall()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Resume Metadata Debug</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                th { background: #4CAF50; color: white; padding: 12px; text-align: left; }
                td { padding: 10px; border-bottom: 1px solid #ddd; }
                tr:hover { background: #f5f5f5; }
                .stats { background: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .warning { background: #ffeb3b; padding: 10px; border-radius: 5px; margin: 10px 0; }
                .success { background: #4CAF50; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
                .actions { margin: 20px 0; }
                .actions a { background: #008CBA; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-right: 10px; display: inline-block; }
            </style>
        </head>
        <body>
            <h1> Resume Metadata Debug</h1>
            
            <div class="stats">
                <h2> Statistics</h2>
        """
        
        # Database candidates
        html += f"<p><strong>Database Candidates with Resumes:</strong> {len(db_candidates)}</p>"
        html += "<ul>"
        for cand in db_candidates:
            resume_file = os.path.basename(cand['resume_path']) if cand['resume_path'] else 'No file'
            html += f"<li><strong>{cand['name']}</strong> (ID: {cand['id']}) - {resume_file}</li>"
        html += "</ul>"
        
        # Vector store stats
        html += f"<p><strong>Vector Store Total Chunks:</strong> {len(all_docs['documents'])}</p>"
        
        # Count unique candidates in vector store
        unique_names = set()
        name_counts = {}
        for metadata in all_docs['metadatas']:
            if metadata:
                name = metadata.get('candidate_name', 'Unknown')
                unique_names.add(name)
                name_counts[name] = name_counts.get(name, 0) + 1
        
        html += f"<p><strong>Unique Candidate Names in Vector Store:</strong> {len(unique_names)}</p>"
        
        # Show name distribution
        html += "<h3>Name Distribution:</h3><ul>"
        for name, count in sorted(name_counts.items()):
            color = "green" if name != "Unknown" else "red"
            html += f"<li style='color: {color};'><strong>{name}</strong>: {count} chunks</li>"
        html += "</ul>"
        
        # Actions
        html += """
            </div>
            
            <div class="actions">
                <a href="/admin/update-resume-names"> Update Resume Names from Database</a>
                <a href="/admin/deduplicate-candidates"> Deduplicate Candidate Chunks</a>
                <a href="/debug/resumes"> View Raw Resumes</a>
            </div>
        """
        
        # Warning if there are Unknown names
        if 'Unknown' in name_counts:
            html += """
            <div class="warning">
                There are chunks with 'Unknown' candidate names. 
                Click the "Update Resume Names" button above to fix this.
            </div>
            """
        
        # Table of all chunks
        html += """
            <h2> All Resume Chunks</h2>
            <table>
                <tr>
                    <th>Candidate Name</th>
                    <th>Source File</th>
                    <th>Candidate ID</th>
                    <th>Preview</th>
                </tr>
        """
        
        for i, doc in enumerate(all_docs['documents']):
            metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
            name = metadata.get('candidate_name', 'Unknown')
            source = metadata.get('source_document', 'Unknown')
            cand_id = metadata.get('candidate_id', 'Unknown')
            
            # Color code the rows
            row_color = ""
            if name == 'Unknown':
                row_color = " style='background-color: #ffeb3b;'"
            elif name != 'Unknown' and cand_id != 'Unknown':
                row_color = " style='background-color: #d4edda;'"
            
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            html += f"<tr{row_color}><td>{name}</td><td>{source}</td><td>{cand_id}</td><td>{preview}</td></tr>"
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"Error: {str(e)}"
@app.route('/admin/update-resume-names')
def update_resume_names():
    """Update old resume metadata with actual candidate names from database"""
    if not is_logged_in('recruiter'):
        return "Please login as recruiter first"
    
    try:
        # Get database connection
        db = get_db()
        
        # Get all candidates with resumes
        candidates = db.execute("""
        SELECT id, name, resume_path 
        FROM candidates 
        WHERE resume_path IS NOT NULL
        """).fetchall()
        
        if not candidates:
            return "No candidates with resumes found"
        
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            deployment=embeddings_deployment,
            api_key=api_key,
            azure_endpoint=api_base,
            openai_api_version=api_version,
            openai_api_type=api_type
        )
        
        # Load vector store
        chroma_db = Chroma(
            persist_directory=DB_CHROMA_RESUME_PATH, 
            embedding_function=embeddings
        )
        
        # Get all documents from vector store
        all_docs = chroma_db.get()
        
        if not all_docs or not all_docs['documents']:
            return "No documents in vector store"
        
        updated_count = 0
        candidate_map = {}
        
        # Create a mapping of filename patterns to candidate info
        for cand in candidates:
            if cand['resume_path']:
                filename = os.path.basename(cand['resume_path'])
                # Store both full filename and pattern (without extension)
                base_name = os.path.splitext(filename)[0]
                candidate_map[filename] = {
                    'name': cand['name'],
                    'id': cand['id']
                }
                candidate_map[base_name] = {
                    'name': cand['name'],
                    'id': cand['id']
                }
        
        print(f"Found {len(candidate_map)} filename patterns")
        
        # Update each document's metadata
        for i, metadata in enumerate(all_docs['metadatas']):
            if not metadata:
                continue
                
            source = metadata.get('source_document', '')
            if not source:
                continue
            
            # Check if this source matches any candidate
            matched = False
            for pattern, cand_info in candidate_map.items():
                if pattern in source or source in pattern:
                    # Update metadata
                    doc_id = all_docs['ids'][i]
                    new_metadata = dict(metadata)  # Copy existing metadata
                    new_metadata['candidate_name'] = cand_info['name']
                    new_metadata['candidate_id'] = str(cand_info['id'])
                    
                    # Update the document
                    chroma_db.update_document(
                        doc_id,
                        Document(
                            page_content=all_docs['documents'][i],
                            metadata=new_metadata
                        )
                    )
                    updated_count += 1
                    matched = True
                    print(f" Updated: {source} -> {cand_info['name']}")
                    break
            
            if not matched:
                print(f" No match for: {source}")
        
        chroma_db.persist()
        
        return f"""
        <h2>Resume Metadata Update Complete</h2>
        <p>Total candidates in database: {len(candidates)}</p>
        <p>Total chunks in vector store: {len(all_docs['documents'])}</p>
        <p>Updated chunks: {updated_count}</p>
        <p><a href="/debug/resumes-with-names">View updated resumes</a></p>
        """
        
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/debug/test-query')
def test_query():
    """Test endpoint to try a simple query"""
    if not CHATBOT_ENABLED:
        return "Chatbot disabled"
    
    try:
        query = request.args.get('q', 'jobs')
        response = chatbot(query, [], DB_CHROMA_JOB_PATH)
        return f"<h2>Query: {query}</h2><p>Response: {response}</p>"
    except Exception as e:
        return f"Error: {str(e)}"

# ================= HOME & AUTH =================
@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clarion - AI-Powered Job Portal</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }

            body {
                min-height: 100vh;
                color: white;
                background: linear-gradient(135deg, #0b1120 0%, #19223c 100%);
                position: relative;
                overflow-x: hidden;
            }

            .background-glow {
                position: fixed;
                top: -50%;
                right: -20%;
                width: 80%;
                height: 80%;
                background: radial-gradient(circle at center, rgba(0, 114, 255, 0.15) 0%, transparent 70%);
                border-radius: 50%;
                z-index: 0;
                pointer-events: none;
            }

            .navbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 22px 50px;
                position: relative;
                z-index: 10;
                backdrop-filter: blur(10px);
                background: rgba(11, 17, 32, 0.3);
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 28px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }

            .brand-logo {
                width: 44px;
                height: 44px;
                border-radius: 12px;
                background: linear-gradient(135deg, #3b82f6, #2563eb);
                display: flex;
                justify-content: center;
                align-items: center;
                box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
            }

            .brand-logo svg {
                width: 26px;
                height: 26px;
                fill: white;
            }

            .nav-links {
                display: flex;
                gap: 30px;
            }

            .nav-links a {
                color: #a0aec0;
                text-decoration: none;
                font-weight: 500;
                transition: color 0.2s;
                font-size: 16px;
            }

            .nav-links a:hover {
                color: white;
            }

            .chatbot-status {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
                margin-left: 10px;
            }
            
            .status-enabled {
                background: #10b981;
                color: white;
            }
            
            .status-disabled {
                background: #ef4444;
                color: white;
            }

            .hero {
                min-height: calc(100vh - 90px);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 30px;
                position: relative;
                z-index: 10;
            }

            .hero-content {
                max-width: 1200px;
                width: 100%;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 50px;
                align-items: center;
            }

            .hero-left h1 {
                font-size: 52px;
                font-weight: 800;
                line-height: 1.2;
                margin-bottom: 24px;
                letter-spacing: -1px;
            }

            .hero-left h1 span {
                background: linear-gradient(135deg, #60a5fa, #3b82f6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .hero-left p {
                font-size: 18px;
                color: #9ca3af;
                margin-bottom: 32px;
                line-height: 1.7;
                max-width: 500px;
            }

            .btn-group {
                display: flex;
                gap: 16px;
                margin-bottom: 40px;
            }

            .btn {
                padding: 14px 32px;
                border-radius: 50px;
                text-decoration: none;
                font-weight: 600;
                font-size: 16px;
                transition: all 0.3s;
                display: inline-block;
            }

            .btn-primary {
                background: #3b82f6;
                color: white;
                box-shadow: 0 10px 20px -8px rgba(59, 130, 246, 0.4);
            }

            .btn-primary:hover {
                background: #2563eb;
                transform: translateY(-2px);
                box-shadow: 0 15px 30px -8px rgba(37, 99, 235, 0.5);
            }

            .btn-secondary {
                background: rgba(255, 255, 255, 0.05);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }

            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.1);
                transform: translateY(-2px);
            }

            .stats {
                display: flex;
                gap: 30px;
            }

            .stat-item h3 {
                font-size: 28px;
                font-weight: 700;
                color: #3b82f6;
                margin-bottom: 4px;
            }

            .stat-item p {
                font-size: 14px;
                color: #6b7280;
                margin: 0;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .hero-right {
                background: rgba(255, 255, 255, 0.02);
                backdrop-filter: blur(20px);
                border-radius: 30px;
                padding: 40px;
                border: 1px solid rgba(255, 255, 255, 0.03);
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }

            .feature-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }

            .feature-card {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                padding: 24px;
                transition: all 0.3s;
            }

            .feature-card:hover {
                background: rgba(255, 255, 255, 0.05);
                border-color: rgba(59, 130, 246, 0.3);
                transform: translateY(-5px);
            }

            .feature-icon {
                width: 48px;
                height: 48px;
                background: rgba(59, 130, 246, 0.1);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 16px;
            }

            .feature-icon svg {
                width: 24px;
                height: 24px;
                fill: #3b82f6;
            }

            .feature-card h4 {
                color: white;
                font-size: 18px;
                margin-bottom: 8px;
            }

            .feature-card p {
                color: #9ca3af;
                font-size: 14px;
                line-height: 1.6;
                margin: 0;
            }

            .auth-links {
                margin-top: 30px;
                display: flex;
                gap: 20px;
                justify-content: center;
            }

            .auth-links a {
                color: #9ca3af;
                text-decoration: none;
                font-size: 15px;
                transition: color 0.2s;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .auth-links a:hover {
                color: #3b82f6;
            }

            .auth-links a svg {
                width: 18px;
                height: 18px;
                fill: currentColor;
            }
            
            .status-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                margin-top: 20px;
                background: rgba(255,255,255,0.1);
            }

            @media (max-width: 1024px) {
                .hero-content {
                    grid-template-columns: 1fr;
                    gap: 40px;
                }
                
                .hero-left {
                    text-align: center;
                }
                
                .hero-left p {
                    margin-left: auto;
                    margin-right: auto;
                }
                
                .btn-group {
                    justify-content: center;
                }
                
                .stats {
                    justify-content: center;
                }
            }

            @media (max-width: 768px) {
                .navbar {
                    padding: 20px;
                    flex-direction: column;
                    gap: 15px;
                }
                
                .hero-left h1 {
                    font-size: 36px;
                }
                
                .feature-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="background-glow"></div>
        
        <nav class="navbar">
            <div class="brand">
                <div class="brand-logo">
                    <svg viewBox="0 0 24 24">
                        <path d="M6 2h9l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm8 1.5V8h4.5L14 3.5zM8 11h8v1.8H8V11zm0 3.5h8v1.8H8v-1.8zm0-7h4v1.8H8V7.5z"/>
                    </svg>
                </div>
                <span>Clarion</span>
                <span class="chatbot-status {% if chatbot_enabled %}status-enabled{% else %}status-disabled{% endif %}">
                    {% if chatbot_enabled %} AI Chatbot Active{% else %} Chatbot Disabled{% endif %}
                </span>
            </div>
            <div class="nav-links">
                <a href="#features">Features</a>
                <a href="#about">About</a>
                <a href="/analysis/demo">Analytics Demo</a>
                {% if 'user' in session %}
                    {% if session['type'] == 'candidate' %}
                        <a href="/candidate/dashboard">Dashboard</a>
                    {% else %}
                        <a href="/recruiter/dashboard">Dashboard</a>
                    {% endif %}
                {% endif %}
            </div>
        </nav>

        <section class="hero">
            <div class="hero-content">
                <div class="hero-left">
                    <h1>Smart Resume Analysis for <span>Smarter Hiring</span></h1>
                    <p>Clarion uses AI to analyze resumes, match skills, and help recruiters find the perfect candidates faster than ever.</p>
                    
                    <div class="btn-group">
                        <a href="/candidate/register" class="btn btn-primary">Candidate Register</a>
                        <a href="/recruiter/register" class="btn btn-secondary">Recruiter Register</a>
                    </div>
                    
                    <div class="stats">
                        <div class="stat-item">
                            <h3 id="job-count">1000+</h3>
                            <p>Active Jobs</p>
                        </div>
                        <div class="stat-item">
                            <h3 id="candidate-count">5000+</h3>
                            <p>Candidates</p>
                        </div>
                        <div class="stat-item">
                            <h3 id="match-rate">95%</h3>
                            <p>Match Rate</p>
                        </div>
                    </div>
                    
                    <div class="status-badge">
                        {% if chatbot_enabled %}
                             Azure OpenAI Connected
                        {% else %}
                             Chatbot Disabled - Check Configuration
                        {% endif %}
                    </div>
                </div>
                
                <div class="hero-right">
                    <div class="feature-grid">
                        <div class="feature-card">
                            <div class="feature-icon">
                                <svg viewBox="0 0 24 24">
                                    <path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/>
                                </svg>
                            </div>
                            <h4>Resume Scoring</h4>
                            <p>AI-powered analysis of candidate resumes against job requirements</p>
                        </div>
                        
                        <div class="feature-card">
                            <div class="feature-icon">
                                <svg viewBox="0 0 24 24">
                                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                                </svg>
                            </div>
                            <h4>Skill Matching</h4>
                            <p>Intelligent matching of candidate skills with job requirements</p>
                        </div>
                        
                        <div class="feature-card">
                            <div class="feature-icon">
                                <svg viewBox="0 0 24 24">
                                    <path d="M21 6h-2v2h-2V6h-2V4h2V2h2v2h2v2zm-10 3c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm0 4c-2.33 0-7 1.17-7 3.5V17h14v-2.5c0-2.33-4.67-3.5-7-3.5z"/>
                                </svg>
                            </div>
                            <h4>AI Chatbot</h4>
                            <p>Get instant answers about jobs, skills, and career advice</p>
                        </div>
                        
                        <div class="feature-card">
                            <div class="feature-icon">
                                <svg viewBox="0 0 24 24">
                                    <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zM7 10h10v2H7zm0 4h6v2H7z"/>
                                </svg>
                            </div>
                            <h4>Analytics</h4>
                            <p>Detailed insights into your hiring process and candidate pool</p>
                        </div>
                    </div>
                    
                    <div class="auth-links">
                        <a href="/candidate/login">
                            <svg viewBox="0 0 24 24">
                                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
                            </svg>
                            Candidate Login
                        </a>
                        <a href="/recruiter/login">
                            <svg viewBox="0 0 24 24">
                                <path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/>
                            </svg>
                            Recruiter Login
                        </a>
                        {% if 'user' in session %}
                            <a href="/logout">
                                <svg viewBox="0 0 24 24">
                                    <path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.59L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z"/>
                                </svg>
                                Logout
                            </a>
                        {% endif %}
                    </div>
                </div>
            </div>
        </section>
        
        <script>
            // Fetch real stats from API
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('job-count').textContent = data.jobs + '+';
                    document.getElementById('candidate-count').textContent = data.candidates + '+';
                    document.getElementById('match-rate').textContent = data.match_rate + '%';
                });
        </script>
    </body>
    </html>
    """, chatbot_enabled=CHATBOT_ENABLED)

@app.route('/api/stats')
def api_stats():
    """API endpoint for stats"""
    db = get_db()
    jobs = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    candidates = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    
    # Calculate average match rate
    avg_match = db.execute("SELECT AVG(similarity_score) FROM applications").fetchone()[0]
    match_rate = int(avg_match * 100) if avg_match else 95
    
    return jsonify({
        'jobs': jobs,
        'candidates': candidates,
        'match_rate': match_rate
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ================= CANDIDATE ROUTES =================
@app.route('/candidate/register', methods=['GET', 'POST'])
def candidate_register():
    if request.method == 'POST':
        try:
            name = request.form['name'].strip()
            email = request.form['email'].strip().lower()
            password = request.form['password']
            
            if not all([name, email, password]):
                return "All fields are required"
            
            db = get_db()
            db.execute("""
            INSERT INTO candidates(name, email, password)
            VALUES(?, ?, ?)
            """, (name, email, generate_password_hash(password)))
            db.commit()
            
            return redirect('/candidate/login')
        except IntegrityError:
            return "Email already exists. Please use a different email."
        except Exception as e:
            return f"Registration failed: {str(e)}"
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Candidate Registration - Clarion</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                min-height: 100vh;
                background: linear-gradient(135deg, #0b1120, #19223c);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .register-container {
                background: rgba(255, 255, 255, 0.02);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.03);
                border-radius: 24px;
                padding: 40px;
                width: 100%;
                max-width: 450px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }
            
            h2 {
                color: white;
                font-size: 32px;
                margin-bottom: 8px;
                text-align: center;
            }
            
            .subtitle {
                color: #9ca3af;
                text-align: center;
                margin-bottom: 32px;
                font-size: 15px;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                color: #d1d5db;
                margin-bottom: 8px;
                font-size: 14px;
                font-weight: 500;
            }
            
            input {
                width: 100%;
                padding: 14px 16px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                color: white;
                font-size: 15px;
                transition: all 0.3s;
            }
            
            input:focus {
                outline: none;
                border-color: #3b82f6;
                background: rgba(59, 130, 246, 0.05);
            }
            
            input::placeholder {
                color: #4b5563;
            }
            
            button {
                width: 100%;
                padding: 14px;
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                margin-top: 10px;
            }
            
            button:hover {
                background: #2563eb;
                transform: translateY(-2px);
                box-shadow: 0 10px 20px -8px rgba(37, 99, 235, 0.4);
            }
            
            .links {
                margin-top: 24px;
                text-align: center;
            }
            
            .links a {
                color: #9ca3af;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
                display: inline-block;
                margin: 0 12px;
            }
            
            .links a:hover {
                color: #3b82f6;
            }
        </style>
    </head>
    <body>
        <div class="register-container">
            <h2>Create Account</h2>
            <div class="subtitle">Join Clarion as a candidate</div>
            
            <form method="post">
                <div class="form-group">
                    <label>Full Name</label>
                    <input type="text" name="name" placeholder="John Doe" required>
                </div>
                
                <div class="form-group">
                    <label>Email Address</label>
                    <input type="email" name="email" placeholder="john@example.com" required>
                </div>
                
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" name="password" placeholder="••••••••" minlength="6" required>
                </div>
                
                <button type="submit">Register</button>
            </form>
            
            <div class="links">
                <a href="/">← Back to Home</a>
                <a href="/candidate/login">Login →</a>
            </div>
        </div>
    </body>
    </html>
    """)

@app.route('/candidate/login', methods=['GET', 'POST'])
def candidate_login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        
        db = get_db()
        user = db.execute(
            "SELECT id, password FROM candidates WHERE email = ?",
            (email,)
        ).fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user'] = user['id']
            session['type'] = 'candidate'
            session['email'] = email
            session['chat_history'] = []
            return redirect('/candidate/dashboard')
        
        return "Invalid email or password"
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Candidate Login - Clarion</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                min-height: 100vh;
                background: linear-gradient(135deg, #0b1120, #19223c);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .login-container {
                background: rgba(255, 255, 255, 0.02);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.03);
                border-radius: 24px;
                padding: 40px;
                width: 100%;
                max-width: 450px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }
            
            h2 {
                color: white;
                font-size: 32px;
                margin-bottom: 8px;
                text-align: center;
            }
            
            .subtitle {
                color: #9ca3af;
                text-align: center;
                margin-bottom: 32px;
                font-size: 15px;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                color: #d1d5db;
                margin-bottom: 8px;
                font-size: 14px;
                font-weight: 500;
            }
            
            input {
                width: 100%;
                padding: 14px 16px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                color: white;
                font-size: 15px;
                transition: all 0.3s;
            }
            
            input:focus {
                outline: none;
                border-color: #3b82f6;
                background: rgba(59, 130, 246, 0.05);
            }
            
            input::placeholder {
                color: #4b5563;
            }
            
            button {
                width: 100%;
                padding: 14px;
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                margin-top: 10px;
            }
            
            button:hover {
                background: #2563eb;
                transform: translateY(-2px);
                box-shadow: 0 10px 20px -8px rgba(37, 99, 235, 0.4);
            }
            
            .links {
                margin-top: 24px;
                text-align: center;
            }
            
            .links a {
                color: #9ca3af;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
                display: inline-block;
                margin: 0 12px;
            }
            
            .links a:hover {
                color: #3b82f6;
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <h2>Welcome Back</h2>
            <div class="subtitle">Login to your candidate account</div>
            
            <form method="post">
                <div class="form-group">
                    <label>Email Address</label>
                    <input type="email" name="email" placeholder="john@example.com" required>
                </div>
                
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" name="password" placeholder="••••••••" required>
                </div>
                
                <button type="submit">Login</button>
            </form>
            
            <div class="links">
                <a href="/">← Back to Home</a>
                <a href="/candidate/register">Register →</a>
            </div>
        </div>
    </body>
    </html>
    """)

@app.route('/candidate/dashboard', methods=['GET', 'POST'])
def candidate_dashboard():
    if not is_logged_in('candidate'):
        return redirect('/candidate/login')
    
    db = get_db()
    
    # Get candidate info
    cand = db.execute("""
    SELECT name, qualification, skills, resume_path, notify_email 
    FROM candidates WHERE id = ?
    """, (session['user'],)).fetchone()
    
    # Get all jobs with view increment
    jobs = db.execute("""
    SELECT id, title, location, salary, company, skills, views 
    FROM jobs ORDER BY created_at DESC
    """).fetchall()
    
    # Get notifications
    notifs = db.execute("""
    SELECT message, timestamp FROM notifications 
    WHERE candidate_id = ? 
    ORDER BY timestamp DESC LIMIT 10
    """, (session['user'],)).fetchall()
    
    # Get applied jobs
    applied = db.execute("""
    SELECT job_id, status, rejection_reason FROM applications WHERE candidate_id = ?
    """, (session['user'],)).fetchall()
    applied_ids = {a['job_id'] for a in applied}
    
    # Get application status for applied jobs
    applied_with_details = db.execute("""
    SELECT a.job_id, a.status, a.rejection_reason, j.title, j.company
    FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE a.candidate_id = ?
    ORDER BY a.applied_at DESC
    """, (session['user'],)).fetchall()
    
    # Get recommended jobs based on skills
    recommended_jobs = []
    if cand and cand['skills']:
        candidate_skills = set([s.strip().lower() for s in cand['skills'].split(',')])
        for job in jobs:
            if job['id'] not in applied_ids and job['skills']:
                job_skills = set([s.strip().lower() for s in job['skills'].split(',')])
                if candidate_skills.intersection(job_skills):
                    match_count = len(candidate_skills.intersection(job_skills))
                    recommended_jobs.append({
                        'id': job['id'],
                        'title': job['title'],
                        'company': job['company'],
                        'location': job['location'],
                        'match_count': match_count
                    })
        recommended_jobs = sorted(recommended_jobs, key=lambda x: x['match_count'], reverse=True)[:5]
    
    # Chatbot logic
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST" and "query" in request.form:
        user_query = request.form.get("query")
        if user_query:
            history_for_llm = []
            for chat in session["chat_history"]:
                if 'question' in chat and 'response' in chat:
                    history_for_llm.append((chat["question"], chat["response"]))
            
            user_context = f"I am a candidate with skills: {cand['skills'] if cand and cand['skills'] else 'Not specified'}"
            response = chatbot(user_query, history_for_llm, DB_CHROMA_JOB_PATH, user_context)
            session["chat_history"].append({"question": user_query, "response": response})
            session.modified = True
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Candidate Dashboard - Clarion</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                background: #f3f4f6;
            }
            
            .header {
                background: linear-gradient(135deg, #0b1120, #19223c);
                color: white;
                padding: 20px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .header h1 {
                font-size: 24px;
                font-weight: 600;
            }
            
            .header a {
                color: white;
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
                transition: all 0.3s;
            }
            
            .header a:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            
            .container {
                max-width: 1400px;
                margin: 30px auto;
                padding: 0 30px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }
            
            .left-column, .right-column {
                display: flex;
                flex-direction: column;
                gap: 30px;
            }
            
            .section {
                background: white;
                border-radius: 16px;
                padding: 25px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            }
            
            .section h2 {
                font-size: 20px;
                margin-bottom: 20px;
                color: #111827;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .form-group {
                margin-bottom: 15px;
            }
            
            .form-group label {
                display: block;
                font-size: 14px;
                font-weight: 500;
                color: #374151;
                margin-bottom: 5px;
            }
            
            .form-group input {
                width: 100%;
                padding: 10px 12px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                font-size: 14px;
                transition: all 0.3s;
            }
            
            .form-group input:focus {
                outline: none;
                border-color: #3b82f6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            }
            
            button {
                background: #3b82f6;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            button:hover {
                background: #2563eb;
                transform: translateY(-1px);
            }
            
            .notification {
                background: #eff6ff;
                border-left: 4px solid #3b82f6;
                padding: 12px 15px;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            
            .notification strong {
                color: #1e40af;
                font-size: 12px;
            }
            
            .notification p {
                color: #1e293b;
                margin-top: 5px;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
            }
            
            th {
                text-align: left;
                padding: 12px;
                background: #f8fafc;
                color: #475569;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            td {
                padding: 15px 12px;
                border-bottom: 1px solid #e5e7eb;
                color: #1e293b;
                font-size: 14px;
            }
            
            .btn-success {
                background: #10b981;
                padding: 6px 12px;
                font-size: 12px;
            }
            
            .btn-success:hover {
                background: #059669;
            }
            
            .btn-secondary {
                background: #6b7280;
                padding: 6px 12px;
                font-size: 12px;
                cursor: default;
            }
            
            .btn-info {
                background: #8b5cf6;
                padding: 6px 12px;
                font-size: 12px;
                text-decoration: none;
                color: white;
                border-radius: 6px;
                display: inline-block;
            }
            
            .status-badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
            }
            
            .status-pending {
                background: #fef3c7;
                color: #92400e;
            }
            
            .status-shortlisted {
                background: #d1fae5;
                color: #065f46;
            }
            
            .status-rejected {
                background: #fee2e2;
                color: #991b1b;
            }
            
            .rejection-reason {
                background: #fef2f2;
                padding: 10px;
                border-radius: 6px;
                font-size: 13px;
                color: #991b1b;
                margin-top: 5px;
            }
            
            .chat-container {
                height: 300px;
                overflow-y: auto;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 20px;
                background: #f9fafb;
                margin-bottom: 15px;
            }
            
            .chat-message {
                margin-bottom: 15px;
            }
            
            .chat-question {
                text-align: right;
            }
            
            .chat-question div {
                display: inline-block;
                background: #3b82f6;
                color: white;
                padding: 10px 15px;
                border-radius: 15px 15px 0 15px;
                max-width: 70%;
            }
            
            .chat-response {
                text-align: left;
            }
            
            .chat-response div {
                display: inline-block;
                background: white;
                color: #1e293b;
                padding: 10px 15px;
                border-radius: 15px 15px 15px 0;
                max-width: 70%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            
            .chat-input {
                display: flex;
                gap: 10px;
            }
            
            .chat-input input {
                flex: 1;
                padding: 12px 15px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                font-size: 14px;
            }
            
            .chat-input input:focus {
                outline: none;
                border-color: #3b82f6;
            }
            
            .chat-input button {
                padding: 12px 24px;
                white-space: nowrap;
            }
            
            .chatbot-status {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                margin-left: 10px;
            }
            
            .status-enabled {
                background: #10b981;
                color: white;
            }
            
            .status-disabled {
                background: #ef4444;
                color: white;
            }
            
            .recommendation-card {
                background: #f0f9ff;
                border: 1px solid #bae6fd;
                border-radius: 12px;
                padding: 15px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .recommendation-info h4 {
                color: #0369a1;
                margin-bottom: 5px;
            }
            
            .recommendation-info p {
                color: #4b5563;
                font-size: 13px;
            }
            
            .match-badge {
                background: #10b981;
                color: white;
                padding: 4px 8px;
                border-radius: 20px;
                font-size: 12px;
            }
            
            @media (max-width: 1024px) {
                .container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Welcome, {{ cand['name'] }}
                <span class="chatbot-status {% if chatbot_enabled %}status-enabled{% else %}status-disabled{% endif %}">
                    {% if chatbot_enabled %} AI Assistant Online{% else %} AI Offline{% endif %}
                </span>
            </h1>
            <a href="/logout">Logout</a>
        </div>
        
        <div class="container">
            <!-- Left Column -->
            <div class="left-column">
                <!-- Profile Update -->
                <div class="section">
                    <h2> Update Profile</h2>
                    <form method="post" enctype="multipart/form-data" action="/candidate/update">
                        <div class="form-group">
                            <label>Qualification</label>
                            <input type="text" name="qualification" value="{{ cand['qualification'] or '' }}" placeholder="e.g., B.Tech Computer Science">
                        </div>
                        
                        <div class="form-group">
                            <label>Skills (comma separated)</label>
                            <input type="text" name="skills" value="{{ cand['skills'] or '' }}" placeholder="e.g., Python, SQL, React">
                        </div>
                        
                        <div class="form-group">
                            <label>Resume (PDF only)</label>
                            <input type="file" name="resume" accept=".pdf">
                            {% if cand['resume_path'] %}
                                <p style="margin-top: 8px; font-size: 13px;">
                                     <a href="/resume/{{ session['user'] }}" target="_blank" style="color: #3b82f6;">View Current Resume</a>
                                </p>
                            {% endif %}
                        </div>
                        
                        <div class="form-group">
                            <label>Notification Email</label>
                            <input type="email" name="notify_email" value="{{ cand['notify_email'] or '' }}" placeholder="Optional notification email">
                        </div>
                        
                        <button type="submit">Save Profile</button>
                    </form>
                </div>
                
                <!-- Notifications -->
                <div class="section">
                    <h2> Notifications</h2>
                    {% if notifs %}
                        {% for notif in notifs %}
                            <div class="notification">
                                <strong>{{ notif['timestamp'] }}</strong>
                                <p>{{ notif['message'] }}</p>
                            </div>
                        {% endfor %}
                    {% else %}
                        <p style="color: #6b7280; text-align: center;">No notifications yet</p>
                    {% endif %}
                </div>
            </div>
            
            <!-- Right Column -->
            <div class="right-column">
                <!-- My Applications -->
                <div class="section">
                    <h2> My Applications</h2>
                    {% if applied_with_details %}
                        <table>
                            <thead>
                                <tr>
                                    <th>Job</th>
                                    <th>Company</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for app in applied_with_details %}
                                    <tr>
                                        <td>{{ app['title'] }}</td>
                                        <td>{{ app['company'] }}</td>
                                        <td>
                                            <span class="status-badge status-{{ app['status'] }}">
                                                {{ app['status']|upper }}
                                            </span>
                                            {% if app['status'] == 'rejected' and app['rejection_reason'] %}
                                                <div class="rejection-reason">
                                                    {{ app['rejection_reason'] }}
                                                </div>
                                            {% endif %}
                                        </td>
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    {% else %}
                        <p style="color: #6b7280; text-align: center;">No applications yet</p>
                    {% endif %}
                </div>
                
                <!-- Recommended Jobs -->
                {% if recommended_jobs %}
                <div class="section">
                    <h2> Recommended for You</h2>
                    {% for job in recommended_jobs %}
                        <div class="recommendation-card">
                            <div class="recommendation-info">
                                <h4>{{ job['title'] }}</h4>
                                <p>{{ job['company'] }} • {{ job['location'] }}</p>
                            </div>
                            <div>
                                <span class="match-badge">{{ job['match_count'] }} skills match</span>
                                <form method="post" action="/apply/{{ job['id'] }}" style="display: inline; margin-left: 10px;">
                                    <button type="submit" class="btn-success" style="padding: 4px 12px;">Apply</button>
                                </form>
                            </div>
                        </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <!-- Available Jobs -->
                <div class="section">
                    <h2> Available Jobs</h2>
                    {% if jobs %}
                        <table>
                            <thead>
                                <tr>
                                    <th>Title</th>
                                    <th>Company</th>
                                    <th>Location</th>
                                    <th>Skills</th>
                                    <th>Views</th>
                                    <th>Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for job in jobs %}
                                    <tr>
                                        <td>{{ job['title'] }}</td>
                                        <td>{{ job['company'] }}</td>
                                        <td>{{ job['location'] }}</td>
                                        <td>{{ job['skills'] or 'Not specified' }}</td>
                                        <td>{{ job['views'] }}</td>
                                        <td>
                                            {% if job['id'] in applied_ids %}
                                                <button class="btn-secondary" disabled>Applied</button>
                                            {% else %}
                                                <form method="post" action="/apply/{{ job['id'] }}" style="display: inline;">
                                                    <button type="submit" class="btn-success">Apply</button>
                                                </form>
                                            {% endif %}
                                        </td>
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    {% else %}
                        <p style="color: #6b7280; text-align: center;">No jobs available</p>
                    {% endif %}
                </div>
                
                <!-- AI Job Assistant -->
                <div class="section">
                    <h2> AI Job Assistant</h2>
                    <p style="color: #6b7280; font-size: 13px; margin-bottom: 15px;">
                        Ask me about jobs, required skills, interview tips, or career advice!
                    </p>
                    
                    <div class="chat-container" id="chat-container">
                        {% for chat in chat_history %}
                            <div class="chat-message chat-question">
                                <div>{{ chat.question }}</div>
                            </div>
                            <div class="chat-message chat-response">
                                <div>{{ chat.response }}</div>
                            </div>
                        {% endfor %}
                        {% if not chat_history %}
                            <p style="color: #9ca3af; text-align: center; margin-top: 100px;">
                                Try asking:<br>
                                "What jobs match my skills?"<br>
                                "Tell me about Python developer roles"<br>
                                "What skills do I need for data science?"
                            </p>
                        {% endif %}
                    </div>
                    
                    <form method="POST" class="chat-input">
                        <input type="text" name="query" placeholder="Ask about jobs or skills..." required 
                               {% if not chatbot_enabled %}disabled{% endif %}>
                        <button type="submit" {% if not chatbot_enabled %}disabled{% endif %}>Send</button>
                    </form>
                    
                    {% if not chatbot_enabled %}
                        <p style="color: #ef4444; font-size: 12px; margin-top: 10px; text-align: center;">
                             AI Assistant is currently offline. Please check Azure OpenAI configuration.
                        </p>
                    {% endif %}
                    
                    <script>
                        var chatContainer = document.getElementById('chat-container');
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    </script>
                </div>
            </div>
        </div>
    </body>
    </html>
    """, cand=cand, jobs=jobs, notifs=notifs, applied_ids=applied_ids, 
          applied_with_details=applied_with_details, recommended_jobs=recommended_jobs,
          chat_history=session["chat_history"], chatbot_enabled=CHATBOT_ENABLED)


@app.route('/candidate/update', methods=['POST'])
def update_candidate():
    if not is_logged_in('candidate'):
        return redirect('/candidate/login')
    
    db = get_db()
    path = None
    
    # Get candidate name from database
    candidate = db.execute("SELECT name FROM candidates WHERE id = ?", (session['user'],)).fetchone()
    candidate_name = candidate['name'] if candidate else "Unknown"
    
    file = request.files.get('resume')
    if file and file.filename:
        filename = secure_filename(file.filename)
        if filename.lower().endswith('.pdf'):
            full_path = os.path.join(UPLOAD_FOLDER, f"candidate_{session['user']}.pdf")
            path = full_path
            file.save(full_path)
            print(f" Resume saved at: {full_path}")
            
            # Extract to vector store with candidate name
            if CHATBOT_ENABLED and LANGCHAIN_AVAILABLE:
                extractpdf(full_path, 1000, 10, candidate_name=candidate_name, candidate_id=session['user'])
        else:
            return "Only PDF files are allowed"
    
    try:
        if not path:
            existing = db.execute(
                "SELECT resume_path FROM candidates WHERE id = ?",
                (session['user'],)
            ).fetchone()
            path = existing['resume_path'] if existing else None
        
        db.execute("""
        UPDATE candidates 
        SET qualification = ?, skills = ?, resume_path = ?, notify_email = ?
        WHERE id = ?
        """, (
            request.form.get('qualification', '').strip(),
            request.form.get('skills', '').strip(),
            path,
            request.form.get('notify_email', '').strip().lower(),
            session['user']
        ))
        db.commit()
    except Exception as e:
        db.rollback()
        return f"Update failed: {str(e)}"
    
    return redirect('/candidate/dashboard')
@app.route('/apply/<int:job_id>', methods=['POST'])
def apply(job_id):
    if not is_logged_in('candidate'):
        return redirect('/candidate/login')
    
    db = get_db()
    try:
        # Increment job view
        increment_job_views(job_id)
        
        # Get candidate info
        cand = db.execute("""
        SELECT qualification, skills, resume_path FROM candidates WHERE id = ?
        """, (session['user'],)).fetchone()
        
        if not cand:
            return "Candidate not found"
        
        # Get job info
        job = db.execute("""
        SELECT title, skills, company FROM jobs WHERE id = ?
        """, (job_id,)).fetchone()
        
        if not job:
            return "Job not found"
        
        # Calculate similarities
        candidate_text = f"{cand['qualification'] or ''} {cand['skills'] or ''}".strip()
        job_skills = job['skills'] or ""
        
        sim = similarity(candidate_text, job_skills) if candidate_text and job_skills else 0.0
        res = resume_similarity(cand['resume_path'], job_skills)
        
        # Check if already applied
        existing = db.execute("""
        SELECT id FROM applications WHERE job_id = ? AND candidate_id = ?
        """, (job_id, session['user'])).fetchone()
        
        if existing:
            return redirect('/candidate/dashboard')
        
        # Insert application
        db.execute("""
        INSERT INTO applications(job_id, candidate_id, similarity_score, resume_score)
        VALUES(?, ?, ?, ?)
        """, (job_id, session['user'], sim, res))
        
        # Update job applications count
        db.execute("UPDATE jobs SET applications_count = applications_count + 1 WHERE id = ?", (job_id,))
        
        # Add notification
        message = f"You have applied for '{job['title']}' at {job['company']}."
        add_notification(session['user'], message)
        
        db.commit()
        return redirect('/candidate/dashboard')
    
    except IntegrityError:
        db.rollback()
        return "Already applied for this job"
    except Exception as e:
        db.rollback()
        return f"Application failed: {str(e)}"

# ================= RECRUITER ROUTES =================
@app.route('/recruiter/register', methods=['GET', 'POST'])
def recruiter_register():
    if request.method == 'POST':
        try:
            name = request.form['name'].strip()
            email = request.form['email'].strip().lower()
            password = request.form['password']
            company = request.form['company'].strip()
            
            if not all([name, email, password, company]):
                return "All fields are required"
            
            db = get_db()
            db.execute("""
            INSERT INTO recruiters(name, email, password, company)
            VALUES(?, ?, ?, ?)
            """, (name, email, generate_password_hash(password), company))
            db.commit()
            
            return redirect('/recruiter/login')
        except IntegrityError:
            return "Email already exists. Please use a different email."
        except Exception as e:
            return f"Registration failed: {str(e)}"
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recruiter Registration - Clarion</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                min-height: 100vh;
                background: linear-gradient(135deg, #0b1120, #19223c);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .register-container {
                background: rgba(255, 255, 255, 0.02);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.03);
                border-radius: 24px;
                padding: 40px;
                width: 100%;
                max-width: 450px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }
            
            h2 {
                color: white;
                font-size: 32px;
                margin-bottom: 8px;
                text-align: center;
            }
            
            .subtitle {
                color: #9ca3af;
                text-align: center;
                margin-bottom: 32px;
                font-size: 15px;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                color: #d1d5db;
                margin-bottom: 8px;
                font-size: 14px;
                font-weight: 500;
            }
            
            input {
                width: 100%;
                padding: 14px 16px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                color: white;
                font-size: 15px;
                transition: all 0.3s;
            }
            
            input:focus {
                outline: none;
                border-color: #10b981;
                background: rgba(16, 185, 129, 0.05);
            }
            
            input::placeholder {
                color: #4b5563;
            }
            
            button {
                width: 100%;
                padding: 14px;
                background: #10b981;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                margin-top: 10px;
            }
            
            button:hover {
                background: #059669;
                transform: translateY(-2px);
                box-shadow: 0 10px 20px -8px rgba(5, 150, 105, 0.4);
            }
            
            .links {
                margin-top: 24px;
                text-align: center;
            }
            
            .links a {
                color: #9ca3af;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
                display: inline-block;
                margin: 0 12px;
            }
            
            .links a:hover {
                color: #10b981;
            }
        </style>
    </head>
    <body>
        <div class="register-container">
            <h2>Create Account</h2>
            <div class="subtitle">Join Clarion as a recruiter</div>
            
            <form method="post">
                <div class="form-group">
                    <label>Full Name</label>
                    <input type="text" name="name" placeholder="John Doe" required>
                </div>
                
                <div class="form-group">
                    <label>Email Address</label>
                    <input type="email" name="email" placeholder="john@company.com" required>
                </div>
                
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" name="password" placeholder="••••••••" minlength="6" required>
                </div>
                
                <div class="form-group">
                    <label>Company Name</label>
                    <input type="text" name="company" placeholder="Tech Corp" required>
                </div>
                
                <button type="submit">Register</button>
            </form>
            
            <div class="links">
                <a href="/">← Back to Home</a>
                <a href="/recruiter/login">Login →</a>
            </div>
        </div>
    </body>
    </html>
    """)

@app.route('/recruiter/login', methods=['GET', 'POST'])
def recruiter_login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        
        db = get_db()
        rec = db.execute("""
        SELECT id, password FROM recruiters WHERE email = ?
        """, (email,)).fetchone()
        
        if rec and check_password_hash(rec['password'], password):
            session['user'] = rec['id']
            session['type'] = 'recruiter'
            session['email'] = email
            session['recruiter_chat_history'] = []
            return redirect('/recruiter/dashboard')
        
        return "Invalid email or password"
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recruiter Login - Clarion</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                min-height: 100vh;
                background: linear-gradient(135deg, #0b1120, #19223c);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .login-container {
                background: rgba(255, 255, 255, 0.02);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.03);
                border-radius: 24px;
                padding: 40px;
                width: 100%;
                max-width: 450px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }
            
            h2 {
                color: white;
                font-size: 32px;
                margin-bottom: 8px;
                text-align: center;
            }
            
            .subtitle {
                color: #9ca3af;
                text-align: center;
                margin-bottom: 32px;
                font-size: 15px;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                color: #d1d5db;
                margin-bottom: 8px;
                font-size: 14px;
                font-weight: 500;
            }
            
            input {
                width: 100%;
                padding: 14px 16px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                color: white;
                font-size: 15px;
                transition: all 0.3s;
            }
            
            input:focus {
                outline: none;
                border-color: #10b981;
                background: rgba(16, 185, 129, 0.05);
            }
            
            input::placeholder {
                color: #4b5563;
            }
            
            button {
                width: 100%;
                padding: 14px;
                background: #10b981;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                margin-top: 10px;
            }
            
            button:hover {
                background: #059669;
                transform: translateY(-2px);
                box-shadow: 0 10px 20px -8px rgba(5, 150, 105, 0.4);
            }
            
            .links {
                margin-top: 24px;
                text-align: center;
            }
            
            .links a {
                color: #9ca3af;
                text-decoration: none;
                font-size: 14px;
                transition: color 0.2s;
                display: inline-block;
                margin: 0 12px;
            }
            
            .links a:hover {
                color: #10b981;
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <h2>Welcome Back</h2>
            <div class="subtitle">Login to your recruiter account</div>
            
            <form method="post">
                <div class="form-group">
                    <label>Email Address</label>
                    <input type="email" name="email" placeholder="john@company.com" required>
                </div>
                
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" name="password" placeholder="••••••••" required>
                </div>
                
                <button type="submit">Login</button>
            </form>
            
            <div class="links">
                <a href="/">← Back to Home</a>
                <a href="/recruiter/register">Register →</a>
            </div>
        </div>
    </body>
    </html>
    """)

@app.route('/recruiter/dashboard', methods=['GET', 'POST'])
def recruiter_dashboard():
    if not is_logged_in('recruiter'):
        return redirect('/recruiter/login')
    
    db = get_db()
    
    # Get recruiter's jobs
    jobs = db.execute("""
    SELECT id, title, company, location, salary, views, applications_count, created_at 
    FROM jobs WHERE recruiter_id = ? 
    ORDER BY created_at DESC
    """, (session['user'],)).fetchall()
    
    # Get total applications across all jobs
    total_apps = db.execute("""
    SELECT COUNT(*) FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ?
    """, (session['user'],)).fetchone()[0]
    
    # Get statistics
    stats = db.execute("""
    SELECT 
        COUNT(CASE WHEN a.status = 'pending' THEN 1 END) as pending_count,
        COUNT(CASE WHEN a.status = 'shortlisted' THEN 1 END) as shortlisted_count,
        COUNT(CASE WHEN a.status = 'rejected' THEN 1 END) as rejected_count
    FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ?
    """, (session['user'],)).fetchone()
    
    # Get recent applications
    recent_apps = db.execute("""
    SELECT a.id, c.name, j.title, a.status, a.applied_at
    FROM applications a
    JOIN candidates c ON a.candidate_id = c.id
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ?
    ORDER BY a.applied_at DESC
    LIMIT 10
    """, (session['user'],)).fetchall()
    
    # Generate report data
    report = generate_analysis_report(session['user'])
    
    # Chatbot for recruiter
    if "recruiter_chat_history" not in session:
        session["recruiter_chat_history"] = []
    
    if request.method == "POST" and "query" in request.form:
        user_query = request.form.get("query")
        if user_query:
            history_for_llm = []
            for chat in session["recruiter_chat_history"]:
                if 'question' in chat and 'response' in chat:
                    history_for_llm.append((chat["question"], chat["response"]))
            
            user_context = f"I am a recruiter looking for candidates. I have posted {len(jobs)} jobs with {total_apps} total applications."
            response = chatbot(user_query, history_for_llm, DB_CHROMA_RESUME_PATH, user_context)
            session["recruiter_chat_history"].append({"question": user_query, "response": response})
            session.modified = True
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recruiter Dashboard - Clarion</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                background: #f3f4f6;
            }
            
            .header {
                background: linear-gradient(135deg, #0b1120, #19223c);
                color: white;
                padding: 20px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .header h1 {
                font-size: 24px;
                font-weight: 600;
            }
            
            .header a {
                color: white;
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
                transition: all 0.3s;
            }
            
            .header a:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            
            .container {
                max-width: 1400px;
                margin: 30px auto;
                padding: 0 30px;
            }
            
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .stat-card {
                background: white;
                padding: 25px;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            }
            
            .stat-card h3 {
                font-size: 32px;
                color: #10b981;
                margin-bottom: 5px;
            }
            
            .stat-card p {
                color: #6b7280;
                font-size: 14px;
                font-weight: 500;
            }
            
            .section {
                background: white;
                border-radius: 16px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            }
            
            .section h2 {
                font-size: 20px;
                margin-bottom: 20px;
                color: #111827;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .form-group {
                margin-bottom: 15px;
            }
            
            .form-group label {
                display: block;
                font-size: 14px;
                font-weight: 500;
                color: #374151;
                margin-bottom: 5px;
            }
            
            .form-group input, .form-group textarea {
                width: 100%;
                padding: 10px 12px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                font-size: 14px;
                transition: all 0.3s;
            }
            
            .form-group input:focus, .form-group textarea:focus {
                outline: none;
                border-color: #10b981;
                box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
            }
            
            .form-group textarea {
                min-height: 100px;
                resize: vertical;
            }
            
            button {
                background: #10b981;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            button:hover {
                background: #059669;
                transform: translateY(-1px);
            }
            
            .btn-analysis {
                background: #8b5cf6;
                padding: 8px 16px;
                text-decoration: none;
                color: white;
                border-radius: 8px;
                display: inline-block;
                margin-right: 10px;
            }
            
            .btn-analysis:hover {
                background: #7c3aed;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
            }
            
            th {
                text-align: left;
                padding: 12px;
                background: #f8fafc;
                color: #475569;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            td {
                padding: 15px 12px;
                border-bottom: 1px solid #e5e7eb;
                color: #1e293b;
                font-size: 14px;
            }
            
            .btn {
                background: #10b981;
                color: white;
                padding: 6px 12px;
                border-radius: 6px;
                text-decoration: none;
                font-size: 12px;
                display: inline-block;
            }
            
            .btn:hover {
                background: #059669;
            }
            
            .chat-container {
                height: 300px;
                overflow-y: auto;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 20px;
                background: #f9fafb;
                margin-bottom: 15px;
            }
            
            .chat-message {
                margin-bottom: 15px;
            }
            
            .chat-question {
                text-align: right;
            }
            
            .chat-question div {
                display: inline-block;
                background: #10b981;
                color: white;
                padding: 10px 15px;
                border-radius: 15px 15px 0 15px;
                max-width: 70%;
            }
            
            .chat-response {
                text-align: left;
            }
            
            .chat-response div {
                display: inline-block;
                background: white;
                color: #1e293b;
                padding: 10px 15px;
                border-radius: 15px 15px 15px 0;
                max-width: 70%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            
            .chat-input {
                display: flex;
                gap: 10px;
            }
            
            .chat-input input {
                flex: 1;
                padding: 12px 15px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                font-size: 14px;
            }
            
            .chat-input input:focus {
                outline: none;
                border-color: #10b981;
            }
            
            .chatbot-status {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                margin-left: 10px;
            }
            
            .status-enabled {
                background: #10b981;
                color: white;
            }
            
            .status-disabled {
                background: #ef4444;
                color: white;
            }
            
            .status-pending {
                color: #d97706;
            }
            
            .status-shortlisted {
                color: #059669;
            }
            
            .status-rejected {
                color: #dc2626;
            }
            
            .insight-card {
                background: #f0f9ff;
                border: 1px solid #bae6fd;
                border-radius: 12px;
                padding: 15px;
                margin-bottom: 10px;
            }
            
            .insight-title {
                font-weight: 600;
                color: #0369a1;
                margin-bottom: 5px;
            }
            
            .insight-value {
                font-size: 24px;
                color: #0c4a6e;
            }
            
            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Recruiter Dashboard
                <span class="chatbot-status {% if chatbot_enabled %}status-enabled{% else %}status-disabled{% endif %}">
                    {% if chatbot_enabled %} AI Hiring Assistant Online{% else %} AI Offline{% endif %}
                </span>
            </h1>
            <a href="/logout">Logout</a>
        </div>
        
        <div class="container">
            <!-- Stats -->
            <div class="stats">
                <div class="stat-card">
                    <h3>{{ jobs|length }}</h3>
                    <p>Jobs Posted</p>
                </div>
                <div class="stat-card">
                    <h3>{{ total_apps }}</h3>
                    <p>Total Applications</p>
                </div>
                <div class="stat-card">
                    <h3>{{ stats['pending_count'] or 0 }}</h3>
                    <p>Pending</p>
                </div>
                <div class="stat-card">
                    <h3>{{ stats['shortlisted_count'] or 0 }}</h3>
                    <p>Shortlisted</p>
                </div>
                <div class="stat-card">
                    <h3>{{ stats['rejected_count'] or 0 }}</h3>
                    <p>Rejected</p>
                </div>
            </div>
            
            <!-- Quick Actions -->
            <div class="section">
                <h2>⚡ Quick Actions</h2>
                <a href="/recruiter/analysis" class="btn-analysis"> View Full Analytics</a>
                <a href="#post-job" class="btn-analysis" style="background: #3b82f6;"> Post New Job</a>
            </div>
            
            <!-- Insights -->
            <div class="grid-2">
                <div class="section">
                    <h2> Hiring Insights</h2>
                    <div class="insight-card">
                        <div class="insight-title">Average Match Score</div>
                        <div class="insight-value">{{ report['avg_similarity'] }}%</div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-title">Average Resume Score</div>
                        <div class="insight-value">{{ report['avg_resume_score'] }}%</div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-title">Conversion Rate</div>
                        <div class="insight-value">
                            {% if total_apps > 0 %}
                                {{ "%.1f"|format((stats['shortlisted_count'] or 0) / total_apps * 100) }}%
                            {% else %}
                                0%
                            {% endif %}
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2> Top Skills from Applicants</h2>
                    {% if report['top_skills'] %}
                        {% for skill, count in report['top_skills'][:5] %}
                            <div style="margin-bottom: 10px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>{{ skill }}</span>
                                    <span>{{ count }} applicants</span>
                                </div>
                                <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                                    <div style="background: #10b981; width: {{ (count / report['top_skills'][0][1] * 100) if report['top_skills'] else 0 }}%; height: 8px; border-radius: 4px;"></div>
                                </div>
                            </div>
                        {% endfor %}
                    {% else %}
                        <p style="color: #6b7280;">No skills data available yet</p>
                    {% endif %}
                </div>
            </div>
            
            <!-- Post Job Section -->
            <div class="section" id="post-job">
                <h2> Post New Job</h2>
                <form method="post" action="/post_job">
                    <div class="form-group">
                        <label>Job Title</label>
                        <input type="text" name="title" required placeholder="e.g., Senior Python Developer">
                    </div>
                    
                    <div class="form-group">
                        <label>Job Description</label>
                        <textarea name="description" required placeholder="Detailed job description..."></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label>Location</label>
                        <input type="text" name="location" required placeholder="e.g., Remote, New York">
                    </div>
                    
                    <div class="form-group">
                        <label>Salary</label>
                        <input type="text" name="salary" placeholder="e.g., $100,000 - $130,000">
                    </div>
                    
                    <div class="form-group">
                        <label>Required Skills (comma separated)</label>
                        <input type="text" name="skills" placeholder="e.g., Python, Django, AWS, Docker">
                    </div>
                    
                    <div class="form-group">
                        <label>Company</label>
                        <input type="text" name="company" required placeholder="Company name">
                    </div>
                    
                    <button type="submit">Post Job</button>
                </form>
            </div>
            
            <!-- Your Jobs -->
            <div class="section">
                <h2> Your Posted Jobs</h2>
                {% if jobs %}
                    <table>
                        <thead>
                            <tr>
                                <th>Title</th>
                                <th>Company</th>
                                <th>Location</th>
                                <th>Views</th>
                                <th>Applications</th>
                                <th>Posted Date</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for job in jobs %}
                                <tr>
                                    <td>{{ job['title'] }}</td>
                                    <td>{{ job['company'] }}</td>
                                    <td>{{ job['location'] }}</td>
                                    <td>{{ job['views'] }}</td>
                                    <td>{{ job['applications_count'] }}</td>
                                    <td>{{ job['created_at'][:10] if job['created_at'] else 'N/A' }}</td>
                                    <td>
                                        <a href="/job/{{ job['id'] }}/applications" class="btn">View Applications</a>
                                    </td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <p style="color: #6b7280; text-align: center;">No jobs posted yet</p>
                {% endif %}
            </div>
            
            <!-- Recent Applications -->
            <div class="section">
                <h2> Recent Applications</h2>
                {% if recent_apps %}
                    <table>
                        <thead>
                            <tr>
                                <th>Candidate</th>
                                <th>Job</th>
                                <th>Status</th>
                                <th>Applied</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for app in recent_apps %}
                                <tr>
                                    <td>{{ app['name'] }}</td>
                                    <td>{{ app['title'] }}</td>
                                    <td class="status-{{ app['status'] }}">{{ app['status']|upper }}</td>
                                    <td>{{ app['applied_at'][:16] }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <p style="color: #6b7280; text-align: center;">No applications yet</p>
                {% endif %}
            </div>
            
            <!-- AI Hiring Assistant -->
            <div class="section">
                <h2> AI Hiring Assistant</h2>
                <p style="color: #6b7280; font-size: 13px; margin-bottom: 15px;">
                    Ask me about candidates, skill gaps, hiring trends, or get candidate recommendations!
                </p>
                
                <div class="chat-container" id="chat-container">
                    {% for chat in chat_history %}
                        <div class="chat-message chat-question">
                            <div>{{ chat.question }}</div>
                        </div>
                        <div class="chat-message chat-response">
                            <div>{{ chat.response }}</div>
                        </div>
                    {% endfor %}
                    {% if not chat_history %}
                        <p style="color: #9ca3af; text-align: center; margin-top: 100px;">
                            Try asking:<br>
                            "Show me top candidates for Python jobs"<br>
                            "What skills are most common among applicants?"<br>
                            "Which jobs have the most applications?"
                        </p>
                    {% endif %}
                </div>
                
                <form method="POST" class="chat-input">
                    <input type="text" name="query" placeholder="Ask about candidates..." required
                           {% if not chatbot_enabled %}disabled{% endif %}>
                    <button type="submit" {% if not chatbot_enabled %}disabled{% endif %}>Send</button>
                </form>
                
                {% if not chatbot_enabled %}
                    <p style="color: #ef4444; font-size: 12px; margin-top: 10px; text-align: center;">
                         AI Hiring Assistant is currently offline. Please check Azure OpenAI configuration.
                    </p>
                {% endif %}
                
                <script>
                    var chatContainer = document.getElementById('chat-container');
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                </script>
            </div>
        </div>
    </body>
    </html>
    """, jobs=jobs, total_apps=total_apps, stats=stats, recent_apps=recent_apps,
          report=report, chat_history=session["recruiter_chat_history"], chatbot_enabled=CHATBOT_ENABLED)

@app.route('/recruiter/analysis')
def recruiter_analysis():
    """Comprehensive analysis page for recruiters"""
    if not is_logged_in('recruiter'):
        return redirect('/recruiter/login')
    
    # Generate all charts
    apps_chart = generate_applications_chart(session['user'])
    status_chart = generate_status_pie_chart(session['user'])
    skills_wordcloud = generate_skills_wordcloud(session['user'])
    job_performance = generate_job_performance_chart(session['user'])
    score_dist = generate_score_distribution(session['user'])
    report = generate_analysis_report(session['user'])
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Dashboard - Clarion</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                background: #f3f4f6;
            }
            
            .header {
                background: linear-gradient(135deg, #0b1120, #19223c);
                color: white;
                padding: 20px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .header h1 {
                font-size: 24px;
                font-weight: 600;
            }
            
            .header a {
                color: white;
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
                transition: all 0.3s;
            }
            
            .header a:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            
            .container {
                max-width: 1400px;
                margin: 30px auto;
                padding: 0 30px;
            }
            
            .section {
                background: white;
                border-radius: 16px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            }
            
            .section h2 {
                font-size: 20px;
                margin-bottom: 20px;
                color: #111827;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .grid-2 {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }
            
            .chart-container {
                text-align: center;
            }
            
            .chart-container img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .stat-card {
                background: #f8fafc;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
            }
            
            .stat-card h3 {
                font-size: 28px;
                color: #10b981;
                margin-bottom: 5px;
            }
            
            .stat-card p {
                color: #6b7280;
                font-size: 14px;
            }
            
            .back-link {
                display: inline-block;
                margin-bottom: 20px;
                color: #10b981;
                text-decoration: none;
                font-weight: 500;
            }
            
            .back-link:hover {
                text-decoration: underline;
            }
            
            @media (max-width: 768px) {
                .grid-2 {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Analytics Dashboard</h1>
            <a href="/recruiter/dashboard">← Back to Dashboard</a>
        </div>
        
        <div class="container">
            <a href="/recruiter/dashboard" class="back-link">← Back to Dashboard</a>
            
            <!-- Key Metrics -->
            <div class="section">
                <h2> Key Metrics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>{{ report['total_jobs'] }}</h3>
                        <p>Total Jobs</p>
                    </div>
                    <div class="stat-card">
                        <h3>{{ report['total_applications'] }}</h3>
                        <p>Total Applications</p>
                    </div>
                    <div class="stat-card">
                        <h3>{{ report['avg_similarity'] }}%</h3>
                        <p>Avg. Match Score</p>
                    </div>
                    <div class="stat-card">
                        <h3>{{ report['avg_resume_score'] }}%</h3>
                        <p>Avg. Resume Score</p>
                    </div>
                </div>
            </div>
            
            <!-- Charts -->
            <div class="grid-2">
                {% if apps_chart %}
                <div class="section chart-container">
                    <h2> Applications Over Time</h2>
                    <img src="data:image/png;base64,{{ apps_chart }}" alt="Applications Chart">
                </div>
                {% endif %}
                
                {% if status_chart %}
                <div class="section chart-container">
                    <h2> Application Status</h2>
                    <img src="data:image/png;base64,{{ status_chart }}" alt="Status Chart">
                </div>
                {% endif %}
                
                {% if skills_wordcloud %}
                <div class="section chart-container">
                    <h2> Skills Word Cloud</h2>
                    <img src="data:image/png;base64,{{ skills_wordcloud }}" alt="Skills Word Cloud">
                </div>
                {% endif %}
                
                {% if job_performance %}
                <div class="section chart-container">
                    <h2>Job Performance</h2>
                    <img src="data:image/png;base64,{{ job_performance }}" alt="Job Performance">
                </div>
                {% endif %}
                
                {% if score_dist %}
                <div class="section chart-container">
                    <h2> Score Distribution</h2>
                    <img src="data:image/png;base64,{{ score_dist }}" alt="Score Distribution">
                </div>
                {% endif %}
            </div>
            
            <!-- Top Skills Table -->
            <div class="section">
                <h2> Top Skills from Applicants</h2>
                <table style="width: 100%;">
                    <thead>
                        <tr>
                            <th>Skill</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% if report['top_skills'] %}
                            {% for skill, count in report['top_skills'] %}
                            <tr>
                                <td>{{ skill }}</td>
                                <td>{{ count }}</td>
                                <td>{{ "%.1f"|format(count / report['total_applications'] * 100) if report['total_applications'] > 0 else 0 }}%</td>
                            </tr>
                            {% endfor %}
                        {% else %}
                            <tr>
                                <td colspan="3" style="text-align: center; color: #6b7280;">No skills data available</td>
                            </tr>
                        {% endif %}
                    </tbody>
                </table>
            </div>
            
            <!-- Application Trends -->
            <div class="section">
                <h2> Application Trends by Day</h2>
                <table style="width: 100%;">
                    <thead>
                        <tr>
                            <th>Day</th>
                            <th>Applications</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for day, count in report['applications_by_day'] %}
                        <tr>
                            <td>{{ day }}</td>
                            <td>{{ count }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """, apps_chart=apps_chart, status_chart=status_chart, 
          skills_wordcloud=skills_wordcloud, job_performance=job_performance,
          score_dist=score_dist, report=report)

@app.route('/analysis/demo')
def analysis_demo():
    """Demo analysis page with sample data"""
    # Generate sample charts for demo
    # This is just for demonstration when no data is available
    
    # Sample data
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
    counts = [np.random.randint(1, 10) for _ in range(30)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dates, counts, marker='o', linestyle='-', color='#10b981')
    plt.title('Sample Applications Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Applications')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    apps_chart = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Demo - Clarion</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                background: #f3f4f6;
            }
            
            .header {
                background: linear-gradient(135deg, #0b1120, #19223c);
                color: white;
                padding: 20px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .header h1 {
                font-size: 24px;
                font-weight: 600;
            }
            
            .header a {
                color: white;
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
            }
            
            .container {
                max-width: 1200px;
                margin: 30px auto;
                padding: 0 30px;
            }
            
            .section {
                background: white;
                border-radius: 16px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            }
            
            .section h2 {
                font-size: 20px;
                margin-bottom: 20px;
                color: #111827;
            }
            
            .chart-container {
                text-align: center;
            }
            
            .chart-container img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
            }
            
            .note {
                background: #fef3c7;
                border-left: 4px solid #d97706;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .back-link {
                display: inline-block;
                margin-bottom: 20px;
                color: #10b981;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Analytics Demo</h1>
            <a href="/">← Back to Home</a>
        </div>
        
        <div class="container">
            <div class="note">
                <strong> Demo Mode:</strong> This is a sample analytics view with generated data. 
                To see real analytics, post jobs and get applications from candidates.
            </div>
            
            <div class="section chart-container">
                <h2> Sample Applications Over Time</h2>
                <img src="data:image/png;base64,{{ apps_chart }}" alt="Sample Chart">
            </div>
            
            <div class="section">
                <h2>Sample Metrics</h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div style="text-align: center;">
                        <h3 style="font-size: 32px; color: #10b981;">12</h3>
                        <p>Jobs Posted</p>
                    </div>
                    <div style="text-align: center;">
                        <h3 style="font-size: 32px; color: #10b981;">156</h3>
                        <p>Applications</p>
                    </div>
                    <div style="text-align: center;">
                        <h3 style="font-size: 32px; color: #10b981;">78%</h3>
                        <p>Avg. Match Rate</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2> Get Started</h2>
                <p style="margin-bottom: 20px;">To see real analytics:</p>
                <ol style="margin-left: 20px;">
                    <li><a href="/recruiter/register">Register as a Recruiter</a></li>
                    <li>Post jobs on your dashboard</li>
                    <li>Wait for candidates to apply</li>
                    <li>View real analytics with your data!</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    """, apps_chart=apps_chart)

@app.route('/post_job', methods=['POST'])
def post_job():
    if not is_logged_in('recruiter'):
        return redirect('/recruiter/login')
    
    db = get_db()
    try:
        cursor = db.execute("""
        INSERT INTO jobs(recruiter_id, title, description, location, salary, skills, company)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """, (
            session['user'],
            request.form['title'].strip(),
            request.form['description'].strip(),
            request.form['location'].strip(),
            request.form.get('salary', '').strip(),
            request.form.get('skills', '').strip(),
            request.form['company'].strip()
        ))
        db.commit()
        
        # Get job ID
        job_id = cursor.lastrowid
        
        # Add to vector store if chatbot is enabled
        if CHATBOT_ENABLED and LANGCHAIN_AVAILABLE:
            job_data = {
                'job_id': job_id,
                'recruiter_id': session['user'],
                'title': request.form['title'].strip(),
                'description': request.form['description'].strip(),
                'location': request.form['location'].strip(),
                'salary': request.form.get('salary', '').strip(),
                'skills': request.form.get('skills', '').strip(),
                'company': request.form['company'].strip()
            }
            add_job_to_vectorstore(job_data)
            print(f" Added job '{job_data['title']}' to vector store")
        
        return redirect('/recruiter/dashboard')
    except Exception as e:
        db.rollback()
        print(f" Error posting job: {e}")
        return f"Failed to post job: {str(e)}"

@app.route('/job/<int:job_id>/applications')
def job_applications(job_id):
    if not is_logged_in('recruiter'):
        return redirect('/recruiter/login')
    
    db = get_db()
    
    # Verify job belongs to recruiter
    job_check = db.execute("""
    SELECT title, company FROM jobs WHERE id = ? AND recruiter_id = ?
    """, (job_id, session['user'])).fetchone()
    
    if not job_check:
        return "Job not found or access denied"
    
    # Get applications for this job
    apps = db.execute("""
    SELECT a.id, c.name, c.email, c.qualification, c.skills, 
           a.similarity_score, a.resume_score, a.status, a.applied_at, c.id as candidate_id
    FROM applications a
    JOIN candidates c ON a.candidate_id = c.id
    WHERE a.job_id = ?
    ORDER BY a.similarity_score DESC, a.resume_score DESC
    """, (job_id,)).fetchall()
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Applications - {{ job_check['title'] }}</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            
            body {
                background: #f3f4f6;
            }
            
            .header {
                background: linear-gradient(135deg, #0b1120, #19223c);
                color: white;
                padding: 20px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .header h1 {
                font-size: 24px;
                font-weight: 600;
            }
            
            .container {
                max-width: 1400px;
                margin: 30px auto;
                padding: 0 30px;
            }
            
            .back-link {
                display: inline-block;
                margin-bottom: 20px;
                color: #10b981;
                text-decoration: none;
                font-weight: 500;
            }
            
            .back-link:hover {
                text-decoration: underline;
            }
            
            .section {
                background: white;
                border-radius: 16px;
                padding: 25px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
            }
            
            th {
                text-align: left;
                padding: 12px;
                background: #f8fafc;
                color: #475569;
                font-size: 13px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            td {
                padding: 15px 12px;
                border-bottom: 1px solid #e5e7eb;
                color: #1e293b;
                font-size: 14px;
            }
            
            .btn {
                padding: 8px 14px;
                border-radius: 6px;
                text-decoration: none;
                font-size: 12px;
                font-weight: 500;
                display: inline-block;
                border: none;
                cursor: pointer;
                margin: 2px;
            }
            
            .btn-info {
                background: #3b82f6;
                color: white;
            }
            
            .btn-success {
                background: #10b981;
                color: white;
            }
            
            .btn-danger {
                background: #ef4444;
                color: white;
            }
            
            .btn:hover {
                opacity: 0.9;
                transform: translateY(-1px);
            }
            
            .status-pending {
                color: #d97706;
                font-weight: 500;
            }
            
            .status-shortlisted {
                color: #059669;
                font-weight: 500;
            }
            
            .status-rejected {
                color: #dc2626;
                font-weight: 500;
            }
            
            .score-high {
                color: #059669;
                font-weight: 600;
            }
            
            .score-medium {
                color: #d97706;
                font-weight: 600;
            }
            
            .score-low {
                color: #dc2626;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Applications for {{ job_check['title'] }} at {{ job_check['company'] }}</h1>
        </div>
        
        <div class="container">
            <a href="/recruiter/dashboard" class="back-link">← Back to Dashboard</a>
            
            <div class="section">
                {% if apps %}
                    <table>
                        <thead>
                            <tr>
                                <th>Candidate</th>
                                <th>Qualification</th>
                                <th>Skills</th>
                                <th>Similarity</th>
                                <th>Resume Score</th>
                                <th>Status</th>
                                <th>Applied Date</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for app in apps %}
                                <tr>
                                    <td>
                                        <strong>{{ app['name'] }}</strong><br>
                                        <small>{{ app['email'] }}</small>
                                    </td>
                                    <td>{{ app['qualification'] or 'Not specified' }}</td>
                                    <td>{{ app['skills'] or 'Not specified' }}</td>
                                    <td>
                                        {% set sim_score = app['similarity_score'] * 100 %}
                                        <span class="{% if sim_score >= 70 %}score-high{% elif sim_score >= 40 %}score-medium{% else %}score-low{% endif %}">
                                            {{ "%.1f"|format(sim_score) }}%
                                        </span>
                                    </td>
                                    <td>
                                        {% set res_score = app['resume_score'] * 100 %}
                                        <span class="{% if res_score >= 70 %}score-high{% elif res_score >= 40 %}score-medium{% else %}score-low{% endif %}">
                                            {{ "%.1f"|format(res_score) }}%
                                        </span>
                                    </td>
                                    <td>
                                        <span class="status-{{ app['status'] }}">{{ app['status']|upper }}</span>
                                    </td>
                                    <td>{{ app['applied_at'][:16] }}</td>
                                    <td>
                                        <a href="/resume/{{ app['candidate_id'] }}" class="btn btn-info" target="_blank">View Resume</a>
                                        <br><br>
                                        <form method="post" action="/update/{{ app['id'] }}/shortlisted" style="display: inline;">
                                            <button type="submit" class="btn btn-success">Shortlist</button>
                                        </form>
                                        <form method="post" action="/update/{{ app['id'] }}/rejected" style="display: inline;">
                                            <button type="submit" class="btn btn-danger">Reject</button>
                                        </form>
                                    </td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <p style="color: #6b7280; text-align: center;">No applications yet for this job</p>
                {% endif %}
            </div>
        </div>
    </body>
    </html>
    """, job_check=job_check, apps=apps)

@app.route('/update/<int:app_id>/<status>', methods=['POST'])
def update_status(app_id, status):
    if not is_logged_in('recruiter'):
        return redirect('/recruiter/login')
    
    db = get_db()
    try:
        # Get application details
        app_info = db.execute("""
        SELECT a.candidate_id, j.title, j.company 
        FROM applications a
        JOIN jobs j ON a.job_id = j.id
        WHERE a.id = ? AND j.recruiter_id = ?
        """, (app_id, session['user'])).fetchone()
        
        if not app_info:
            return "Application not found or access denied"
        
        # Update status
        db.execute("""
        UPDATE applications SET status = ? WHERE id = ?
        """, (status, app_id))
        
        # Add notification to candidate
        candidate_id = app_info['candidate_id']
        job_title = app_info['title']
        company = app_info['company']
        
        if status == 'shortlisted':
            message = f"Congratulations! You have been shortlisted for '{job_title}' at {company}."
        elif status == 'rejected':
            message = f"Your application for '{job_title}' at {company} has been reviewed but not selected at this time."
        else:
            message = f"Your application status for '{job_title}' at {company} has been updated to '{status}'."
        
        add_notification(candidate_id, message)
        
        db.commit()
        
        # Go back to the applications page
        referrer = request.headers.get('Referer', '/recruiter/dashboard')
        return redirect(referrer)
    
    except Exception as e:
        db.rollback()
        return f"Failed to update status: {str(e)}"

# ================= COMMON ROUTES =================
@app.route('/resume/<int:candidate_id>')
def resume(candidate_id):
    db = get_db()
    row = db.execute("""
    SELECT resume_path, name FROM candidates WHERE id = ?
    """, (candidate_id,)).fetchone()
    
    if not row or not row['resume_path'] or not os.path.exists(row['resume_path']):
        return "Resume not found", 404
    
    return send_file(row['resume_path'], 
                     download_name=f"{row['name']}_resume.pdf",
                     as_attachment=False,
                     mimetype='application/pdf')

@app.route('/api/job/<int:job_id>/view', methods=['POST'])
def track_job_view(job_id):
    """API endpoint to track job views"""
    increment_job_views(job_id)
    return jsonify({'success': True})

# ================= ERROR HANDLERS =================
@app.errorhandler(404)
def not_found(e):
    return "Page not found", 404

@app.errorhandler(500)
def server_error(e):
    return "Internal server error", 500

# ================= RUN APPLICATION =================
if __name__ == "__main__":
    print("=" * 70)
    print(" Starting Clarion - AI-Powered Resume Analyzer")
    print("=" * 70)
    print(f" Upload folder: {UPLOAD_FOLDER}")
    print(f"  Database: {DATABASE}")
    print(f" Email notifications: {' Enabled' if EMAIL_ENABLED else ' Disabled'}")
    print(f" Chatbot: {' Enabled' if CHATBOT_ENABLED else ' Disabled'}")
    if CHATBOT_ENABLED:
        print(f"    API Base: {api_base}")
        print(f"    Chat Deployment: {chat_completion_deployment}")
        print(f"    Embeddings Deployment: {embeddings_deployment}")
    else:
        print("        To enable chatbot:")
        print("      - Install LangChain: pip install langchain langchain-openai langchain-community")
        print("      - Set CHATBOT_ENABLED=True in .env file")
        print("      - Verify Azure OpenAI credentials in .env file")
    print("=" * 70)
    print(" Open http://localhost:8080")
    print(" Debug config: http://localhost:8080/debug/config")
    print(" Debug jobs: http://localhost:8080/debug/jobs")
    print(" Debug resumes: http://localhost:8080/debug/resumes")
    print("=" * 70)
    
    app.run(debug=True, port=8080, host='0.0.0.0', threaded=False)