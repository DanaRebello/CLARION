from flask import Flask, request, session, redirect, send_file, render_template_string, g
import sqlite3, os, smtplib, threading
from email.mime.text import MIMEText
from werkzeug.security import generate_password_hash, check_password_hash
from sqlite3 import IntegrityError
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import traceback

app = Flask(__name__)
app.secret_key = "your_secret_key_here_change_this_in_production"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= EMAIL CONFIG =================
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_ENABLED = False

def send_email(to_email, subject, body):
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
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# ================= DATABASE MANAGEMENT =================
def get_db():
    """Get database connection for current thread"""
    if not hasattr(g, 'database'):
        g.database = sqlite3.connect("users.db", check_same_thread=False)
        g.database.row_factory = sqlite3.Row
    return g.database

def init_db():
    """Initialize database with proper connection"""
    db = sqlite3.connect("users.db", check_same_thread=False)
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
    
    # Create indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_applications_job ON applications(job_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_applications_candidate ON applications(candidate_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_notifications_candidate ON notifications(candidate_id)")
    
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
def similarity(a, b):
    """Calculate text similarity between two strings"""
    if not a or not b:
        return 0.0
    try:
        tfidf = TfidfVectorizer()
        v = tfidf.fit_transform([a, b])
        return float(cosine_similarity(v[0], v[1])[0][0])
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
        
        if cand and cand['notify_email']:  # If notify_email is set
            send_email(cand['notify_email'], "Job Portal Notification", message)
        
        db.commit()
    except Exception as e:
        print(f"Error adding notification: {e}")
        db.rollback()

def is_logged_in(user_type=None):
    """Check if user is logged in, optionally check type"""
    if 'user' not in session or 'type' not in session:
        return False
    if user_type and session['type'] != user_type:
        return False
    return True

# ================= HOME & AUTH =================
@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Job Portal</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .nav { display: flex; justify-content: center; gap: 20px; margin: 30px 0; }
            .nav a { padding: 12px 24px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
            .nav a:hover { background: #0056b3; }
            .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Job Portal</h1>
            <div class="section">
                <h2>For Candidates</h2>
                <div class="nav">
                    <a href="/candidate/register">Register as Candidate</a>
                    <a href="/candidate/login">Candidate Login</a>
                </div>
            </div>
            <div class="section">
                <h2>For Recruiters</h2>
                <div class="nav">
                    <a href="/recruiter/register">Register as Recruiter</a>
                    <a href="/recruiter/login">Recruiter Login</a>
                </div>
            </div>
            {% if 'user' in session %}
                <p style="text-align: center; color: green;">
                    Logged in as {{ session['type'] }} | 
                    <a href="/logout">Logout</a>
                </p>
            {% endif %}
        </div>
    </body>
    </html>
    """)

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
        <title>Candidate Registration</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 500px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { color: #333; text-align: center; }
            form { display: flex; flex-direction: column; gap: 15px; }
            input[type="text"], input[type="email"], input[type="password"] {
                padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px;
            }
            button { padding: 12px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #218838; }
            .back { display: block; text-align: center; margin-top: 20px; color: #007bff; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Candidate Registration</h2>
            <form method="post">
                <input type="text" name="name" placeholder="Full Name" required>
                <input type="email" name="email" placeholder="Email Address" required>
                <input type="password" name="password" placeholder="Password" required minlength="6">
                <button type="submit">Register</button>
            </form>
            <a href="/" class="back">Back to Home</a>
            <a href="/candidate/login" class="back">Already have an account? Login</a>
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
            return redirect('/candidate/dashboard')
        
        return "Invalid email or password"
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Candidate Login</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 500px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { color: #333; text-align: center; }
            form { display: flex; flex-direction: column; gap: 15px; }
            input[type="email"], input[type="password"] {
                padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px;
            }
            button { padding: 12px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .back { display: block; text-align: center; margin-top: 20px; color: #007bff; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Candidate Login</h2>
            <form method="post">
                <input type="email" name="email" placeholder="Email Address" required>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
            <a href="/" class="back">Back to Home</a>
            <a href="/candidate/register" class="back">Don't have an account? Register</a>
        </div>
    </body>
    </html>
    """)

@app.route('/candidate/dashboard')
def candidate_dashboard():
    if not is_logged_in('candidate'):
        return redirect('/candidate/login')
    
    db = get_db()
    
    # Get candidate info
    cand = db.execute("""
    SELECT name, qualification, skills, resume_path, notify_email 
    FROM candidates WHERE id = ?
    """, (session['user'],)).fetchone()
    
    # Get all jobs
    jobs = db.execute("""
    SELECT id, title, location, salary, company, skills 
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
    SELECT job_id FROM applications WHERE candidate_id = ?
    """, (session['user'],)).fetchall()
    applied_ids = {a['job_id'] for a in applied}
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Candidate Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }
            .header { background: #007bff; color: white; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            h2 { color: #333; margin-top: 0; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #f8f9fa; }
            .btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
            .btn-primary { background: #007bff; color: white; }
            .btn-success { background: #28a745; color: white; }
            .btn-secondary { background: #6c757d; color: white; }
            .btn-danger { background: #dc3545; color: white; }
            .form-group { margin: 10px 0; }
            input, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            .notification { background: #e7f3ff; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }
            .logout { float: right; color: white; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Candidate Dashboard</h1>
            <a href="/logout" class="logout">Logout</a>
        </div>
        
        <div class="container">
            <!-- Profile Update Section -->
            <div class="section">
                <h2>Update Profile</h2>
                <form method="post" enctype="multipart/form-data" action="/candidate/update">
                    <div class="form-group">
                        <label>Qualification:</label>
                        <input type="text" name="qualification" value="{{ cand['qualification'] or '' }}" placeholder="e.g., B.Tech Computer Science">
                    </div>
                    <div class="form-group">
                        <label>Skills (comma separated):</label>
                        <input type="text" name="skills" value="{{ cand['skills'] or '' }}" placeholder="e.g., Python, Flask, SQL, Machine Learning">
                    </div>
                    <div class="form-group">
                        <label>Resume (PDF only):</label>
                        <input type="file" name="resume" accept=".pdf">
                        {% if cand['resume_path'] %}
                            <p>Current resume: <a href="/resume/{{ session['user'] }}">Download</a></p>
                        {% endif %}
                    </div>
                    <div class="form-group">
                        <label>Notification Email:</label>
                        <input type="email" name="notify_email" value="{{ cand['notify_email'] or '' }}" placeholder="Optional: Different from login email">
                    </div>
                    <button type="submit" class="btn btn-primary">Save Profile</button>
                </form>
            </div>
            
            <!-- Notifications Section -->
            <div class="section">
                <h2>Notifications</h2>
                {% if notifs %}
                    {% for notif in notifs %}
                        <div class="notification">
                            <strong>{{ notif['timestamp'] }}</strong><br>{{ notif['message'] }}
                        </div>
                    {% endfor %}
                {% else %}
                    <p>No notifications yet.</p>
                {% endif %}
            </div>
            
            <!-- Available Jobs Section -->
            <div class="section">
                <h2>Available Jobs</h2>
                {% if jobs %}
                    <table>
                        <thead>
                            <tr>
                                <th>Title</th>
                                <th>Company</th>
                                <th>Location</th>
                                <th>Salary</th>
                                <th>Skills Required</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for job in jobs %}
                                <tr>
                                    <td>{{ job['title'] }}</td>
                                    <td>{{ job['company'] }}</td>
                                    <td>{{ job['location'] }}</td>
                                    <td>{{ job['salary'] or 'Not specified' }}</td>
                                    <td>{{ job['skills'] or 'Not specified' }}</td>
                                    <td>
                                        {% if job['id'] in applied_ids %}
                                            <span class="btn btn-secondary">Applied</span>
                                        {% else %}
                                            <form method="post" action="/apply/{{ job['id'] }}" style="display: inline;">
                                                <button type="submit" class="btn btn-success">Apply</button>
                                            </form>
                                        {% endif %}
                                    </td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <p>No jobs available at the moment.</p>
                {% endif %}
            </div>
        </div>
    </body>
    </html>
    """, cand=cand, jobs=jobs, notifs=notifs, applied_ids=applied_ids)

@app.route('/candidate/update', methods=['POST'])
def update_candidate():
    if not is_logged_in('candidate'):
        return redirect('/candidate/login')
    
    path = None
    file = request.files.get('resume')
    if file and file.filename:
        if file.filename.endswith('.pdf'):
            path = os.path.join(UPLOAD_FOLDER, f"candidate_{session['user']}.pdf")
            file.save(path)
        else:
            return "Only PDF files are allowed for resumes"
    
    db = get_db()
    try:
        # Keep existing resume if new one not uploaded
        if not path:
            existing = db.execute("SELECT resume_path FROM candidates WHERE id = ?", 
                                   (session['user'],)).fetchone()
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
        
        # Add notification
        message = f"You have applied for '{job['title']}' at {job['company']}.%"
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
        <title>Recruiter Registration</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 500px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { color: #333; text-align: center; }
            form { display: flex; flex-direction: column; gap: 15px; }
            input[type="text"], input[type="email"], input[type="password"] {
                padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px;
            }
            button { padding: 12px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #218838; }
            .back { display: block; text-align: center; margin-top: 20px; color: #007bff; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Recruiter Registration</h2>
            <form method="post">
                <input type="text" name="name" placeholder="Full Name" required>
                <input type="email" name="email" placeholder="Email Address" required>
                <input type="password" name="password" placeholder="Password" required minlength="6">
                <input type="text" name="company" placeholder="Company Name" required>
                <button type="submit">Register</button>
            </form>
            <a href="/" class="back">Back to Home</a>
            <a href="/recruiter/login" class="back">Already have an account? Login</a>
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
            return redirect('/recruiter/dashboard')
        
        return "Invalid email or password"
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recruiter Login</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 500px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { color: #333; text-align: center; }
            form { display: flex; flex-direction: column; gap: 15px; }
            input[type="email"], input[type="password"] {
                padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px;
            }
            button { padding: 12px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .back { display: block; text-align: center; margin-top: 20px; color: #007bff; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Recruiter Login</h2>
            <form method="post">
                <input type="email" name="email" placeholder="Email Address" required>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
            <a href="/" class="back">Back to Home</a>
            <a href="/recruiter/register" class="back">Don't have an account? Register</a>
        </div>
    </body>
    </html>
    """)

@app.route('/recruiter/dashboard')
def recruiter_dashboard():
    if not is_logged_in('recruiter'):
        return redirect('/recruiter/login')
    
    db = get_db()
    
    # Get recruiter's jobs
    jobs = db.execute("""
    SELECT id, title, company, location, salary, created_at 
    FROM jobs WHERE recruiter_id = ? 
    ORDER BY created_at DESC
    """, (session['user'],)).fetchall()
    
    # Get total applications across all jobs
    total_apps = db.execute("""
    SELECT COUNT(*) FROM applications a
    JOIN jobs j ON a.job_id = j.id
    WHERE j.recruiter_id = ?
    """, (session['user'],)).fetchone()[0]
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recruiter Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }
            .header { background: #dc3545; color: white; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            h2 { color: #333; margin-top: 0; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #f8f9fa; }
            .btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
            .btn-primary { background: #007bff; color: white; }
            .btn-info { background: #17a2b8; color: white; }
            .form-group { margin: 10px 0; }
            input, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            .stats { display: flex; gap: 20px; margin: 20px 0; }
            .stat-card { flex: 1; background: white; padding: 20px; border-radius: 5px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .stat-card h3 { margin: 0; color: #007bff; }
            .logout { float: right; color: white; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Recruiter Dashboard</h1>
            <a href="/logout" class="logout">Logout</a>
        </div>
        
        <div class="container">
            <!-- Stats Section -->
            <div class="stats">
                <div class="stat-card">
                    <h3>{{ jobs|length }}</h3>
                    <p>Jobs Posted</p>
                </div>
                <div class="stat-card">
                    <h3>{{ total_apps }}</h3>
                    <p>Total Applications</p>
                </div>
            </div>
            
            <!-- Post Job Section -->
            <div class="section">
                <h2>Post New Job</h2>
                <form method="post" action="/post_job">
                    <div class="form-group">
                        <label>Job Title:</label>
                        <input type="text" name="title" required placeholder="e.g., Python Developer">
                    </div>
                    <div class="form-group">
                        <label>Job Description:</label>
                        <textarea name="description" rows="3" required placeholder="Detailed job description..."></textarea>
                    </div>
                    <div class="form-group">
                        <label>Location:</label>
                        <input type="text" name="location" required placeholder="e.g., Remote, New York, etc.">
                    </div>
                    <div class="form-group">
                        <label>Salary:</label>
                        <input type="text" name="salary" placeholder="e.g., $80,000 - $100,000">
                    </div>
                    <div class="form-group">
                        <label>Required Skills (comma separated):</label>
                        <input type="text" name="skills" placeholder="e.g., Python, Flask, AWS, Docker">
                    </div>
                    <div class="form-group">
                        <label>Company:</label>
                        <input type="text" name="company" required placeholder="Company name">
                    </div>
                    <button type="submit" class="btn btn-primary">Post Job</button>
                </form>
            </div>
            
            <!-- Posted Jobs Section -->
            <div class="section">
                <h2>Your Posted Jobs</h2>
                {% if jobs %}
                    <table>
                        <thead>
                            <tr>
                                <th>Title</th>
                                <th>Company</th>
                                <th>Location</th>
                                <th>Salary</th>
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
                                    <td>{{ job['salary'] or 'Not specified' }}</td>
                                    <td>{{ job['created_at'] or 'Not available' }}</td>
                                    <td>
                                        <a href="/job/{{ job['id'] }}/applications" class="btn btn-info">
                                            View Applications
                                        </a>
                                    </td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <p>You haven't posted any jobs yet.</p>
                {% endif %}
            </div>
        </div>
    </body>
    </html>
    """, jobs=jobs, total_apps=total_apps)

@app.route('/post_job', methods=['POST'])
def post_job():
    if not is_logged_in('recruiter'):
        return redirect('/recruiter/login')
    
    db = get_db()
    try:
        db.execute("""
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
        return redirect('/recruiter/dashboard')
    except Exception as e:
        db.rollback()
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
            body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }
            .header { background: #17a2b8; color: white; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            h2 { color: #333; margin-top: 0; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #f8f9fa; }
            .btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
            .btn-success { background: #28a745; color: white; }
            .btn-danger { background: #dc3545; color: white; }
            .btn-info { background: #17a2b8; color: white; }
            .btn-secondary { background: #6c757d; color: white; }
            .status-pending { color: #ffc107; font-weight: bold; }
            .status-shortlisted { color: #28a745; font-weight: bold; }
            .status-rejected { color: #dc3545; font-weight: bold; }
            .back { display: inline-block; margin: 20px 0; color: #007bff; text-decoration: none; }
            .score { font-weight: bold; }
            .score-high { color: #28a745; }
            .score-medium { color: #ffc107; }
            .score-low { color: #dc3545; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Applications for {{ job_check['title'] }} at {{ job_check['company'] }}</h1>
        </div>
        
        <div class="container">
            <a href="/recruiter/dashboard" class="back">← Back to Dashboard</a>
            
            <div class="section">
                {% if apps %}
                    <table>
                        <thead>
                            <tr>
                                <th>Candidate</th>
                                <th>Qualification</th>
                                <th>Skills</th>
                                <th>Similarity Score</th>
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
                                        <span class="score {% if sim_score >= 70 %}score-high{% elif sim_score >= 40 %}score-medium{% else %}score-low{% endif %}">
                                            {{ "%.1f"|format(sim_score) }}%
                                        </span>
                                    </td>
                                    <td>
                                        {% set res_score = app['resume_score'] * 100 %}
                                        <span class="score {% if res_score >= 70 %}score-high{% elif res_score >= 40 %}score-medium{% else %}score-low{% endif %}">
                                            {{ "%.1f"|format(res_score) }}%
                                        </span>
                                    </td>
                                    <td>
                                        {% if app['status'] == 'pending' %}
                                            <span class="status-pending">Pending</span>
                                        {% elif app['status'] == 'shortlisted' %}
                                            <span class="status-shortlisted">Shortlisted</span>
                                        {% elif app['status'] == 'rejected' %}
                                            <span class="status-rejected">Rejected</span>
                                        {% endif %}
                                    </td>
                                    <td>{{ app['applied_at'] or 'Not available' }}</td>
                                    <td>
                                        <a href="/resume/{{ app['candidate_id'] }}" class="btn btn-info" target="_blank">
                                            View Resume
                                        </a><br><br>
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
                    <p>No applications yet for this job.</p>
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

# ================= ERROR HANDLERS =================
@app.errorhandler(404)
def not_found(e):
    return "Page not found", 404

@app.errorhandler(500)
def server_error(e):
    return "Internal server error", 500

# ================= RUN APPLICATION =================
if __name__ == "__main__":
    print("Starting Job Portal Application...")
    print("Access the application at: http://localhost:8080")
    print("Note: Email notifications are currently disabled.")
    print("To enable email, set EMAIL_ENABLED = True and configure EMAIL_SENDER/EMAIL_PASSWORD")
    
    # Configure Flask to use single thread mode for SQLite
    app.run(debug=True, port=8080, host='0.0.0.0', threaded=False)
