from flask import Flask, request, session, redirect, send_file
import sqlite3, os
from werkzeug.security import generate_password_hash, check_password_hash
from sqlite3 import IntegrityError
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ----------------
#def get_db():
#   return sqlite3.connect("users.db", check_same_thread=False)
def get_db():
    print("FLASK DB PATH:", os.path.abspath("users.db"))
    return sqlite3.connect("users.db", check_same_thread=False)


def create_tables():
    with get_db() as conn:
        c = conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS candidates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            qualification TEXT,
            skills TEXT,
            resume_path TEXT
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS recruiters(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            company TEXT
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS jobs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recruiter_id INTEGER,
            title TEXT,
            description TEXT,
            location TEXT,
            salary TEXT,
            skills TEXT,
            company TEXT
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS applications(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        candidate_id INTEGER,
        similarity_score REAL,
        resume_score REAL,
        status TEXT DEFAULT 'pending',
        UNIQUE(job_id, candidate_id)
        )
        """)


        c.execute("""
        CREATE TABLE IF NOT EXISTS shortlisted(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER UNIQUE,
            recruiter_id INTEGER,
            candidate_id INTEGER,
            job_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS notifications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

create_tables()


# ---------------- SIMILARITY ----------------
def similarity(a, b):
    if not a or not b:
        return 0.0
    tfidf = TfidfVectorizer()
    v = tfidf.fit_transform([a, b])
    return float(cosine_similarity(v[0], v[1])[0][0])
#Resume similarity 
def resume_similarity(resume_path, job_skills):
    if not resume_path or not os.path.exists(resume_path):
        return None

    try:
        reader = PdfReader(resume_path)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return similarity(text, job_skills)
    except:
        return None

# ---------------- HOME ----------------
@app.route('/')
def home():
    return """
    <h2>Job Portal</h2>
    <a href="/candidate/register">Candidate Register</a> |
    <a href="/candidate/login">Candidate Login</a><br><br>
    <a href="/recruiter/register">Recruiter Register</a> |
    <a href="/recruiter/login">Recruiter Login</a>
    """

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ================= CANDIDATE =================
@app.route('/candidate/register', methods=['GET','POST'])
def candidate_register():
    if request.method == 'POST':
        try:
            with get_db() as conn:
                conn.execute("""
                INSERT INTO candidates(name,email,password)
                VALUES(?,?,?)
                """,(request.form['name'],
                     request.form['email'],
                     generate_password_hash(request.form['password'])))
            return redirect('/candidate/login')
        except IntegrityError:
            return "Email already exists"

    return """
    <form method="post">
    Name <input name="name"><br>
    Email <input name="email"><br>
    Password <input type="password" name="password"><br>
    <button>Register</button>
    </form>
    """

@app.route('/candidate/login', methods=['GET','POST'])
def candidate_login():
    if request.method == 'POST':
        with get_db() as conn:
            user = conn.execute(
                "SELECT id,password FROM candidates WHERE email=?",
                (request.form['email'],)
            ).fetchone()

        if user and check_password_hash(user[1], request.form['password']):
            session['user'] = user[0]
            session['type'] = 'candidate'
            return redirect('/candidate/dashboard')
        return "Invalid credentials"

    return """
    <form method="post">
    Email <input name="email"><br>
    Password <input type="password" name="password"><br>
    <button>Login</button>
    </form>
    """

@app.route('/candidate/update', methods=['POST'])
def update_candidate():
    file = request.files.get('resume')
    path = None
    if file:
        path = f"{UPLOAD_FOLDER}/{session['user']}.pdf"
        file.save(path)

    with get_db() as conn:
        conn.execute("""
        UPDATE candidates
        SET qualification=?, skills=?, resume_path=?
        WHERE id=?
        """,(request.form['qualification'],
             request.form['skills'],
             path,
             session['user']))
    return redirect('/candidate/dashboard')

@app.route('/candidate/dashboard')
def candidate_dashboard():
    if session.get('type') != 'candidate':
        return redirect('/candidate/login')

    with get_db() as conn:
        jobs = conn.execute("""
        SELECT id,title,location,salary,company FROM jobs
        """).fetchall()

        apps = conn.execute("""
        SELECT j.title,j.company,a.status
        FROM applications a
        JOIN jobs j ON a.job_id=j.id
        WHERE a.candidate_id=?
        """,(session['user'],)).fetchall()

        notifs = conn.execute("""
        SELECT message FROM notifications
        WHERE candidate_id=?
        ORDER BY timestamp DESC
        """,(session['user'],)).fetchall()

    return f"""
    <h2>Candidate Dashboard</h2>
    <a href="/logout">Logout</a><hr>

    <form method="post" enctype="multipart/form-data" action="/candidate/update">
    Qualification <input name="qualification"><br>
    Skills <input name="skills"><br>
    Resume <input type="file" name="resume"><br>
    <button>Save Profile</button>
    </form>

    <h3>Jobs</h3>
    {"".join(f'''
    <div>
    <b>{j[1]}</b> | {j[2]} | {j[3]}
    <form method="post" action="/apply/{j[0]}">
        <button>Apply</button>
    </form>
    </div><hr>''' for j in jobs)}

    <h3>Your Applications</h3>
    <ul>{"".join(f"<li>{a[0]} | {a[1]} | {a[2]}</li>" for a in apps)}</ul>

    <h3>Notifications</h3>
    <ul>{"".join(f"<li>{n[0]}</li>" for n in notifs)}</ul>
    """

@app.route('/apply/<int:job_id>', methods=['POST'])
def apply(job_id):
    with get_db() as conn:
        if conn.execute("""
        SELECT 1 FROM applications WHERE job_id=? AND candidate_id=?
        """,(job_id, session['user'])).fetchone():
            return redirect('/candidate/dashboard')

        cand = conn.execute("""
        SELECT qualification,skills FROM candidates WHERE id=?
        """,(session['user'],)).fetchone()

        job = conn.execute("""
        SELECT skills FROM jobs WHERE id=?
        """,(job_id,)).fetchone()

        resume_path = conn.execute("""
        SELECT resume_path FROM candidates WHERE id=?
        """,(session['user'],)).fetchone()[0]

        sim_score = similarity(" ".join(cand or []), job[0] if job else "")
        res_score = resume_similarity(resume_path, job[0] if job else "")

        conn.execute("""
        INSERT INTO applications(job_id,candidate_id,similarity_score,resume_score)
        VALUES(?,?,?,?)
        """,(job_id, session['user'], sim_score, res_score))


        

    return redirect('/candidate/dashboard')

@app.route('/resume/<int:candidate_id>')
def resume(candidate_id):
    with get_db() as conn:
        row = conn.execute("""
        SELECT resume_path FROM candidates WHERE id=?
        """,(candidate_id,)).fetchone()

    if not row or not row[0]:
        return "Resume not uploaded"

    return send_file(row[0], as_attachment=True)

# ================= RECRUITER =================
@app.route('/recruiter/register', methods=['GET','POST'])
def recruiter_register():
    if request.method == 'POST':
        try:
            with get_db() as conn:
                conn.execute("""
                INSERT INTO recruiters(name,email,password,company)
                VALUES(?,?,?,?)
                """,(request.form['name'],
                     request.form['email'],
                     generate_password_hash(request.form['password']),
                     request.form['company']))
            return redirect('/recruiter/login')
        except IntegrityError:
            return "Email already exists"

    return """
    <form method="post">
    Name <input name="name"><br>
    Email <input name="email"><br>
    Company <input name="company"><br>
    Password <input type="password" name="password"><br>
    <button>Register</button>
    </form>
    """

@app.route('/recruiter/login', methods=['GET','POST'])
def recruiter_login():
    if request.method == 'POST':
        with get_db() as conn:
            r = conn.execute("""
            SELECT id,password FROM recruiters WHERE email=?
            """,(request.form['email'],)).fetchone()

        if r and check_password_hash(r[1], request.form['password']):
            session['user'] = r[0]
            session['type'] = 'recruiter'
            return redirect('/recruiter/dashboard')

        return "Invalid credentials"

    return """
    <form method="post">
    Email <input name="email"><br>
    Password <input type="password" name="password"><br>
    <button>Login</button>
    </form>
    """

@app.route('/recruiter/dashboard')
def recruiter_dashboard():
    if session.get('type') != 'recruiter':
        return redirect('/recruiter/login')

    with get_db() as conn:
        apps = conn.execute("""
        SELECT a.id,c.id,c.name,c.skills,a.similarity_score,a.resume_score,a.status
        FROM applications a
        JOIN candidates c ON a.candidate_id=c.id
        JOIN jobs j ON a.job_id=j.id
        WHERE j.recruiter_id=?
        ORDER BY a.similarity_score DESC, a.resume_score DESC
        """,(session['user'],)).fetchall()


    rows = "".join(f"""
    <tr>
        <td>{a[2]}</td>
        <td>{a[3]}</td>
        <td>{round(a[4],2)}</td>
        <td>{round(a[5],2) if a[5] is not None else '-'}</td>
        <td>{a[6]}</td>
        <td><a href="/resume/{a[1]}">View Resume</a></td>
        <td>
            <form method="post" action="/update/{a[0]}/shortlisted">
                <button>Shortlist</button>
            </form>
            <form method="post" action="/update/{a[0]}/rejected">
                <button>Reject</button>
            </form>
        </td>
    </tr>
    """ for a in apps)

    return f"""
    <h2>Recruiter Dashboard</h2>
    <a href="/recruiter/shortlisted">View Shortlisted Candidates</a> |
    <a href="/logout">Logout</a><hr>

    <form method="post" action="/post_job">
    Title <input name="title"><br>
    Description <input name="description"><br>
    Location <input name="location"><br>
    Salary <input name="salary"><br>
    Skills <input name="skills"><br>
    Company <input name="company"><br>
    <button>Post Job</button>
    </form>

    <table border="1">
    <tr>
    <th>Name</th>
    <th>Skills</th>
    <th>Profile Match</th>
    <th>Resume Match</th>
    <th>Status</th>
    <th>Resume</th>
    <th>Action</th>
    </tr>

    {rows}
    </table>
    """

@app.route('/recruiter/shortlisted')
def recruiter_shortlisted():
    if session.get('type') != 'recruiter':
        return redirect('/recruiter/login')

    with get_db() as conn:
        data = conn.execute("""
        SELECT c.name,j.title,j.company,s.timestamp,c.id
        FROM shortlisted s
        JOIN candidates c ON s.candidate_id=c.id
        JOIN jobs j ON s.job_id=j.id
        WHERE s.recruiter_id=?
        """,(session['user'],)).fetchall()

    rows = "".join(f"""
    <tr>
        <td>{d[0]}</td>
        <td>{d[1]}</td>
        <td>{d[2]}</td>
        <td>{d[3]}</td>
        <td><a href="/resume/{d[4]}">View Resume</a></td>
    </tr>
    """ for d in data)

    return f"""
    <h2>Shortlisted Candidates</h2>
    <a href="/recruiter/dashboard">Back</a><hr>

    <table border="1">
    <tr>
        <th>Name</th>
        <th>Job Title</th>
        <th>Company</th>
        <th>Date</th>
        <th>Resume</th>
    </tr>
    {rows}
    </table>
    """

@app.route('/post_job', methods=['POST'])
def post_job():
    with get_db() as conn:
        conn.execute("""
        INSERT INTO jobs(recruiter_id,title,description,location,salary,skills,company)
        VALUES(?,?,?,?,?,?,?)
        """,(session['user'],
             request.form['title'],
             request.form['description'],
             request.form['location'],
             request.form['salary'],
             request.form['skills'],
             request.form['company']))
    return redirect('/recruiter/dashboard')

@app.route('/update/<int:app_id>/<status>', methods=['POST'])
def update(app_id, status):
    with get_db() as conn:
        conn.execute("UPDATE applications SET status=? WHERE id=?", (status, app_id))

        job_id, cand_id = conn.execute("""
        SELECT job_id,candidate_id FROM applications WHERE id=?
        """,(app_id,)).fetchone()

        job = conn.execute("""
        SELECT title,company FROM jobs WHERE id=?
        """,(job_id,)).fetchone()

        if status == "shortlisted":
            try:
                conn.execute("""
                INSERT INTO shortlisted(application_id,recruiter_id,candidate_id,job_id)
                VALUES (?,?,?,?)
                """,(app_id, session['user'], cand_id, job_id))
            except IntegrityError:
                pass

            msg = f"""We are pleased to inform you that you have been shortlisted for the position of {job[0]} at {job[1]}. Our recruitment team will contact you shortly regarding the next steps."""

        else:
            msg = f"""Thank you for your interest in the position of {job[0]} at {job[1]}. After careful consideration, we regret to inform you that your application was not successful."""

        conn.execute("""
        INSERT INTO notifications(candidate_id,message)
        VALUES(?,?)
        """,(cand_id, msg))

    return redirect('/recruiter/dashboard')

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, port=8080)

