from flask import Flask, request, session, redirect, send_file
import sqlite3, os
from werkzeug.security import generate_password_hash, check_password_hash
from sqlite3 import IntegrityError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ----------------
def get_db():
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
            experience INTEGER,
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
            requirements TEXT,
            company TEXT
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS applications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            candidate_id INTEGER,
            similarity_score REAL,
            status TEXT DEFAULT 'pending'
        )
        """)

create_tables()

# ---------------- SIMILARITY ----------------
def similarity(a, b):
    tfidf = TfidfVectorizer()
    v = tfidf.fit_transform([a, b])
    return float(cosine_similarity(v[0], v[1])[0][0])

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ---------------- HOME ----------------
@app.route('/')
def home():
    return """
    <h1>Job Portal</h1>
    <a href="/candidate/register">Candidate Register</a> |
    <a href="/candidate/login">Candidate Login</a><br><br>
    <a href="/recruiter/register">Recruiter Register</a> |
    <a href="/recruiter/login">Recruiter Login</a>
    """

# ---------------- CANDIDATE REGISTER ----------------
@app.route('/candidate/register', methods=['GET','POST'])
def candidate_register():
    if request.method == 'POST':
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO candidates(name,email,password) VALUES(?,?,?)",
                    (request.form['name'], request.form['email'],
                     generate_password_hash(request.form['password']))
                )
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

# ---------------- CANDIDATE LOGIN ----------------
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

# ---------------- CANDIDATE DASHBOARD ----------------
@app.route('/candidate/dashboard')
def candidate_dashboard():
    if session.get('type') != 'candidate':
        return redirect('/candidate/login')

    with get_db() as conn:
        jobs = conn.execute("SELECT id,title,company FROM jobs").fetchall()
        apps = conn.execute("""
        SELECT j.title,j.company,a.status
        FROM applications a
        JOIN jobs j ON a.job_id=j.id
        WHERE a.candidate_id=?
        """,(session['user'],)).fetchall()

    jobs_html = ""
    for j in jobs:
        jobs_html += f"""
        <div>
        <b>{j[1]}</b> | {j[2]}
        <form method="post" action="/apply/{j[0]}">
            <button>Apply</button>
        </form>
        </div><hr>
        """

    apps_html = ""
    for a in apps:
        apps_html += f"<li>{a[0]} | {a[1]} | Status: {a[2]}</li>"

    return f"""
    <h2>Candidate Dashboard</h2>
    <a href="/logout">Logout</a><hr>

    <h3>Update Profile</h3>
    <form method="post" enctype="multipart/form-data" action="/update_candidate">
    Qualification <input name="qualification"><br>
    Skills <input name="skills"><br>
    Experience <input name="experience"><br>
    Resume <input type="file" name="resume"><br>
    <button>Update</button>
    </form>

    <h3>Available Jobs</h3>
    {jobs_html}

    <h3>Your Applications</h3>
    <ul>{apps_html}</ul>
    """

# ---------------- UPDATE CANDIDATE ----------------
@app.route('/update_candidate', methods=['POST'])
def update_candidate():
    file = request.files['resume']
    path = f"{UPLOAD_FOLDER}/{session['user']}.pdf"
    file.save(path)

    with get_db() as conn:
        conn.execute("""
        UPDATE candidates
        SET qualification=?, skills=?, experience=?, resume_path=?
        WHERE id=?
        """,(request.form['qualification'],
             request.form['skills'],
             request.form['experience'],
             path,
             session['user']))
    return redirect('/candidate/dashboard')

# ---------------- APPLY JOB ----------------
@app.route('/apply/<int:job_id>', methods=['POST'])
def apply(job_id):
    with get_db() as conn:
        cand = conn.execute(
            "SELECT qualification,skills,experience FROM candidates WHERE id=?",
            (session['user'],)
        ).fetchone()

        job = conn.execute(
            "SELECT requirements FROM jobs WHERE id=?",
            (job_id,)
        ).fetchone()

        score = similarity(" ".join(map(str,cand)), job[0])

        conn.execute("""
        INSERT INTO applications(job_id,candidate_id,similarity_score)
        VALUES(?,?,?)
        """,(job_id, session['user'], score))

    return redirect('/candidate/dashboard')

# ---------------- DOWNLOAD RESUME ----------------
@app.route('/resume/<int:candidate_id>')
def resume(candidate_id):
    with get_db() as conn:
        path = conn.execute(
            "SELECT resume_path FROM candidates WHERE id=?",
            (candidate_id,)
        ).fetchone()[0]
    return send_file(path, as_attachment=True)

# ---------------- RECRUITER REGISTER ----------------
@app.route('/recruiter/register', methods=['GET','POST'])
def recruiter_register():
    if request.method == 'POST':
        with get_db() as conn:
            conn.execute("""
            INSERT INTO recruiters(name,email,password,company)
            VALUES(?,?,?,?)
            """,(request.form['name'],
                 request.form['email'],
                 generate_password_hash(request.form['password']),
                 request.form['company']))
        return redirect('/recruiter/login')

    return """
    <form method="post">
    Name <input name="name"><br>
    Email <input name="email"><br>
    Password <input type="password" name="password"><br>
    Company <input name="company"><br>
    <button>Register</button>
    </form>
    """

# ---------------- RECRUITER LOGIN ----------------
@app.route('/recruiter/login', methods=['GET','POST'])
def recruiter_login():
    if request.method == 'POST':
        with get_db() as conn:
            r = conn.execute(
                "SELECT id,password FROM recruiters WHERE email=?",
                (request.form['email'],)
            ).fetchone()

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

# ---------------- POST JOB ----------------
@app.route('/post_job', methods=['POST'])
def post_job():
    with get_db() as conn:
        conn.execute("""
        INSERT INTO jobs(recruiter_id,title,requirements,company)
        VALUES(?,?,?,?)
        """,(session['user'],
             request.form['title'],
             request.form['requirements'],
             request.form['company']))
    return redirect('/recruiter/dashboard')

# ---------------- RECRUITER DASHBOARD ----------------
@app.route('/recruiter/dashboard')
def recruiter_dashboard():
    if session.get('type') != 'recruiter':
        return redirect('/recruiter/login')

    with get_db() as conn:
        apps = conn.execute("""
        SELECT a.id,c.id,c.name,c.skills,c.experience,c.resume_path,
               a.similarity_score,a.status
        FROM applications a
        JOIN candidates c ON a.candidate_id=c.id
        """).fetchall()

    pending = ""
    shortlisted = ""

    for a in apps:
        row = f"""
        <tr>
        <td>{a[2]}</td>
        <td>{a[3]}</td>
        <td>{a[4]}</td>
        <td><a href="/resume/{a[1]}">Download</a></td>
        <td>{round(a[6],2)}</td>
        <td>{a[7]}</td>
        <td>
            <form method="post" action="/update/{a[0]}/shortlisted" style="display:inline">
                <button>Shortlist</button>
            </form>
            <form method="post" action="/update/{a[0]}/rejected" style="display:inline">
                <button>Reject</button>
            </form>
        </td>
        </tr>
        """

        if a[7] == "shortlisted":
            shortlisted += row
        else:
            pending += row

    return f"""
    <h2>Recruiter Dashboard</h2>
    <a href="/logout">Logout</a><hr>

    <h3>Post Job</h3>
    <form method="post" action="/post_job">
    Title <input name="title"><br>
    Requirements <input name="requirements"><br>
    Company <input name="company"><br>
    <button>Post</button>
    </form>

    <h3>Applications</h3>
    <table border="1">
    <tr>
    <th>Name</th><th>Skills</th><th>Experience</th>
    <th>Resume</th><th>Score</th><th>Status</th><th>Action</th>
    </tr>
    {pending}
    </table>

    <h3>Shortlisted Candidates</h3>
    <table border="1">
    <tr>
    <th>Name</th><th>Skills</th><th>Experience</th>
    <th>Resume</th><th>Score</th><th>Status</th><th>Action</th>
    </tr>
    {shortlisted}
    </table>
    """

# ---------------- UPDATE STATUS ----------------
@app.route('/update/<int:app_id>/<status>', methods=['POST'])
def update(app_id, status):
    with get_db() as conn:
        conn.execute(
            "UPDATE applications SET status=? WHERE id=?",
            (status, app_id)
        )
    return redirect('/recruiter/dashboard')

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, port=8080)

