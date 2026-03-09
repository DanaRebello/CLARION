import sqlite3

# Connect to your existing database
conn = sqlite3.connect("users.db")
c = conn.cursor()

# Add 'message' column to 'candidates' table
try:
    c.execute("ALTER TABLE candidates ADD COLUMN message TEXT DEFAULT ''")
    print("Column 'message' added successfully!")
except sqlite3.OperationalError:
    print("Column 'message' already exists. No changes made.")

conn.commit()
conn.close()
