import sqlite3
import json

def inspect():
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM corrections")
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} corrections:")
    for row in rows:
        print(row)
        
    conn.close()

if __name__ == "__main__":
    inspect()
