import sqlite3

def fix():
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    
    # Delete bad rules
    ids_to_delete = [12, 30]
    cursor.execute(f"DELETE FROM corrections WHERE id IN ({','.join(map(str, ids_to_delete))})")
    
    print(f"Deleted {cursor.rowcount} bad rules.")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    fix()
