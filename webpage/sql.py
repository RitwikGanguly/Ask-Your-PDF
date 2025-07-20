import sqlite3

# Connect to the chroma.sqlite3 database
db_path = '/path/to/your/chroma.sqlite3'  
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Function to list all collections (tables) in the database
def list_collections():
    # cursor.execute("SELECT name FROM sqlite_master;")

    cursor.execute("SELECT * FROM collections;")
    collections = cursor.fetchall()
    if collections:
        print("Collections (Tables) in the database:")
        for collection in collections:
            print(collection[0])
    else:
        print("No collections found in the database.")

# Function to delete a specific collection (table)
def delete_collection(collection_name):
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {collection_name};")
        conn.commit()
        print(f"Collection '{collection_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting collection: {e}")

# Function to delete all collections (tables)
def delete_all_collections():
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    collections = cursor.fetchall()
    if collections:
        for collection in collections:
            delete_collection(collection[0])
        print("All collections deleted successfully.")
    else:
        print("No collections to delete.")

# View all collections
list_collections()


conn.close()
