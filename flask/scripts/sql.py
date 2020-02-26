import psycopg2

def connect_database():
    conn = psycopg2.connect(database="tien", user="postgres")

    # Create a cursor object
    cursor = conn.cursor()
    return cursor


def generate_query(text):
    text = text.lower()
    keywords = text.split()
    query = """SELECT name FROM captions WHERE ("""
    for i in range(len(keywords)):
        if i == len(keywords) - 1:
            string = "caption LIKE '%{}%');".format(keywords[i])
        else:
            string = "caption LIKE '%{}%' AND ".format(keywords[i])
        query += string
    return query


def run_command(query, cursor):
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result 

def main(text):
    cursor = connect_database()
    query = generate_query(text)
    result = run_command(query, cursor)
    return result
