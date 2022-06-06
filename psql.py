import psycopg2

class DB():
    def __init__(self) -> None:
        self.dbname='postgres'
        self.user='postgres'
        self.password='admin'
        self.host='localhost'


    def connect(self):
        conn = psycopg2.connect(dbname=self.dbname, user=self.user, 
                        password=self.password, host=self.host)
        conn.autocommit = True
        cursor = conn.cursor()

        return conn, cursor

    def query(self, q):
        conn, cursor = self.connect()
        
        cursor.execute(q)
        try:
            result = cursor.fetchall()
        except psycopg2.ProgrammingError:
            result = None
        cursor.close()
        conn.close()
        return result
