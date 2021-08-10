import psycopg2

# Connect to the database


class Connection():
    """Connection instance for  databce

    Attributes
    ----------
    conn : type
        The database connection instance
    cur : type
        The database active cursor

    """

    def __init__(self, dev=False):
        if dev:
            self.conn = psycopg2.connect(host="chunee.db.elephantsql.com", database="cdlcxsow", user="cdlcxsow", password="JmDqdlizdn0C70HRv8eQDJJa5wyHfHx5")
        else:
            self.conn = psycopg2.connect(host="queenie.db.elephantsql.com", database="dyevhwdt", user="dyevhwdt", password="PcWHI8bYvT2Gz1hYWqOEDsAhKtqjipQM")
        self.cur = self.conn.cursor()

    def execute(self, command, returns=False):
        self.cur.execute(command)
        if returns:
            return self.cur.fetchall()
        pass


if __name__ == "__main__":
    conn = Connection()
