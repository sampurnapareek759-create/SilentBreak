"""Run SQL typed on stdin against silentbreak.duckdb and print the result."""
import sys
import duckdb
from src.config import DB_PATH

query = sys.stdin.read()
with duckdb.connect(str(DB_PATH), read_only=True) as con:
    print(con.sql(query))
