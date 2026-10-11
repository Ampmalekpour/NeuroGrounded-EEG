import sqlite3
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS cars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER NOT NULL,
    photo TEXT
);
CREATE TABLE IF NOT EXISTS attrs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    car_id INTEGER NOT NULL REFERENCES cars(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS groups (
    chat_id INTEGER PRIMARY KEY,
    title TEXT
);
"""


class DB:
    def __init__(self, path: str):
        self.path = path
        with self._c() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _c(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- cars ----
    def add_car(self, brand, model, year, photo, attrs):
        with self._c() as c:
            cur = c.execute(
                "INSERT INTO cars(brand, model, year, photo) VALUES (?,?,?,?)",
                (brand, model, year, photo),
            )
            car_id = cur.lastrowid
            c.executemany(
                "INSERT INTO attrs(car_id, name, value) VALUES (?,?,?)",
                [(car_id, n, v) for n, v in attrs],
            )
            return car_id

    def delete_car(self, car_id):
        with self._c() as c:
            c.execute("DELETE FROM cars WHERE id=?", (car_id,))

    def brands(self):
        with self._c() as c:
            return [r[0] for r in c.execute("SELECT DISTINCT brand FROM cars ORDER BY brand")]

    def cars_by_brand(self, brand):
        with self._c() as c:
            return c.execute(
                "SELECT * FROM cars WHERE brand=? ORDER BY model, year", (brand,)
            ).fetchall()

    def car(self, car_id):
        with self._c() as c:
            return c.execute("SELECT * FROM cars WHERE id=?", (car_id,)).fetchone()

    def attrs(self, car_id):
        with self._c() as c:
            return c.execute("SELECT * FROM attrs WHERE car_id=? ORDER BY id", (car_id,)).fetchall()

    # ---- groups ----
    def add_group(self, chat_id, title):
        with self._c() as c:
            c.execute(
                "INSERT INTO groups(chat_id, title) VALUES (?,?) "
                "ON CONFLICT(chat_id) DO UPDATE SET title=excluded.title",
                (chat_id, title),
            )

    def remove_group(self, chat_id):
        with self._c() as c:
            c.execute("DELETE FROM groups WHERE chat_id=?", (chat_id,))

    def groups(self):
        with self._c() as c:
            return c.execute("SELECT * FROM groups ORDER BY title").fetchall()
