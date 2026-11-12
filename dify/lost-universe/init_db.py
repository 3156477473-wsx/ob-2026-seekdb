import json
from pathlib import Path
from app import connect

if __name__ == '__main__':
    case = json.loads(Path(__file__).with_name('case.json').read_text(encoding='utf-8'))
    with connect() as db:
        with db.cursor() as cur:
            cur.execute('CREATE TABLE IF NOT EXISTS lost_cases (id VARCHAR(32) PRIMARY KEY, payload TEXT NOT NULL)')
            cur.execute('CREATE TABLE IF NOT EXISTS lost_players (id VARCHAR(128) PRIMARY KEY, state TEXT NOT NULL)')
            cur.execute('INSERT INTO lost_cases (id,payload) VALUES (%s,%s) ON DUPLICATE KEY UPDATE payload=VALUES(payload)',
                        ('moon', json.dumps(case, ensure_ascii=False)))
        db.commit()
    print('seekdb: moon case initialized; existing player progress preserved')
