"""失物宇宙：seekdb 保存关卡与玩家进度，Dify 负责叙事。"""
import json
import os
import secrets
from contextlib import asynccontextmanager

import pymysql
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


def connect():
    return pymysql.connect(host=os.environ['SEEKDB_HOST'],
        port=int(os.getenv('SEEKDB_PORT', '2881')), user=os.getenv('SEEKDB_USER', 'root'),
        password=os.environ['SEEKDB_PASSWORD'], database=os.getenv('SEEKDB_DATABASE', 'test'),
        charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)


@asynccontextmanager
async def lifespan(app):
    if not os.getenv('GAME_TOKEN'):
        raise RuntimeError('GAME_TOKEN must be configured')
    yield


app = FastAPI(title='失物宇宙', lifespan=lifespan)


class Turn(BaseModel):
    player: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    action: str = Field(min_length=1, max_length=100)


def advance(state, action, case):
    """Only deterministic rules decide wins; the model never sees unrevealed answers."""
    if action == '开始':
        return {'clues': 0, 'done': False}, '新一轮开始。' + case['intro']
    if state['done']:
        return state, '本轮已结束。输入「开始」重新挑战。'
    if action == '线索':
        n = state['clues']
        if n >= len(case['clues']):
            return state, '线索已全部公开，请输入「归还 姓名」。'
        return {**state, 'clues': n + 1}, case['clues'][n]
    if action.startswith('归还 '):
        if action[3:].strip() == case['owner']:
            return {**state, 'done': True}, f"归还成功！获得 {max(1, 4-state['clues'])} 颗星。" + case['ending']
        return state, '认领失败，物品拒绝跟此人离开。再找一条线索吧。'
    return state, '可用指令：开始、线索、归还 姓名。' + case['intro']


@app.post('/turn')
def turn(body: Turn, authorization: str = Header(default='')):
    if not secrets.compare_digest(authorization, 'Bearer ' + os.environ['GAME_TOKEN']):
        raise HTTPException(401, 'Invalid game token')
    try:
        with connect() as db:
            with db.cursor() as cur:
                cur.execute('SELECT payload FROM lost_cases WHERE id=%s', ('moon',))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(503, 'Run init_db.py first')
                case = json.loads(row['payload'])
                cur.execute('INSERT IGNORE INTO lost_players (id,state) VALUES (%s,%s)',
                    (body.player, json.dumps({'clues': 0, 'done': False})))
                cur.execute('SELECT state FROM lost_players WHERE id=%s FOR UPDATE', (body.player,))
                state, event = advance(json.loads(cur.fetchone()['state']), body.action, case)
                cur.execute('UPDATE lost_players SET state=%s WHERE id=%s',
                    (json.dumps(state), body.player))
            db.commit()
        return {'event': event, 'state': state, 'source': 'seekdb/lost_cases/moon',
                'rule': '只渲染event，不添加线索、人物事实、分数或成功判定。'}
    except pymysql.MySQLError:
        raise HTTPException(503, 'seekdb unavailable; progress was not confirmed') from None
