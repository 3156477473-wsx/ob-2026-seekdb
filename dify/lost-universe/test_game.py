import json
import unittest
from pathlib import Path
from app import advance


class GameTest(unittest.TestCase):
    def test_complete_round(self):
        case = json.loads(Path(__file__).with_name('case.json').read_text(encoding='utf-8'))
        state, _ = advance({}, '开始', case)
        for count in range(3):
            state, text = advance(state, '线索', case)
            self.assertEqual(state['clues'], count + 1)
            self.assertEqual(text, case['clues'][count])
        self.assertIn('全部', advance(state, '线索', case)[1])
        self.assertFalse(advance(state, '归还 白羽', case)[0]['done'])
        state, text = advance(state, '归还 小满', case)
        self.assertTrue(state['done'])
        self.assertIn('成功', text)
        self.assertIn('已结束', advance(state, '线索', case)[1])
        self.assertEqual(advance(state, '开始', case)[0], {'clues': 0, 'done': False})


if __name__ == '__main__':
    unittest.main()
