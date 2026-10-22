"""Writes a Kaggle submission (template order, header user_id,tv_show_id, five space-separated ids) and validates it:
1,260 rows, same user order as the template, exactly 5 distinct ids per row, every id in the test-window pool."""
import numpy as np
from common import load_test_users, ROOT


def write(name, split, user_ix, P):
    """user_ix: indices into split.users for the template rows; P: (n, 5) item indices."""
    tu, sub = load_test_users()
    assert np.array_equal(split.users[user_ix], tu)
    pool = set(split.pool.tolist())
    lines = ["user_id,tv_show_id"]
    for u, row in zip(tu, P):
        ids = split.items[row].tolist()
        assert len(ids) == 5 and len(set(ids)) == 5 and all(i in pool for i in ids), (u, ids)
        lines.append(f"{u},{' '.join(str(i) for i in ids)}")
    f = ROOT / "submissions" / f"{name}.csv"
    f.parent.mkdir(exist_ok=True)
    f.write_text("\n".join(lines) + "\n")
    return f
