from collections import Counter, defaultdict
import math

import numpy as np

from learning.codec import numeric, require_fields, unique

PRODUCT = '课程意见研读'
KEYS = {'feedback': ['document'], 'terminology': ['document', 'original', 'applies_from']}


def validate(tables):
    if set(tables) != set(KEYS):
        raise ValueError("资料表集合与产品契约不一致")
    for name, fields in KEYS.items():
        require_fields(tables[name], fields)
        unique(tables[name], fields)
    documents = {row["document"] for row in tables["feedback"]}
    for row in tables["feedback"]:
        if not isinstance(row["text"], str) or not row["text"].strip() or row["term"] < 0:
            raise ValueError("反馈文本或学期不合法")
    for row in tables["terminology"]:
        if row["document"] not in documents or not row["original"] or not row["preferred"]:
            raise ValueError("术语意见无对应文本")


def overview(tables, parameters):
    terms = defaultdict(Counter)
    for row in tables["feedback"]:
        terms[row["term"]].update(row["text"].split())
    return {"terms": [{"term": term, "words": sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:20]} for term, counts in sorted(terms.items())],
            "channels": dict(Counter(row["channel"] for row in tables["feedback"])),
            "people": len({row["person"] for row in tables["feedback"]})}


ANALYSES = {"overview": overview}
