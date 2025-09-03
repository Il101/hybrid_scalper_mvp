from random import randint

def sm_score(symbol: str) -> float:
    return float(max(0, min(100, 50 + randint(-10, 10))))
