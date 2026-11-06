"""Answer-match metric; refusals receive zero on answerable scalar QA."""
import re

REFUSAL_PHRASE = 'do not have enough information'

def normalize(value):
    return re.sub(r'\s+', ' ', str(value or '').lower().strip())

def is_refusal(value):
    text = normalize(value)
    return not text or REFUSAL_PHRASE in text

def substring_em(answer, golds):
    """Bidirectional substring match after excluding the fixed refusal.

    Dataset-native unanswerability scoring is separate from this diagnostic.
    """
    if is_refusal(answer):
        return 0
    if not isinstance(golds, (tuple, list)):
        golds = [golds]
    answer = normalize(answer)
    return int(any(gold and (answer in gold or gold in answer)
                   for gold in map(normalize, golds)))

def word_boundary_substring_em(answer, golds):
    """Mechanical sensitivity check; not a semantic correctness judgment.

    Keep the primary metric's normalization and symmetric containment rule,
    adding Unicode word boundaries at the two ends of the contained string.
    This removes ball/football matches, but can also reject valid inflections.
    """
    if is_refusal(answer):
        return 0
    if not isinstance(golds, (tuple, list)):
        golds = [golds]
    answer = normalize(answer)
    def contains(value, text):
        return re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', text) is not None
    return int(any(gold and (contains(answer, gold) or contains(gold, answer))
                   for gold in map(normalize, golds)))

def regression_checks():
    assert substring_em('I do not have enough information to answer', 'No') == 0
    assert substring_em('I do not have enough information to answer.', ['no']) == 0
    assert substring_em('', ['']) == 0
    assert substring_em('No', 'no') == 1
    assert substring_em('  New   York ', ['New York City']) == 1
    assert substring_em('Paris', ['London', 'PARIS']) == 1
    assert substring_em('Oslo', 'Paris') == 0
    assert substring_em('a career in football', 'ball') == 1
    assert word_boundary_substring_em('a career in football', 'ball') == 0
    assert word_boundary_substring_em('The player is chasing a ball.', 'ball') == 1
    assert word_boundary_substring_em('No', 'no') == 1
    assert word_boundary_substring_em('I do not have enough information to answer', 'No') == 0
    assert word_boundary_substring_em('guitars', 'guitar') == 0

if __name__ == '__main__':
    regression_checks()
    print('PASS: refusal/No regression and unchanged non-refusal matching')
