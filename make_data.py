"""Generate and verify fixed mask rows using only the three local book files."""
import csv
import hashlib
import json
from pathlib import Path
import random
import re

from tokenizers import BertWordPieceTokenizer

HERE = Path(__file__).resolve().parent
HEADER = ('masked_text', 'label_1', 'label_2', 'label_3')
BOOKS = (
    ('The Adventures of Sherlock Holmes.txt', r'^[IVX]+\. [A-Z].+$', 12),
    ('The Return of Sherlock Holmes.txt', r'^THE ADVENTURE OF .+$', 13),
    ('Some later reminiscences of Sherlock Holmes.txt',
     r'^(?:The Adventure of .+|The Disappearance of Lady Frances Carfax|His Last Bow: .+)$', 7),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenizer():
    return BertWordPieceTokenizer(str(HERE / 'resources/tokenizer/vocab.txt'), lowercase=True)


def stories(path, pattern, expected):
    text = path.read_text(encoding='utf-8-sig')
    titles = list(re.finditer(pattern, text, re.MULTILINE))
    assert len(titles) == expected, (path, len(titles))
    for index, title in enumerate(titles):
        stop = titles[index + 1].start() if index + 1 < expected else len(text)
        yield title.group().strip(), ' '.join(text[title.end():stop].split())


def windows(text, encoded, rng):
    offsets = encoded.offsets
    start = 0
    while start < len(offsets):
        range_start, range_end = rng.choices(((12, 28), (29, 58), (59, 112)), (30, 60, 10))[0]
        end = min(start + rng.randint(range_start, range_end), len(offsets))
        while end < len(offsets) and encoded.tokens[end].startswith('##'):
            end += 1
        end = min(end, start + 124)
        left, right = offsets[start][0], offsets[end - 1][1]
        yield text[left:right], encoded.ids[start:end], [
            (lo - left, hi - left) for lo, hi in offsets[start:end]
        ]
        start = end


def rows_for_story(text, tok, rng, retain, stats):
    vocab = tok.id_to_token
    raw = tok.encode(text, add_special_tokens=False)
    for window, ids, spans in windows(text, raw, rng):
        if tok.encode(window, add_special_tokens=False).ids != ids:
            continue
        # Single WordPiece targets may be suffixes. Verify that replacing their exact
        # surface span leaves every unmasked token intact, then back off to singles.
        positions = [
            i for i, (lo, hi) in enumerate(spans)
            if ids[i] not in (0, 100, 101, 102, 103) and hi > lo
        ]
        rng.shuffle(positions)
        cursor = 0
        while cursor < len(positions):
            size = rng.choices((1, 2, 3), (20, 40, 40))[0]
            group = sorted(positions[cursor:cursor + size])
            cursor += size
            if rng.random() >= retain:
                continue
            candidates = [group]
            while candidates:
                group = candidates.pop()
                if not group:
                    continue
                masked = window
                expected = list(ids)
                for pos in reversed(group):
                    lo, hi = spans[pos]
                    masked = masked[:lo] + ' [MASK] ' + masked[hi:]
                    expected[pos] = 103
                masked = ' '.join(masked.split())
                actual = tok.encode(masked).ids
                if actual != [101, *expected, 102]:
                    stats['roundtrip_rejections'] += 1
                    if len(group) > 1:
                        candidates.extend([[pos] for pos in reversed(group)])
                    continue
                labels = [vocab(ids[pos]) for pos in group]
                assert len(actual) <= 128 and actual.count(103) == len(labels)
                yield (masked, *labels, *([''] * (3 - len(labels)))), tuple(ids)


def generate():
    tok = tokenizer()
    manifest = {'sources': [], 'train': {}, 'dev': {}}
    # Exact unmasked duplicates across split are excluded as well as source overlap.
    dev_contexts = set()
    for split, book_ids, passes in (
        ('dev', (2,), ((19041, 0.085),)),
        ('train', (0, 1), ((18973, 1.0), (25741, 0.31))),
    ):
        stats = {'rows': 0, 'labels': 0, 'roundtrip_rejections': 0, 'duplicate_rows': 0,
                 'shared_context_rows': 0}
        seen = set()
        path = HERE / f'{split}.csv'
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for book_id in book_ids:
                filename, pattern, count = BOOKS[book_id]
                source = HERE / 'books' / filename
                sections = list(stories(source, pattern, count))
                manifest['sources'].append(dict(split=split, book=filename, sha256=digest(source),
                                                stories=[title for title, _ in sections]))
                for seed, retain in passes:
                    rng = random.Random(seed + book_id)
                    for _, text in sections:
                        for row, context in rows_for_story(text, tok, rng, retain, stats):
                            if split == 'train' and context in dev_contexts:
                                stats['shared_context_rows'] += 1
                                continue
                            if row in seen:
                                stats['duplicate_rows'] += 1
                                continue
                            if split == 'dev':
                                dev_contexts.add(context)
                            seen.add(row)
                            writer.writerow(row)
                            stats['rows'] += 1
                            stats['labels'] += sum(bool(label) for label in row[1:])
        stats.update(bytes=path.stat().st_size, sha256=digest(path))
        manifest[split] = stats
    assert manifest['train']['bytes'] <= 50_000_000
    (HERE / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('train', 'dev')}), flush=True)
    return manifest


if __name__ == '__main__':
    generate()
