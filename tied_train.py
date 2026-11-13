"""One frozen-parent experiment with a supervised head and then a learned gate."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from make_data import BOOKS, HERE, digest, rows_for_story, stories, tokenizer
from student_model import StudentModel, load_model
from train import batches, collate, load_csv, save_state, setup, target_groups, write_json
from v1_model import HEAD_WIDTH, VOCAB_SIZE, load_model as load_v1

V1 = HERE.parent / 'clean_slate'
PARENT_SHA = 'beb596b2cce81af37b32575b8e8169ca1ae3a5cce77ad226a4b854607a79a7b3'
ZIP_SHA = 'b71210e34ed78b6423edbafea742f47b8a45e1cb74a2a1ac5bab4f1166ba53aa'
TRAIN_SHA = '531eddba991dac9cb481c28e073e6e6fdd0055d1ec47470da009e78cb9209528'
DEV_SHA = 'a83170077a8fc4801168190bd6fa6369d6e8c3295b23c47ca679324b4fc18738'
SEED = 403102
MASK_BATCH = 384
HEAD_EPOCHS = 6
MIX_EPOCHS = 2
HEAD_LR = 0.004
GATE_LR = 0.003
MIX_HEAD_LR = 0.00035


def verify_sources():
    assert digest(HERE / 'train.csv') == TRAIN_SHA
    assert digest(HERE / 'dev.csv') == DEV_SHA
    assert (HERE / 'train.csv').stat().st_size <= 50_000_000
    manifest = json.loads((HERE / 'data_manifest.json').read_text())
    for source in manifest['sources']:
        assert digest(HERE / 'books' / source['book']) == source['sha256']
    assert {x['book'] for x in manifest['sources'] if x['split'] == 'train'} == {BOOKS[0][0], BOOKS[1][0]}
    assert {x['book'] for x in manifest['sources'] if x['split'] == 'dev'} == {BOOKS[2][0]}


def story_labels(tok):
    filename, pattern, count = BOOKS[2]
    rng = random.Random(19043)
    seen = set()
    indices = []
    names = []
    with (HERE / 'dev.csv').open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        next(reader)
        for i, (name, text) in enumerate(stories(HERE / 'books' / filename, pattern, count)):
            names.append(name)
            for row, _ in rows_for_story(text, tok, rng, 0.085, {'roundtrip_rejections': 0}):
                if row in seen:
                    continue
                seen.add(row)
                assert tuple(next(reader)) == row
                indices.append(i)
        assert next(reader, None) is None
    return names, np.asarray(indices, dtype=np.int64)


def initialize(parent, counts, device, pilot):
    if pilot:
        assert digest(parent) == PARENT_SHA
        assert digest(V1 / 'submission/50028181.zip') == ZIP_SHA
    else:
        history = json.loads((parent.parent / 'training_history.json').read_text())
        assert digest(parent) == history['model_sha256']
        assert history['train_sha256'] == TRAIN_SHA and history['dev_sha256'] == DEV_SHA
    torch.manual_seed(SEED)
    model = StudentModel()
    parent_state = torch.load(parent, map_location='cpu', weights_only=True)
    expected = {key for key in model.state_dict() if not key.startswith(('tied.', 'gate.')) and key != 'tied_only'}
    assert set(parent_state) == expected
    head_before = {key: value.clone() for key, value in model.state_dict().items() if key not in expected}
    missing, unexpected = model.load_state_dict(parent_state, strict=False)
    assert set(missing) == set(head_before) and not unexpected
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in head_before.items())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in parent_state.items())
    reference = HERE / 'resources/encoder_reference.pt'
    inventory = json.loads((HERE / 'resources/provenance.json').read_text())
    assert digest(reference) == inventory['filtered_reference_sha256']
    original = torch.load(reference, weights_only=True, map_location='cpu')
    assert set(original) == {f'bert.{key}' for key in model.bert.state_dict()} == set(inventory['allowed_keys'])
    assert len(original) == 37
    embed_name = 'bert.embeddings.word_embeddings.weight'
    original_e, current_e = original[embed_name], parent_state[embed_name]
    cosine = float(F.cosine_similarity(original_e, current_e).mean())
    assert cosine > 0.95
    with torch.no_grad():
        model.tied.bias.copy_(torch.from_numpy(np.log1p(counts)))
    model.requires_grad_(False).to(device).eval()
    embedding = model.output_embedding
    assert isinstance(embedding, torch.nn.Parameter)
    assert embedding is model.bert.embeddings.word_embeddings.weight
    registered = [(name, p) for name, p in model.named_parameters(remove_duplicate=False)]
    assert [name for name, p in registered if p.untyped_storage().data_ptr() == embedding.untyped_storage().data_ptr()] == [embed_name]
    assert [key for key, value in model.state_dict().items() if value.shape == (VOCAB_SIZE, 128)] == [embed_name]
    total = sum(p.numel() for p in model.parameters())
    assert total == 7_064_891 and total <= 9_000_000
    return model, dict(parameters=total, encoder_tensors=len(original), embedding_cosine_vs_original=cosine,
                       parent_sha256=digest(parent), reference_sha256=digest(reference),
                       fresh_parameters=[k for k in head_before if k != 'tied_only'],
                       shared_parameter=True, serialized_embedding_tables=1)


@torch.inference_mode()
def cache_train(model, train, device):
    """Cache representations and exact Path A gold log-probs from frozen V1."""
    size = sum(len(item[2]) for item in train)
    features = torch.empty((size, HEAD_WIDTH), device=device)
    labels = torch.empty(size, dtype=torch.long, device=device)
    path_a = torch.empty(size, device=device)
    inverse = model.decoder.vocab_to_rank.cpu().numpy()
    offset = 0
    for batch in batches(train, None, 192):
        inputs, targets = collate(batch, device)
        hidden = model.masked_features(*inputs)
        root, tails = target_groups(targets, inverse, device)
        decoder = model.decoder
        rows = torch.arange(len(targets), device=device)
        logp = decoder.root(decoder.features(hidden, 0)).log_softmax(-1)[rows, root]
        for band, (indices, ranks) in enumerate(tails, 1):
            if ranks.numel():
                logits = decoder.tails[band - 1](decoder.features(hidden[indices], band))
                logp[indices] += logits.log_softmax(-1).gather(1, ranks[:, None]).squeeze(1)
        n = len(targets)
        features[offset:offset+n] = hidden
        labels[offset:offset+n] = torch.as_tensor(targets, device=device, dtype=torch.long)
        path_a[offset:offset+n] = logp + decoder.log_main_mass
        if offset == 0:
            y = labels[:n]
            assert torch.isneginf(decoder.backoff_log_probs[y]).all()
            reference = -F.cross_entropy(decoder(hidden).float(), y, reduction='none')
            # Full-vocabulary normalization on CPU accumulates slightly differently.
            torch.testing.assert_close(path_a[:n], reference,
                                       atol=0.001 if device.type == 'cpu' else 0.0002, rtol=0)
        offset += n
    assert offset == size and torch.isfinite(features).all() and torch.isfinite(path_a).all()
    return features, labels, path_a


@torch.inference_mode()
def cache_dev(model, dev, story_ids, counts, device):
    features, labels, a_logp, frequency, story, mask_counts = [], [], [], [], [], []
    for start in range(0, len(dev), 48):
        examples = dev[start:start + 48]
        inputs, targets = collate(examples, device)
        hidden = model.masked_features(*inputs)
        y = torch.as_tensor(targets, device=device, dtype=torch.long)
        sizes = np.array([len(x[2]) for x in examples])
        features.append(hidden)
        labels.append(y)
        a_logp.append(-F.cross_entropy(model.decoder(hidden).float(), y, reduction='none'))
        frequency.append(counts[targets])
        story.append(np.repeat(story_ids[start:start + len(examples)], sizes))
        mask_counts.append(np.repeat(sizes, sizes))
    return (torch.cat(features), torch.cat(labels), torch.cat(a_logp),
            np.concatenate(frequency), np.concatenate(story), np.concatenate(mask_counts))


def summarize(loss, baseline, frequency, story, mask_counts, story_names):
    gains = baseline - loss
    trim_count = math.ceil(len(gains) * 0.01)
    groups = {'ordinary': np.ones(len(loss), dtype=bool), 'common': frequency >= 10,
              'rare': (frequency > 0) & (frequency < 10), 'zero_target': frequency == 0,
              'one_mask': mask_counts == 1, 'two_masks': mask_counts == 2, 'three_masks': mask_counts == 3,
              'multiple_masks': mask_counts >= 2}
    def metrics(selected):
        return dict(ce=float(loss[selected].mean(dtype=np.float64)),
                    improvement=float(gains[selected].mean(dtype=np.float64)), masks=int(selected.sum()))
    return dict(groups={key: metrics(sel) for key, sel in groups.items()},
                stories={name: metrics(story == i) for i, name in enumerate(story_names)},
                tail_excluded_improvement=float(np.sort(gains)[:-trim_count].mean(dtype=np.float64)),
                excluded_favorable_tokens=trim_count)


@torch.inference_mode()
def measure(model, cache, names):
    features, labels, a_logp, *groups = cache
    a = -a_logp.cpu().numpy()
    b_losses, mix_losses, weights = [], [], []
    for start in range(0, len(a), MASK_BATCH):
        s = slice(start, start + MASK_BATCH)
        hidden = features[s]
        b = -F.cross_entropy(model.path_b(hidden).float(), labels[s], reduction='none')
        b_losses.append(-b.cpu().numpy())
        mix_losses.append(-model.mix(hidden, a_logp[s], b).cpu().numpy())
        weights.append(model.gate(hidden).sigmoid().cpu().numpy())
    return dict(A=summarize(a, a, *groups, names),
                B=summarize(np.concatenate(b_losses), a, *groups, names),
                mix=summarize(np.concatenate(mix_losses), a, *groups, names),
                mean_gate=float(np.concatenate(weights).mean()))


def ce(metrics, path):
    return metrics[path]['groups']['ordinary']['ce']


def optimize(model, data, optimizer, epoch_count, device, mix, callback):
    features, labels, path_a = data
    rng = np.random.default_rng(SEED + int(mix))
    steps = math.ceil(len(labels) / MASK_BATCH)
    rates = [group['lr'] for group in optimizer.param_groups]
    total_steps = epoch_count * steps
    registered = [p for group in optimizer.param_groups for p in group['params']]
    assert len({id(p) for p in registered}) == len(registered)
    embedding = model.output_embedding
    assert not embedding.requires_grad
    embedding_groups = [group for group in optimizer.param_groups
                        if any(p is embedding for p in group['params'])]
    assert len(embedding_groups) == int(mix)
    assert all(group['lr'] == 0 for group in embedding_groups)
    trainable = [p for p in registered if p.requires_grad]
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in trainable}
    for epoch in range(1, epoch_count + 1):
        order = rng.permutation(len(labels))
        running = torch.zeros((), device=device)
        t0 = monotonic()
        for batch, start in enumerate(range(0, len(labels), MASK_BATCH), 1):
            step = (epoch - 1) * steps + batch
            phase = max(0, (step - 150) / (total_steps - 150))
            scale = min(1., (step + 25) / 150) * (0.18 + 0.82 * (1 + math.cos(math.pi * phase)) / 2)
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate * scale
            ix = torch.as_tensor(order[start:start + MASK_BATCH], device=device)
            hidden = features[ix]
            optimizer.zero_grad(set_to_none=True)
            b_loss = F.cross_entropy(model.path_b(hidden).float(), labels[ix], reduction='none')
            loss = -model.mix(hidden, path_a[ix], -b_loss).mean() if mix else b_loss.mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 2.0, foreach=False)
            optimizer.step()
            running += loss.detach() * len(ix)
            if batch % 300 == 0 or batch == steps:
                assert math.isfinite(float(norm)) and math.isfinite(float(running))
                print(f'{"B" if mix else "A"} epoch {epoch} step {batch}/{steps} train CE '
                      f'{float(running) / min(batch * MASK_BATCH, len(labels)):.5f} '
                      f'elapsed {monotonic()-t0:.1f}s', flush=True)
        callback(epoch, step, float(running) / len(labels), monotonic() - t0)


def accuracy_gate(metrics):
    groups = metrics['groups']
    story_values = list(metrics['stories'].values())
    wins = sum(item['improvement'] > 0 for item in story_values[:6])
    tests = dict(ordinary=groups['ordinary']['improvement'] >= 0.03, stories=wins >= 4,
                 trimmed=metrics['tail_excluded_improvement'] > 0,
                 common=groups['common']['improvement'] >= -0.02,
                 rare=groups['rare']['improvement'] >= -0.15)
    return dict(passed=all(tests.values()), tests=tests, wins_first_six=wins,
                wins_all=sum(item['improvement'] > 0 for item in story_values))


@torch.inference_mode()
def cpu_audit(parent, candidates, dev, out):
    import socket

    def deny(*args, **kwargs):
        raise RuntimeError('Unexpected network access during offline CPU inference')
    original_socket = socket.socket
    socket.socket = deny
    try:
        device = torch.device('cpu')
        models = {'A': load_v1(parent, device)}
        models.update({key: load_model(path, device) for key, path in candidates.items()})
        samples = [
            torch.tensor([101, 103, 102, 0, 0, 0, 0]),
            torch.tensor([101, 103, 1996, 103, 102, 0, 0]),
            torch.tensor([101, 103, 1996, 103, 1996, 103, 102]),
        ]
        ids = torch.stack(samples)
        result = {}
        for key, model in models.items():
            def predict(x):
                return model(x, x.ne(0).long(), x.eq(103))
            if key != 'A':
                assert model.output_embedding is model.bert.embeddings.word_embeddings.weight
                assert sum(p.numel() for p in model.parameters()) == 7_064_891
                logits = predict(ids)
                padded = predict(F.pad(ids, (0, 128 - ids.shape[1])))
                singles = torch.cat([predict(x[None]) for x in samples])
                error = max(float((logits - padded).abs().max()), float((logits - singles).abs().max()))
                assert logits.shape == (6, VOCAB_SIZE) and torch.isfinite(logits).all() and error < 0.001
            predict(ids)
            n, total, elapsed = 0, 0., 0.
            for start in range(0, len(dev), 16):
                (x, _, _, _), targets = collate(dev[start:start+16], device)
                t0 = monotonic()
                logits = predict(x)
                elapsed += monotonic() - t0
                assert torch.isfinite(logits).all()
                total += float(F.cross_entropy(logits, torch.from_numpy(targets).long(), reduction='sum'))
                n += len(targets)
            result[key] = dict(seconds=elapsed, ce=total/n, masks=n)
        assert all(value['masks'] == 6093 for value in result.values())
        for key, info in result.items():
            info['relative_runtime'] = info['seconds'] / result['A']['seconds']
            info['practical'] = info['seconds'] <= 30 and info['relative_runtime'] <= 2
        write_json(out / 'cpu.json', result)
        return result
    finally:
        socket.socket = original_socket


def run(parent, out, pilot=True):
    device = setup()
    start_time = monotonic()
    out.mkdir(exist_ok=True)
    verify_sources()
    tok = tokenizer()
    train = load_csv(HERE / 'train.csv', tok)
    dev = load_csv(HERE / 'dev.csv', tok)
    names, story = story_labels(tok)
    counts = np.bincount(np.concatenate([x[2] for x in train]), minlength=VOCAB_SIZE)
    model, receipt = initialize(parent, counts, device, pilot)
    record = dict(device=str(device), **receipt, train_sha256=TRAIN_SHA, dev_sha256=DEV_SHA, epochs=[],
                  budget=dict(head_epochs=HEAD_EPOCHS, mix_epochs=MIX_EPOCHS, masks_per_step=MASK_BATCH,
                              head_lr=HEAD_LR, gate_lr=GATE_LR, mix_head_lr=MIX_HEAD_LR),
                  criteria=dict(min_ce_gain=0.03, min_story_wins_first_six=4, trim_favorable_fraction=0.01,
                                common_max_regression=0.02, rare_max_regression=0.15, cpu_max_ratio=2))
    cache = cache_dev(model, dev, story, counts, device)
    initial = measure(model, cache, names)
    baseline = ce(initial, 'A')
    expected_masks = sum(len(row[2]) for row in dev)
    features, labels, logp, frequencies, story_ids, mask_counts = cache
    assert features.shape == (expected_masks, HEAD_WIDTH)
    assert labels.shape == logp.shape == frequencies.shape == story_ids.shape == mask_counts.shape == (expected_masks,)
    assert torch.isfinite(features).all() and torch.isfinite(logp).all()
    assert all(math.isfinite(ce(initial, key)) for key in ('A', 'B', 'mix'))
    assert initial['A']['groups']['ordinary']['masks'] == expected_masks
    record['initial'] = initial
    write_json(out / 'history.json', record)
    print(f'Verified matched Path A {baseline:.6f}, initial B {ce(initial, "B"):.6f}. Caching frozen train features.',
          flush=True)
    data = cache_train(model, train, device)
    print(f'Cached {len(data[1]):,} masked training targets in {monotonic()-start_time:.1f}s total. '
          f'Training {HEAD_EPOCHS}+{MIX_EPOCHS} full passes.', flush=True)

    best = {'B': math.inf, 'mix': math.inf}
    best_metrics = {}
    choices = {}
    def assess(stage, epoch, steps, training_ce, elapsed):
        result = measure(model, cache, names)
        assert abs(ce(result, 'A') - baseline) < 1e-10
        item = dict(stage=stage, epoch=epoch, steps=steps, train_ce=training_ce, seconds=elapsed, dev=result)
        checkpoint = out / f'{stage}_{epoch:02d}.pt'
        save_state(model, checkpoint)
        for key in (('B',) if stage == 'head' else ('B', 'mix')):
            value = ce(result, key)
            assert math.isfinite(value)
            if value < best[key]:
                best[key], best_metrics[key], choices[key] = value, result[key], checkpoint
        record['epochs'].append(item)
        record['best'] = {key: {'ce': best[key], 'checkpoint': path.name} for key, path in choices.items()}
        write_json(out / 'history.json', record)
        print(f'{stage} epoch {epoch} dev A {ce(result, "A"):.6f} B {ce(result, "B"):.6f} '
              f'mix {ce(result, "mix"):.6f}, mean gate {result["mean_gate"]:.4f}', flush=True)

    model.tied.requires_grad_(True)
    head_optimizer = torch.optim.AdamW(model.tied.parameters(), lr=HEAD_LR, betas=(0.9, 0.98),
                                       weight_decay=0.01, foreach=False)
    optimize(model, data, head_optimizer, HEAD_EPOCHS, device, False,
             lambda *args: assess('head', *args))
    model.load_state_dict(torch.load(choices['B'], weights_only=True, map_location='cpu'), strict=True)
    model.gate.requires_grad_(True)
    mix_optimizer = torch.optim.AdamW([{'params': model.tied.parameters(), 'lr': MIX_HEAD_LR},
                                      {'params': model.gate.parameters(), 'lr': GATE_LR},
                                      {'params': [model.output_embedding], 'lr': 0.0}],
                                      betas=(0.9, 0.98), weight_decay=0.01, foreach=False)
    optimize(model, data, mix_optimizer, MIX_EPOCHS, device, True, lambda *args: assess('mix', *args))
    parent_state = torch.load(parent, weights_only=True, map_location='cpu')
    candidates = {}
    for key, path in choices.items():
        model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'), strict=True)
        assert all(torch.equal(model.state_dict()[name].cpu(), value) for name, value in parent_state.items())
        model.tied_only.fill_(key == 'B')
        candidates[key] = out / f'best_{key}.pt'
        save_state(model, candidates[key])

    record['best_metrics'] = best_metrics
    record['promotion'] = {key: accuracy_gate(metrics) for key, metrics in best_metrics.items()}
    if pilot:
        cpu = cpu_audit(parent, candidates, dev, out)
        eligible = [key for key, gate in record['promotion'].items()
                    if gate['passed'] and cpu[key]['practical']]
        winner = min(eligible, key=best.get) if eligible else None
    else:
        # The submitted architecture is the mixture. Select its best finite
        # checkpoint on this run's held-book data, independently of pilot gates.
        winner = 'mix'
        assert best[winner] == min(ce(item['dev'], winner) for item in record['epochs'] if item['stage'] == 'mix')
        cpu = cpu_audit(parent, {winner: candidates[winner]}, dev, out)
    record['cpu'] = cpu
    selected = candidates[winner] if winner else None
    record.update(winner=winner, winner_path=str(selected if pilot else selected.relative_to(HERE)) if selected else None,
                  model_sha256=digest(selected) if selected else None,
                  base_tensors_bitwise_unchanged=True, stage_b_embedding_groups=1, embedding_lr=0.0,
                  seconds=monotonic() - start_time)
    if pilot:
        assert digest(V1 / 'submission/50028181.zip') == ZIP_SHA
    write_json(out / 'history.json', record)
    print(json.dumps(dict(winner=winner, best=record['best'], promotion=record['promotion'], cpu=cpu,
                          seconds=record['seconds']), indent=2), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=V1 / 'model.pt')
    parser.add_argument('--out', type=Path, default=HERE / 'pilot')
    args = parser.parse_args()
    run(args.parent, args.out)
