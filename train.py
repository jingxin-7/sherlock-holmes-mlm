"""Train one model from the verified encoder and static CSV files."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from make_data import HERE, HEADER, digest, tokenizer
from v1_model import CLUSTER_SIZES, HEAD_WIDTH, MASK_ID, StudentModel, VOCAB_SIZE

SEED = 3102
BATCH = 80
MAX_EPOCHS = 12
PATIENCE = 3
MIN_DELTA = 0.004
WARMUP = 300
ENCODER_FROZEN_STEPS = 240
REFERENCE_SHA256 = 'e81dca4aab522f192ea2f8847dda0fb8aedb748581a8b77d426cae0bdcfb3964'


def setup():
    torch.set_num_threads(4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_csv(path, tok):
    examples = []
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == HEADER
        records = list(reader)
    for start in range(0, len(records), 4096):
        block = records[start:start + 4096]
        encoded = tok.encode_batch([row['masked_text'] for row in block])
        for ids, row in zip(encoded, block):
            sequence = np.asarray(ids.ids, dtype=np.int32)
            positions = np.flatnonzero(sequence == MASK_ID).astype(np.int32)
            labels = [row[field] for field in HEADER[1:] if row[field]]
            label_ids = [tok.token_to_id(label) for label in labels]
            assert 1 <= len(positions) == len(label_ids) <= 3
            assert len(sequence) <= 128 and sequence[0] == 101 and sequence[-1] == 102
            assert all(i is not None and i not in (0, 100, 101, 102, MASK_ID) for i in label_ids)
            assert [tok.id_to_token(i) for i in label_ids] == labels
            examples.append((sequence, positions, np.array(label_ids, dtype=np.int32)))
    return examples


def initialize_encoder():
    model = StudentModel()
    reference = HERE / 'resources/encoder_reference.pt'
    assert digest(reference) == REFERENCE_SHA256
    state = torch.load(reference, map_location='cpu', weights_only=True)
    inventory = json.loads((HERE / 'resources/provenance.json').read_text())
    expected = {f'bert.{key}' for key in model.bert.state_dict()}
    assert set(state) == expected == set(inventory['allowed_keys']) and len(expected) == 37
    model.bert.load_state_dict({key.removeprefix('bert.'): value for key, value in state.items()}, strict=True)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())
    pointers = [p.untyped_storage().data_ptr() for p in model.parameters()]
    assert len(pointers) == len(set(pointers))
    assert sum(p.numel() for p in model.parameters()) <= 9_000_000
    return model


def batches(examples, rng, batch_size=BATCH, limit=None):
    order = np.arange(len(examples))
    if rng is not None:
        rng.shuffle(order)
    if limit is not None:
        order = order[:limit]
    packs = []
    for start in range(0, len(order), batch_size * 16):
        pool = sorted(order[start:start + batch_size * 16], key=lambda idx: len(examples[idx][0]))
        packs.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))
    if rng is not None:
        rng.shuffle(packs)
    for pack in packs:
        yield [examples[i] for i in pack]


def collate(examples, device):
    longest = max(len(x[0]) for x in examples)
    width = min(128, (longest + 7) // 8 * 8)
    ids = np.zeros((len(examples), width), dtype=np.int64)
    for row, (seq, _, _) in enumerate(examples):
        ids[row, :len(seq)] = seq
    rows = np.repeat(np.arange(len(examples)), [len(x[1]) for x in examples])
    positions = np.concatenate([x[1] for x in examples])
    targets = np.concatenate([x[2] for x in examples])
    inputs = tuple(torch.as_tensor(x, device=device, dtype=torch.long) for x in (ids, ids != 0, rows, positions))
    return inputs, targets


def target_groups(targets, vocab_to_rank, device):
    ranks = vocab_to_rank[targets]
    root = ranks.copy()
    groups = []
    start = CLUSTER_SIZES[0]
    for band, size in enumerate(CLUSTER_SIZES[1:]):
        rows = np.flatnonzero((ranks >= start) & (ranks < start + size))
        groups.append(tuple(torch.as_tensor(x, device=device, dtype=torch.long) for x in (rows, ranks[rows] - start)))
        root[rows] = CLUSTER_SIZES[0] + band
        start += size
    return torch.as_tensor(root, device=device, dtype=torch.long), groups


@torch.no_grad()
def initialize_decoder(model, train, counts, device):
    order = np.argsort(-counts, kind='stable')
    inverse = np.argsort(order)
    decoder = model.decoder
    decoder.vocab_to_rank.copy_(torch.from_numpy(inverse))
    unseen = counts == 0
    backoff = float(np.clip(np.count_nonzero(counts == 1) / counts.sum(), 0.005, 0.025))
    decoder.backoff_log_probs[torch.from_numpy(unseen).to(device)] = math.log(backoff / unseen.sum())
    decoder.log_main_mass.fill_(math.log1p(-backoff))
    smoothed = counts[order] + 0.07
    common, middle, _ = CLUSTER_SIZES
    partitions = np.split(smoothed, [common, common + middle])
    root_counts = np.r_[partitions[0], partitions[1].sum(), partitions[2].sum()]
    decoder.root.bias.copy_(torch.from_numpy(np.log(root_counts)))
    for classifier, frequencies in zip(decoder.tails, partitions[1:]):
        classifier.bias.copy_(torch.from_numpy(np.log(frequencies)))

    model.eval()
    sums = torch.zeros((VOCAB_SIZE, HEAD_WIDTH))
    observations = torch.zeros(VOCAB_SIZE)
    for examples in batches(train, np.random.default_rng(SEED + 1), batch_size=128, limit=32768):
        inputs, targets = collate(examples, device)
        hidden = model.masked_features(*inputs).cpu()
        labels = torch.from_numpy(targets).long()
        sums.index_add_(0, labels, hidden)
        observations.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    average = sums.sum(0) / observations.sum()
    centered = sums / observations[:, None].clamp_min(1) - average
    reliability = observations / (observations + 6)
    start = 0
    for band, (classifier, size) in enumerate(zip([decoder.root, *decoder.tails], CLUSTER_SIZES)):
        ids = order[start:start + size]
        prototype = decoder.features(centered[ids].to(device), band)
        weights = F.normalize(prototype, dim=-1) * (0.65 * reliability[ids, None].to(device))
        classifier.weight[:size].copy_(weights)
        start += size
    decoder.root.weight[common:].zero_()
    print(f'Initialized decoder from {int(observations.sum()):,} static training targets. Unseen mass {backoff:.5f}.',
          flush=True)
    return inverse, backoff


@torch.inference_mode()
def score(model, dev, counts, device):
    model.eval()
    groups = {'all': [0., 0], 'frequent': [0., 0], 'rare': [0., 0], 'unseen': [0., 0]}
    for examples in batches(dev, None, 48):
        inputs, targets = collate(examples, device)
        loss = F.cross_entropy(model.decoder(model.masked_features(*inputs)).float(),
                               torch.as_tensor(targets, device=device, dtype=torch.long), reduction='none')
        values = loss.cpu().numpy()
        frequency = counts[targets]
        for name, losses in (
            ('all', values), ('frequent', values[frequency >= 10]),
            ('rare', values[(frequency > 0) & (frequency < 10)]), ('unseen', values[frequency == 0]),
        ):
            groups[name][0] += float(losses.sum(dtype=np.float64))
            groups[name][1] += len(losses)
    return {key: {'ce': total / n if n else None, 'masks': n} for key, (total, n) in groups.items()}


def optimizer_for(model):
    return torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 0.00016},
        {'params': model.gap.parameters(), 'lr': 0.0012},
        {'params': model.decoder.parameters(), 'lr': 0.003},
    ], betas=(0.9, 0.98), weight_decay=0.025, foreach=False)


def optimize_step(model, optimizer, examples, inverse, device):
    inputs, targets = collate(examples, device)
    root, tails = target_groups(targets, inverse, device)
    optimizer.zero_grad(set_to_none=True)
    hidden = model.masked_features(*inputs)
    loss = model.decoder.loss(hidden, root, tails)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=False, foreach=False)
    optimizer.step()
    return loss.detach(), norm.detach(), len(targets)


def save_state(model, path):
    temporary = path.with_suffix('.tmp')
    torch.save({key: val.detach().cpu() for key, val in model.state_dict().items()}, temporary)
    os.replace(temporary, path)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def run(out_dir=HERE):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = setup()
    start_time = monotonic()
    tok = tokenizer()
    train = load_csv(HERE / 'train.csv', tok)
    dev = load_csv(HERE / 'dev.csv', tok)
    counts = np.bincount(np.concatenate([x[2] for x in train]), minlength=VOCAB_SIZE)
    model = initialize_encoder().to(device)
    inverse, backoff = initialize_decoder(model, train, counts, device)
    optimizer = optimizer_for(model)
    base_rates = [group['lr'] for group in optimizer.param_groups]
    out = out_dir / 'development_checkpoints'
    out.mkdir(exist_ok=True)
    prior = (counts + 0.07) / (counts + 0.07).sum() * (1 - backoff)
    prior[counts == 0] += backoff / (counts == 0).sum()
    prior_ce = float(-np.log(prior[np.concatenate([x[2] for x in dev])]).mean())
    history = dict(device=str(device), parameter_count=sum(p.numel() for p in model.parameters()),
                   train_sha256=digest(HERE / 'train.csv'), dev_sha256=digest(HERE / 'dev.csv'),
                   encoder_sha256=REFERENCE_SHA256, train_rows=len(train), dev_rows=len(dev),
                   seed=SEED, max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA,
                   unigram_dev_ce=prior_ce, initial_dev=score(model, dev, counts, device), epochs=[])
    print(f'Starting on {device}. Frequency prior CE {prior_ce:.4f}, initialized CE {history["initial_dev"]["all"]["ce"]:.4f}.',
          flush=True)
    steps_per_epoch = math.ceil(len(train) / BATCH)
    best = significant_best = math.inf
    stale = step = 0
    model.bert.requires_grad_(False)
    rng = np.random.default_rng(SEED + 2)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        if step < ENCODER_FROZEN_STEPS:
            model.bert.eval()
        timer = monotonic()
        loss_sum = torch.zeros((), device=device)
        masks = 0
        for examples in batches(train, rng):
            step += 1
            if step == ENCODER_FROZEN_STEPS + 1:
                model.bert.requires_grad_(True).train()
            decay_fraction = max(0, (step - WARMUP) / (MAX_EPOCHS * steps_per_epoch - WARMUP))
            decay = 0.18 + 0.82 * (1 + math.cos(math.pi * decay_fraction)) / 2
            warmup = min(1, (step + 35) / WARMUP)
            encoder_warmup = min(1, max(0, step - ENCODER_FROZEN_STEPS) / WARMUP)
            for group, base_rate, scale in zip(optimizer.param_groups, base_rates,
                                               (encoder_warmup, warmup, warmup)):
                group['lr'] = base_rate * scale * decay
            loss, norm, mask_count = optimize_step(model, optimizer, examples, inverse, device)
            loss_sum += loss * mask_count
            masks += mask_count
            if step % 500 == 0:
                assert math.isfinite(float(norm))
                print(f'Step {step:,}, epoch {epoch}, train CE {float(loss_sum) / masks:.4f}, '
                      f'elapsed {monotonic() - start_time:.1f}s.', flush=True)
        dev_scores = score(model, dev, counts, device)
        value = dev_scores['all']['ce']
        assert math.isfinite(value)
        history['epochs'].append(dict(epoch=epoch, steps=step, train_ce=float(loss_sum) / masks,
                                      dev=dev_scores, seconds=monotonic() - timer))
        if value < best:
            best = value
            path = out / f'epoch_{epoch:02}.pt'
            save_state(model, path)
            history.update(selected_epoch=epoch, selected_ce=best, selected_path=path.name)
        if value < significant_best - MIN_DELTA:
            significant_best, stale = value, 0
        else:
            stale += 1
        write_json(out_dir / 'training_history.json', history)
        print(json.dumps(history['epochs'][-1]), flush=True)
        if stale >= PATIENCE:
            break
    selected = out / history['selected_path']
    # Copy the development winner atomically. The notebook never reads a prior model.pt.
    temporary = out_dir / 'model.tmp'
    temporary.write_bytes(selected.read_bytes())
    os.replace(temporary, out_dir / 'model.pt')
    history.update(seconds=monotonic() - start_time, model_sha256=digest(out_dir / 'model.pt'),
                   stopped_early=stale >= PATIENCE)
    write_json(out_dir / 'training_history.json', history)
    print(f'Saved model.pt from epoch {history["selected_epoch"]}, held-book CE {best:.6f}.', flush=True)
    return history
