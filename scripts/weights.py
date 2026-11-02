#!/usr/bin/env python3
"""Inspect, compare and project one floating-point safetensors file. No model execution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FLOATS = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
CHUNK = 1_000_000


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def chunks(tensor):
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), CHUNK):
        yield flat[start:start + CHUNK]


def health(tensor):
    bad = 0
    square_sum = 0.0
    abs_max = 0.0
    for part in chunks(tensor):
        x = part.double()
        finite = torch.isfinite(x)
        bad += int((~finite).sum())
        good = x[finite]
        if good.numel():
            square_sum += float(torch.sum(good * good))
            abs_max = max(abs_max, float(good.abs().max()))
    return {'nonfinite': bad, 'finite_l2': math.sqrt(square_sum), 'finite_abs_max': abs_max}


def inspect_file(path, match=''):
    records = []
    total = 0
    with safe_open(path, framework='pt', device='cpu') as f:
        for name in sorted(f.keys()):
            view = f.get_slice(name)
            shape = list(view.get_shape())
            n = math.prod(shape)
            total += n
            if match in name:
                t = f.get_tensor(name)
                records.append({'name': name, 'shape': shape, 'dtype': str(t.dtype),
                                'numel': n, **health(t)})
        metadata = f.metadata()
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path),
            'total_stored_elements': total, 'metadata': metadata, 'tensors': records}


def tensor_difference(base, candidate):
    if base.shape != candidate.shape or base.dtype != candidate.dtype:
        raise ValueError('shape and dtype must match')
    if base.dtype not in FLOATS:
        raise ValueError('comparison supports F16, BF16, F32 and F64 tensors only')
    changed = 0
    delta_sq = 0.0
    base_sq = 0.0
    max_delta = 0.0
    for a, b in zip(chunks(base), chunks(candidate)):
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError('comparison requires finite tensors')
        changed += int((a != b).sum())
        x, y = a.double(), b.double()
        delta = y - x
        delta_sq += float(torch.sum(delta * delta))
        base_sq += float(torch.sum(x * x))
        max_delta = max(max_delta, float(delta.abs().max()))
    norm = math.sqrt(delta_sq)
    return {'changed_elements': changed, 'total_elements': base.numel(),
            'delta_l2': norm, 'max_abs_delta': max_delta,
            'relative_delta_l2': norm / math.sqrt(base_sq) if base_sq else None}


def compare_files(base_path, candidate_path):
    records = []
    total = changed = 0
    with safe_open(base_path, framework='pt', device='cpu') as a, \
            safe_open(candidate_path, framework='pt', device='cpu') as b:
        if set(a.keys()) != set(b.keys()):
            raise ValueError('checkpoint tensor names must match exactly')
        for name in sorted(a.keys()):
            stats = tensor_difference(a.get_tensor(name), b.get_tensor(name))
            total += stats['total_elements']
            changed += stats['changed_elements']
            if stats['changed_elements']:
                records.append({'name': name, **stats})
    return {'base_sha256': sha256(base_path), 'candidate_sha256': sha256(candidate_path),
            'changed_elements': changed, 'total_stored_elements': total,
            'changed_percent_of_file': 100.0 * changed / total if total else 0.0,
            'changed_tensors': records}


def project_tensor(weight, direction, columns, alpha):
    if weight.dtype not in FLOATS or weight.ndim != 2:
        raise ValueError('projection requires a 2D F16, BF16, F32 or F64 matrix')
    if not bool(torch.isfinite(weight).all()):
        raise ValueError('source matrix contains nonfinite values')
    d = torch.tensor(direction, dtype=torch.float64)
    if d.ndim != 1 or d.numel() != weight.shape[0]:
        raise ValueError('direction length must equal the matrix output dimension (axis 0)')
    if not bool(torch.isfinite(d).all()):
        raise ValueError('direction must be finite')
    # Scale first so normalizing very large/small finite vectors is stable.
    largest = float(d.abs().max()) if d.numel() else 0.0
    if largest == 0:
        raise ValueError('direction must be nonzero')
    u = d / largest
    u /= torch.linalg.vector_norm(u)
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError('alpha must be finite and in (0, 1]')
    if not columns or any(type(j) is not int for j in columns):
        raise ValueError('supply an explicit nonempty list of integer columns')
    if len(set(columns)) != len(columns) or min(columns) < 0 or max(columns) >= weight.shape[1]:
        raise ValueError('columns must be unique and within the matrix input dimension')
    with torch.no_grad():
        selected = weight[:, columns].double()
        new = selected - alpha * u[:, None] * (u @ selected)[None, :]
        edited = weight.clone()
        edited[:, columns] = new.to(weight.dtype)
    if not bool(torch.isfinite(edited).all()):
        raise ValueError('projection overflows the stored dtype')
    return edited


def project_file(source, name, direction_path, columns, alpha=1.0, output=None):
    source = Path(source).resolve()
    output = Path(output).absolute() if output is not None else None
    sidecar = Path(str(output) + '.manifest.json') if output else None
    if output is not None:
        if output.resolve() == source or output.exists() or output.is_symlink():
            raise ValueError('output must be a new path distinct from the source')
        if sidecar.exists() or sidecar.is_symlink():
            raise ValueError('output manifest already exists')
        if not output.parent.is_dir():
            raise ValueError('output parent directory must already exist')
    base_hash = sha256(source)
    direction_bytes = Path(direction_path).read_bytes()
    direction = json.loads(direction_bytes)
    with safe_open(source, framework='pt', device='cpu') as f:
        if name not in f.keys():
            raise ValueError(f'tensor not found: {name}')
        weight = f.get_tensor(name)
        edited = project_tensor(weight, direction, columns, alpha)
        stats = tensor_difference(weight, edited)
        total = sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
        manifest = {
            'operation': 'selected_column_direction_removal', 'source': str(source),
            'source_sha256': base_hash, 'tensor': name, 'shape': list(weight.shape),
            'dtype': str(weight.dtype), 'columns': columns, 'alpha': alpha,
            'direction_sha256': hashlib.sha256(direction_bytes).hexdigest(),
            'direction': direction, 'requested_elements': weight.shape[0] * len(columns),
            **stats, 'total_stored_elements': total,
            'changed_percent_of_file': 100.0 * stats['changed_elements'] / total if total else 0.0,
            'output': str(output) if output else None, 'written': False,
        }
        if output is None:
            return manifest
        # Saving a whole shard requires RAM for its tensors plus serialization work.
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        tensors[name] = edited
        metadata = f.metadata()
        with tempfile.TemporaryDirectory(prefix='.wsp-', dir=output.parent) as tmp:
            staged = Path(tmp) / 'candidate.safetensors'
            save_file(tensors, str(staged), metadata=metadata)
            if sha256(source) != base_hash:
                raise ValueError('source changed during editing; candidate discarded')
            manifest.update(written=True, output_sha256=sha256(staged))
            staged_manifest = Path(tmp) / 'manifest.json'
            staged_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
            # Hard links publish complete files atomically without overwriting a path.
            os.link(staged, output)
            try:
                os.link(staged_manifest, sidecar)
            except BaseException:
                output.unlink()
                raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inspect', help='inventory one safetensors file')
    p.add_argument('checkpoint')
    p.add_argument('--match', default='', help='tensor-name substring; totals still cover the full file')
    p = sub.add_parser('diff', help='compare floating-point checkpoints with identical schemas')
    p.add_argument('base')
    p.add_argument('candidate')
    p = sub.add_parser('project', help='preview an explicit projection; --output writes a fresh file')
    p.add_argument('checkpoint')
    p.add_argument('--tensor', required=True)
    p.add_argument('--direction', required=True, help='JSON array in residual/output coordinates')
    p.add_argument('--columns', required=True, help='comma-separated input column indices')
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--output', help='fresh output safetensors path; omitted means no writes')
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            result = inspect_file(args.checkpoint, args.match)
        elif args.command == 'diff':
            result = compare_files(args.base, args.candidate)
        else:
            columns = [int(x.strip()) for x in args.columns.split(',')]
            result = project_file(args.checkpoint, args.tensor, args.direction,
                                  columns, args.alpha, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, TypeError, OverflowError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
