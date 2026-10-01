"""Read-only inventory and group-disjoint splits; never modifies source images."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
LABELS = {'no_tumor', 'tumor', 'benign', 'malignant'}
METADATA_FIELDS = ['path', 'label', 'patient_id', 'source_image_id', 'is_original', 'label_verified']


def write_csv(path, rows, fieldnames=None):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def image_hash(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def resolve_image(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError('Manifest image paths must be relative to --data-root.')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Image escapes data root: {relative}')
    return path


def inventory(root):
    root = Path(root).resolve(strict=True)
    rows, failures = [], []
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        label = 'malignant' if path.parent.name == 'malignant_extra' else path.parent.name
        if label not in LABELS:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            safe_path = resolve_image(root, relative)
            with Image.open(safe_path) as image:
                width, height = image.size
                image.load()
            rows.append(dict(path=relative, label=label, patient_id='', source_image_id='',
                             is_original='', label_verified='', pixel_sha256=image_hash(safe_path),
                             file_sha256=hashlib.sha256(safe_path.read_bytes()).hexdigest(),
                             width=width, height=height,
                             existing_partition='test' if relative.startswith('test/') else 'development'))
        except (OSError, ValueError) as error:
            failures.append(dict(path=relative, error=str(error)))
    if not rows:
        raise ValueError('No readable images in the expected class folders.')
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row['pixel_sha256']].append(row)
    duplicates = [[r['path'] for r in group] for group in by_hash.values() if len(group) > 1]
    conflicts, cross_partition = [], []
    for group in by_hash.values():
        labels = {r['label'] for r in group}
        if ('no_tumor' in labels and len(labels) > 1) or {'benign', 'malignant'} <= labels:
            conflicts.append([{'path': r['path'], 'label': r['label']} for r in group])
        if len({r['existing_partition'] for r in group}) > 1:
            cross_partition.append([r['path'] for r in group])
    report = dict(status='METADATA_REQUIRED', images=len(rows), unique_pixels=len(by_hash),
                  class_counts=dict(Counter(r['label'] for r in rows)), unreadable=failures,
                  duplicate_groups=duplicates, conflicting_label_groups=conflicts,
                  existing_test_overlap=cross_partition,
                  missing_metadata=METADATA_FIELDS[2:],
                  note='Pixel hashes detect exact decoded duplicates, not patients or augmented variants.')
    return rows, report


def merge_metadata(rows, metadata, group_by='patient_id', allow_unverified_labels=False):
    by_path = {}
    for row in metadata:
        if row['path'] in by_path:
            raise ValueError(f'Duplicate metadata path: {row["path"]}')
        by_path[row['path']] = row
    result, missing = [], []
    for source in rows:
        raw = by_path.get(source['path'], {})
        entry = {key: (raw.get(key) or '').strip() for key in METADATA_FIELDS}
        if (not entry.get(group_by, '').strip() or not entry.get('source_image_id', '').strip()
                or entry.get('is_original', '').lower() not in {'true', 'false'}
                or entry.get('label_verified', '').lower() not in
                   ({'true', 'false'} if allow_unverified_labels else {'true'})
                or entry.get('label') not in LABELS):
            missing.append(source['path'])
            continue
        row = {**source, **{key: entry.get(key, '').strip() for key in METADATA_FIELDS}}
        result.append(row)
    if missing:
        raise ValueError(f'Metadata incomplete/unverified for {len(missing)} images; first: {missing[:3]}')
    extras = set(by_path) - {r['path'] for r in rows}
    if extras:
        raise ValueError(f'Metadata contains paths outside the inventory: {sorted(extras)[:3]}')
    return result


def verify_label_consistency(rows):
    labels_by_source = defaultdict(set)
    for row in rows:
        for field in ('source_image_id', 'pixel_sha256'):
            if not row.get(field):
                continue
            labels_by_source[(field, row[field])].add(row['label'])
    for (field, identifier), labels in labels_by_source.items():
        if ('no_tumor' in labels and len(labels) > 1) or {'benign', 'malignant'} <= labels:
            raise ValueError(f'Conflicting labels for {field}: {identifier}')


def assign_groups(rows, group_by):
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for field in {group_by, 'patient_id', 'source_image_id', 'pixel_sha256'}:
            if not row.get(field):
                continue
            key = (field, row[field])
            if key in seen:
                parent[find(index)] = find(seen[key])
            seen[key] = index
    members = defaultdict(list)
    for index, row in enumerate(rows):
        members[find(index)].append(row['path'])
    group_ids = {key: hashlib.sha256('\n'.join(sorted(paths)).encode()).hexdigest()[:20]
                 for key, paths in members.items()}
    return [{**row, 'group_id': group_ids[find(index)]} for index, row in enumerate(rows)]


def verify_manifest(rows, require_patient=True, allow_unverified_labels=False, allow_unknown_ancestry=False):
    if not rows:
        raise ValueError('Manifest is empty.')
    seen = defaultdict(set)
    known = {'train', 'val', 'test'}
    paths = set()
    for row in rows:
        if row['path'] in paths:
            raise ValueError(f'Duplicate manifest path: {row["path"]}')
        paths.add(row['path'])
        if row.get('split') not in known or row.get('label') not in LABELS:
            raise ValueError('Invalid split or label.')
        if row.get('label_verified', '').lower() not in ({'true', 'false'} if allow_unverified_labels else {'true'}):
            raise ValueError('Label provenance is unverified; exploratory use requires --allow-unverified-labels and explicit true/false status.')
        original_status = row.get('is_original', '').lower()
        if original_status not in ({'true', 'false', 'unknown'} if allow_unknown_ancestry else {'true', 'false'}):
            raise ValueError('Original/augmentation status must be explicit.')
        if row['split'] != 'train' and (original_status == 'false' or (original_status != 'true' and not allow_unknown_ancestry)):
            raise ValueError('Validation and test must contain original images only.')
        fields = ['group_id', 'pixel_sha256']
        if not allow_unknown_ancestry or row.get('source_image_id'):
            fields.append('source_image_id')
        if require_patient:
            fields.append('patient_id')
        for field in fields:
            if not row.get(field):
                raise ValueError(f'Missing {field}: {row["path"]}')
            seen[(field, row[field])].add(row['split'])
        if row.get('patient_id'):
            seen[('patient_id', row['patient_id'])].add(row['split'])
    if any(len(splits) != 1 for splits in seen.values()):
        raise ValueError('Patient/source image/duplicate pixels cross split boundaries.')
    verify_label_consistency(rows)
    for split in known:
        labels = {r['label'] for r in rows if r['split'] == split}
        if not {'no_tumor', 'benign', 'malignant'} <= labels:
            raise ValueError(f'{split} must cover no_tumor, benign and malignant.')


def make_split(rows, seed, group_by='patient_id', preserve_test=True, allow_unverified_labels=False,
               allow_unknown_ancestry=False):
    from sklearn.model_selection import GroupShuffleSplit

    verify_label_consistency(rows)
    rows = assign_groups(rows, group_by)
    held_out = [r for r in rows if preserve_test and r['existing_partition'] == 'test']
    pool = [r for r in rows if not preserve_test or r['existing_partition'] != 'test']
    if preserve_test:
        held_groups = {r['group_id'] for r in held_out}
        if any(r['group_id'] in held_groups for r in pool):
            raise ValueError('Existing test overlaps development by patient/source/pixels; resolve provenance first.')
        if not held_out:
            raise ValueError('No existing test images; use --resplit-all explicitly for an exploratory split.')
    last_error = None
    for offset in range(100):
        try:
            groups = [r['group_id'] for r in pool]
            first = GroupShuffleSplit(n_splits=1, test_size=0.2 if held_out else 0.3,
                                      random_state=seed + offset)
            tr, rest = next(first.split(pool, groups=groups))
            if held_out:
                assignments = {int(i): 'train' for i in tr} | {int(i): 'val' for i in rest}
            else:
                rest_rows = [pool[i] for i in rest]
                second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + offset)
                va, te = next(second.split(rest_rows, groups=[r['group_id'] for r in rest_rows]))
                assignments = ({int(i): 'train' for i in tr}
                               | {int(rest[i]): 'val' for i in va}
                               | {int(rest[i]): 'test' for i in te})
            result = [{**r, 'split': assignments[i]} for i, r in enumerate(pool)
                      if assignments[i] == 'train' or r['is_original'].lower() == 'true'
                      or (allow_unknown_ancestry and r['is_original'].lower() == 'unknown')]
            result += [{**r, 'split': 'test'} for r in held_out]
            verify_manifest(result, require_patient=group_by == 'patient_id',
                            allow_unverified_labels=allow_unverified_labels, allow_unknown_ancestry=allow_unknown_ancestry)
            return result, seed + offset
        except ValueError as error:
            last_error = error
    raise ValueError(f'Could not create valid grouped splits: {last_error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New directory; existing outputs are never overwritten.')
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--group-by', choices=['patient_id', 'source_image_id'], default='patient_id')
    parser.add_argument('--allow-unverified-labels', action='store_true',
                        help='Exploratory use of provided labels; does not verify their medical correctness.')
    parser.add_argument('--exploratory-files', action='store_true',
                        help='Use current files with unknown patient/augmentation ancestry and provided folder labels.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resplit-all', action='store_true', help='Explicit exploratory split of previously used data.')
    args = parser.parse_args()
    if args.exploratory_files and args.metadata:
        parser.error('--exploratory-files is the explicit fallback without metadata; use --metadata when records exist.')
    rows, report = inventory(args.data_root)
    prepared = None
    if args.metadata or args.exploratory_files:
        if report['unreadable']:
            raise ValueError('Unreadable input images must be resolved before preparing a split; see the inventory audit.')
        if args.exploratory_files:
            args.group_by = 'pixel_sha256'
            rows = [{**r, 'is_original': 'unknown', 'label_verified': 'false'} for r in rows]
        else:
            rows = merge_metadata(rows, read_csv(args.metadata), args.group_by, args.allow_unverified_labels)
        prepared, split_seed = make_split(rows, args.seed, args.group_by, not args.resplit_all,
                                         args.allow_unverified_labels or args.exploratory_files,
                                         allow_unknown_ancestry=args.exploratory_files)
        unverified = sum(r['label_verified'].lower() != 'true' for r in rows)
        report.update(status='PREPARED_EXPLORATORY' if unverified or args.group_by != 'patient_id' else 'PREPARED',
                      label_provenance='UNVERIFIED' if unverified else 'CHECKED_SOURCE',
                      unverified_label_rows=unverified, group_by=args.group_by, split_seed=split_seed,
                      patient_disjoint=args.group_by == 'patient_id',
                      ancestry_status='UNKNOWN' if args.exploratory_files else 'KNOWN_IMAGE_GROUPS',
                      independent_originals_confirmed=not args.exploratory_files,
                      exploratory_resplit=args.resplit_all,
                      split_counts=dict(Counter(r['split'] for r in prepared)))
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / 'inventory.csv', rows)
    write_csv(args.out / 'metadata_template.csv', [{key: row[key] for key in METADATA_FIELDS} for row in rows])
    if prepared is not None:
        write_csv(args.out / 'manifest.csv', prepared)
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status', 'images', 'unique_pixels', 'class_counts']}, indent=2))


if __name__ == '__main__':
    main()
