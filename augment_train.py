"""Reproduce an augmentation recipe on known train originals, with an audit trail."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import PIL
import yaml
from PIL import Image

from prepare_data import LABELS, image_hash, read_csv, resolve_image, verify_manifest, write_csv


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    params = config['augmentation']
    if config['version'] != 1 or not isinstance(config['seed'], int) or config['seed'] < 0:
        raise ValueError('Expected version 1 and a nonnegative integer seed.')
    if not isinstance(config['copies_per_original'], int) or config['copies_per_original'] < 1:
        raise ValueError('copies_per_original must be a positive integer.')
    if (not all(np.isfinite(value) for value in params.values())
            or not 0 <= params['rotation_degrees'] <= 180
            or not 0 <= params['horizontal_flip_probability'] <= 1
            or not 0 < params['brightness_min'] <= params['brightness_max']):
        raise ValueError('Invalid augmentation parameters.')
    return config


def variant_seed(base_seed, source_image_id, variant):
    payload = json.dumps([base_seed, source_image_id, variant], ensure_ascii=False).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def augment_image(image, params, seed):
    rng = np.random.RandomState(seed)
    angle = float(rng.uniform(-params['rotation_degrees'], params['rotation_degrees']))
    flipped = bool(rng.random_sample() > 1 - params['horizontal_flip_probability'])
    brightness = float(rng.uniform(params['brightness_min'], params['brightness_max']))
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    result = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    if flipped:
        result = cv2.flip(result, 1)
    result = np.clip(result.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return result, dict(seed=seed, angle_degrees=angle, horizontal_flip=flipped, brightness_factor=brightness)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest', type=Path, help='Prepared manifest with known original training images.')
    inputs.add_argument('--input-path', help='One current training file, relative to data root; old ancestry stays unknown.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New directory outside source data.')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('augmentation.yaml'))
    parser.add_argument('--seed', type=int, help='Optional new base seed, overriding the configuration.')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        if args.seed < 0:
            parser.error('Seed must be nonnegative.')
        config['seed'] = args.seed
    root = args.data_root.resolve(strict=True)
    if args.out.resolve().is_relative_to(root):
        parser.error('Output must be outside the source data tree.')
    if args.input_path:
        path = resolve_image(root, args.input_path)
        relative = path.relative_to(root)
        if relative.parts[0] not in {'train', 'train_presence'} or path.parent.name not in LABELS:
            parser.error('A current-file preview must come from a recognized training class folder.')
        originals = [dict(path=relative.as_posix(), label=path.parent.name, patient_id='', source_image_id='',
                          is_original='', label_verified='false', pixel_sha256=image_hash(path), split='train', group_id='')]
    else:
        rows = read_csv(args.manifest)
        # Generation inherits supplied labels; it makes no label-verification claim.
        verify_manifest(rows, require_patient=False, allow_unverified_labels=True)
        originals = sorted([r for r in rows if r['split'] == 'train' and r['is_original'].lower() == 'true'],
                           key=lambda row: row['path'])
    if not originals:
        parser.error('No known original training images. Unknown ancestry cannot be reconstructed from a seed.')
    for row in originals:
        if image_hash(resolve_image(root, row['path'])) != row['pixel_sha256']:
            raise ValueError(f'Input image changed: {row["path"]}')
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for index, row in enumerate(originals):
        source_path = resolve_image(root, row['path'])
        with Image.open(source_path) as image:
            source = np.array(image.convert('RGB'))
        source_file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        for variant in range(config['copies_per_original']):
            # A pixel hash identifies the current input; it is never a patient/original-image ID.
            seed = variant_seed(config['seed'], row['source_image_id'] or row['pixel_sha256'], variant)
            result, parameters = augment_image(source, config['augmentation'], seed)
            relative = f'train/{row["label"]}/aug_{index:05d}_{variant:03d}.png'
            path = args.out / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result).save(path)
            records.append(dict(path=relative, label=row['label'], patient_id=row['patient_id'],
                                source_image_id=row['source_image_id'], is_original='false',
                                label_verified=row['label_verified'], split='train', group_id=row['group_id'],
                                source_path=row['path'], source_is_original=row['is_original'] or 'unknown',
                                source_file_sha256=source_file_hash,
                                source_pixel_sha256=row['pixel_sha256'], pixel_sha256=image_hash(path),
                                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **parameters))
    write_csv(args.out / 'metadata.csv', records)
    audit = dict(config=config, source_data_root=str(root),
                 manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None,
                 source_mode='CURRENT_FILE_UNKNOWN_ANCESTRY' if args.input_path else 'KNOWN_TRAIN_ORIGINALS',
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 numpy=np.__version__, opencv=cv2.__version__, pillow=PIL.__version__,
                 rng='MT19937; per-image SHA256-derived seed', generated_images=len(records),
                 note='New augmented training variants. This does not restore deleted images or historical RNG state.')
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'Generated {len(records)} train variants with source mappings: {args.out}')


if __name__ == '__main__':
    main()
