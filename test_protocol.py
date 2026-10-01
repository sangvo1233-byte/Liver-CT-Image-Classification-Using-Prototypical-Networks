"""Runnable regression checks for leakage, budgets, scores and pipeline errors."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from augment_train import augment_image, load_config, variant_seed
from liver_ct import (binary_metrics, build_model, choose_threshold, episode_indices,
                      load_artifact, patient_bootstrap, pipeline_labels, predict_images,
                      preprocessing, prototype_logits, select_budget, set_seed, task_rows, validate_pair)
from prepare_data import (assign_groups, image_hash, make_split, merge_metadata,
                          resolve_image, verify_manifest, write_csv)


def expect_error(function, message):
    try:
        function()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError(message)


def fixture(root):
    rows = []
    for label_index, label in enumerate(('no_tumor', 'benign', 'malignant')):
        for index in range(12):
            path = root / label / f'{index}.png'
            path.parent.mkdir(exist_ok=True)
            rng = np.random.default_rng(index + 100 * label_index)
            Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(path)
            rows.append(dict(path=path.relative_to(root).as_posix(), label=label,
                             patient_id=f'{label}_{index}', source_image_id=f'{label}_{index}_scan',
                             is_original='true', label_verified='true', pixel_sha256=image_hash(path),
                             existing_partition='test' if index >= 9 else 'development'))
    return rows


def check(root):
    config = load_config(Path(__file__).with_name('augmentation.yaml'))
    sample = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    unchanged = sample.copy()
    seed = variant_seed(42, 'synthetic_original', 0)
    first, parameters = augment_image(sample, config['augmentation'], seed)
    np.random.seed(1234)
    repeated, repeated_parameters = augment_image(sample, config['augmentation'], seed)
    assert np.array_equal(first, repeated) and parameters == repeated_parameters
    assert np.array_equal(sample, unchanged)
    _, other_parameters = augment_image(sample, config['augmentation'], variant_seed(43, 'synthetic_original', 0))
    assert parameters != other_parameters
    rows = fixture(root)
    split, _ = make_split(rows, 42)
    verify_manifest(split)
    assert all(row['split'] == 'test' for row in split if row['existing_partition'] == 'test')
    leak = [dict(row) for row in split]
    first_train = next(row for row in leak if row['split'] == 'train')
    first_test = next(row for row in leak if row['split'] == 'test')
    first_test['patient_id'] = first_train['patient_id']
    expect_error(lambda: verify_manifest(leak), 'Patient leakage must be rejected.')
    leak = [dict(row) for row in split]
    next(row for row in leak if row['split'] == 'test')['is_original'] = 'false'
    expect_error(lambda: verify_manifest(leak), 'Augmented test data must be rejected.')
    expect_error(lambda: merge_metadata(rows, []), 'Missing metadata must not be invented.')
    expect_error(lambda: resolve_image(root, '../outside.png'), 'Paths must stay in the data root.')
    linked = assign_groups([rows[0], {**rows[1], 'source_image_id': rows[0]['source_image_id']}], 'patient_id')
    assert linked[0]['group_id'] == linked[1]['group_id']
    linked = assign_groups([rows[0], {**rows[1], 'patient_id': rows[0]['patient_id']}], 'source_image_id')
    assert linked[0]['group_id'] == linked[1]['group_id']  # Known patients still bind fallback groups.
    contradictory = [dict(row) for row in rows]
    contradictory[12]['source_image_id'] = contradictory[0]['source_image_id']
    expect_error(lambda: make_split(contradictory, 42), 'One original image cannot have conflicting labels.')
    originals = [dict(row) for row in rows]
    duplicate_in_pool = {**rows[0], 'path': 'copy.png', 'pixel_sha256': rows[0]['pixel_sha256']}
    linked = assign_groups(originals + [duplicate_in_pool], 'patient_id')
    assert linked[0]['group_id'] == linked[-1]['group_id']
    for row in originals:
        row['patient_id'] = ''
    fallback, _ = make_split(originals, 42, group_by='source_image_id')
    verify_manifest(fallback, require_patient=False)
    expect_error(lambda: verify_manifest(fallback), 'Patient-less splits must require an explicit weaker mode.')
    weak = [{**row, 'patient_id': '', 'label_verified': 'false'} for row in rows]
    expect_error(lambda: merge_metadata(rows, weak, 'source_image_id'), 'Unverified labels require explicit exploratory use.')
    accepted = merge_metadata(rows, weak, 'source_image_id', allow_unverified_labels=True)
    exploratory, _ = make_split(accepted, 42, group_by='source_image_id', allow_unverified_labels=True)
    verify_manifest(exploratory, require_patient=False, allow_unverified_labels=True)
    expect_error(lambda: verify_manifest(exploratory, require_patient=False), 'Exploratory labels cannot pass the default gate.')
    unknown = [{**row, 'label_verified': ''} for row in weak]
    expect_error(lambda: merge_metadata(rows, unknown, 'source_image_id', True), 'Label status must be recorded explicitly.')
    augmented = [{**row, 'is_original': 'false' if row['split'] == 'test' else row['is_original']}
                 for row in exploratory]
    expect_error(lambda: verify_manifest(augmented, require_patient=False, allow_unverified_labels=True),
                 'The exploratory label flag must not disable augmentation/split protection.')
    expect_error(lambda: validate_pair({'task': 'presence', 'label_provenance': 'UNVERIFIED'},
                                      {'task': 'subtype', 'label_provenance': 'CHECKED_SOURCE'}),
                 'A pipeline must not mix label-provenance statuses.')
    anonymous = [{**row, 'patient_id': '', 'source_image_id': '', 'is_original': 'unknown',
                  'label_verified': 'false'} for row in rows]
    pixel_split, _ = make_split(anonymous, 42, group_by='pixel_sha256', allow_unverified_labels=True,
                                allow_unknown_ancestry=True)
    verify_manifest(pixel_split, require_patient=False, allow_unverified_labels=True, allow_unknown_ancestry=True)
    expect_error(lambda: verify_manifest(pixel_split, require_patient=False, allow_unverified_labels=True),
                 'Unknown ancestry must not pass without an explicit opt-in.')
    assert all(row['patient_id'] == row['source_image_id'] == '' for row in pixel_split)
    selected = select_budget(task_rows(pixel_split, 'subtype', 'train'), 'subtype', 2, 42, True)
    assert len(selected) == len({row['pixel_sha256'] for row in selected}) == 4
    leak = [dict(row) for row in pixel_split]
    next(row for row in leak if row['split'] == 'test')['pixel_sha256'] = next(row for row in leak if row['split'] == 'train')['pixel_sha256']
    expect_error(lambda: verify_manifest(leak, require_patient=False, allow_unverified_labels=True,
                                         allow_unknown_ancestry=True), 'Exploratory mode must still reject exact-pixel leakage.')
    duplicate = {**rows[0], 'path': 'different/path.png', 'patient_id': 'another_patient'}
    expect_error(lambda: make_split(rows + [{**duplicate, 'existing_partition': 'test'}], 42),
                 'Duplicates crossing existing test must be rejected.')

    for task in ('presence', 'subtype'):
        train = task_rows(split, task, 'train')
        support = select_budget(train, task, 2, 42)
        assert support == select_budget(list(reversed(train)), task, 2, 42)
        assert len(support) == len({row['group_id'] for row in support}) == 4
        expect_error(lambda: select_budget(train, task, 100, 42), 'Budgets must count independent groups.')
    original = {**next(row for row in split if row['split'] == 'train'), 'path': 'z_original.png'}
    augmented_copy = {**original, 'path': 'a_augmented.png', 'is_original': 'false'}
    assert task_rows([augmented_copy, original], 'presence', 'train') == [original]
    mixed = [dict(path=str(i), label='no_tumor' if i < 3 else 'benign',
                  group_id=group, is_original='true')
             for i, group in enumerate(('a', 'b', 'c', 'a', 'b', 'd'))]
    selected = select_budget(mixed, 'presence', 2, 42)
    assert len({row['group_id'] for row in selected}) == 4
    si, qi = episode_indices([0] * 5 + [1] * 5, 42, 1)
    assert set(si).isdisjoint(qi) and len(si) == 4 and len(qi) == 6
    assert len(si + qi) == len(set(si + qi))

    logits = prototype_logits(torch.tensor([[0., 0.]]), torch.tensor([[2., 0.], [2.2, 0.]]))
    score = torch.softmax(logits, 1)[0, 1].item()
    assert abs(score - 0.3015348) < 1e-5 and score < 0.42
    threshold = choose_threshold([0, 0, 1, 1], [0.1, 0.4, 0.5, 0.8], 1.0)
    assert threshold == 0.5
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
    assert result['fn'] == 1 and result['recall'] == 0.5
    pred = pipeline_labels([0.1, 0.8, 0.9], [0.9, 0.2, 0.8], 0.5, 0.5)
    assert pred.tolist() == [0, 1, 2]  # First malignant query is lost at Stage 1.
    assert binary_metrics([1, 0, 1], [0.09, 0.16, 0.72], 0.5, pred == 2)['fn'] == 1
    ci = patient_bootstrap([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], ['a', 'a', 'b', 'b'], 30)
    assert ci['recall']['low'] == 1.0 and ci['recall']['valid_replicates'] < 30

    torch.set_num_threads(2)
    set_seed(42)
    images = [Image.new('RGB', (40, 40), 'gray'), Image.new('RGB', (40, 40), 'black')]
    config = preprocessing(32)
    for method in ('frozen', 'finetune', 'protonet'):
        model = build_model(method, pretrained=False).eval()
        artifact = dict(format_version=1, method=method, task='subtype', classes=['benign', 'malignant'],
                        preprocessing=config, threshold=0.5, threshold_source='val', manifest_sha256='fixture',
                        state_dict=model.state_dict(), prototypes=None if method == 'finetune' else torch.randn(2, 512))
        expected = predict_images(model, artifact, images)
        path = root / f'{method}.pt'
        torch.save(artifact, path)
        loaded, locked = load_artifact(path)
        assert np.allclose(expected, predict_images(loaded, locked, images), atol=1e-6)
        artifact['threshold_source'] = 'test'
        invalid = root / f'{method}_invalid.pt'
        torch.save(artifact, invalid)
        expect_error(lambda: load_artifact(invalid), 'A test-tuned threshold must not be accepted.')
        del model, loaded, artifact
    print('PASS: grouped splits, metadata gates, budgets, disjoint episodes, metrics and artifact inference parity.')
    return split


def check_app(run, root):
    import io
    import os
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    config_path = root / 'app-test.yaml'
    config_path.write_text(f'presence_artifact: { (run / "presence/model.pt").as_posix() }\n'
                           f'subtype_artifact: { (run / "subtype/model.pt").as_posix() }\n', encoding='utf-8')
    image = Image.new('RGB', (40, 40), 'gray')
    uploaded = io.BytesIO()
    image.save(uploaded, format='PNG')
    uploaded.name = 'fixture.png'
    app_path = Path(__file__).parent / 'app' / 'app.py'
    with patch.dict(os.environ, {'LIVER_CT_CONFIG': str(config_path)}), \
            patch('streamlit.file_uploader', return_value=[uploaded]):
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        assert not app.exception and len(app.button) == 1
        app.button[0].click().run()
        assert not app.exception
        assert any('Kết quả dự đoán' in element.value for element in app.markdown)
    with patch.dict(os.environ, {'LIVER_CT_CONFIG': str(config_path)}), \
            patch('streamlit.file_uploader', return_value=[]):
        app.run()
        assert not app.exception
        assert not any('Kết quả dự đoán' in element.value for element in app.markdown)
    print('PASS: Streamlit artifact loading, classification and stale-result clearing.')


def main():
    # Caller supplies a leased output root; the checks never delete or alter source data.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--app-run', type=Path, help='Optional presence/subtype artifact pair for the UI check.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    split = check(args.out)
    write_csv(args.out / 'manifest.csv', split)
    if args.app_run:
        check_app(args.app_run.resolve(), args.out)
    print(f'Synthetic integration fixture: {args.out}')


if __name__ == '__main__':
    main()
