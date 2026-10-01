"""Matched-label baselines, validation-only tuning, and complete two-stage evaluation."""
import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix, f1_score

from liver_ct import (CLASSES, METHODS, PIPELINE_CLASSES, binary_metrics, build_model,
                      choose_threshold, class_prototypes, episode_indices, patient_bootstrap,
                      pipeline_labels, predict_images, preprocessing, prototype_logits,
                      select_budget, set_seed, task_label, task_rows, tensor_images, validate_pair)
from prepare_data import image_hash, read_csv, resolve_image, verify_manifest, write_csv

PREDICTION_FIELDS = ['path', 'patient_id', 'group_id', 'true_label', 'predicted_label',
                     'positive_score', 'presence_score', 'subtype_score', 'error_stage']


def open_images(rows, root):
    images = []
    for row in rows:
        with Image.open(resolve_image(root, row['path'])) as image:
            images.append(image.convert('RGB').copy())
    return images


@torch.no_grad()
def support_prototypes(model, rows, root, task, config, device, batch_size):
    model.eval()
    embeddings = []
    for offset in range(0, len(rows), batch_size):
        images = open_images(rows[offset:offset + batch_size], root)
        embeddings.append(model(tensor_images(images, config).to(device)).float())
    labels = torch.tensor([task_label(row, task) for row in rows], device=device)
    return class_prototypes(torch.cat(embeddings), labels).cpu()


def evaluate_scores(model, artifact, rows, root, device, batch_size):
    values = []
    for offset in range(0, len(rows), batch_size):
        images = open_images(rows[offset:offset + batch_size], root)
        values.extend(predict_images(model, artifact, images, device).tolist())
    return np.asarray(values)


def fit_stage(method, task, support, validation, root, args, seed, budget, manifest_hash):
    set_seed(seed)
    model = build_model(method, pretrained=not args.random_init).to(args.device)
    config = preprocessing(args.image_size)
    artifact = dict(format_version=1, method=method, task=task, classes=CLASSES[task],
                    preprocessing=config, threshold=0.5, threshold_source='val',
                    manifest_sha256=manifest_hash, group_by=args.group_by, label_provenance=args.label_provenance,
                    budget_per_class=budget, seed=seed,
                    support_paths=[row['path'] for row in support],
                    support_pixel_sha256=[row['pixel_sha256'] for row in support],
                    annotation_count=len(support), training_groups=len(support), budget_unit=args.budget_unit,
                    ancestry_status='UNKNOWN' if args.exploratory_files else 'KNOWN_IMAGE_GROUPS',
                    validation_label_count=len(validation), code_sha256=args.code_sha256,
                    pretrained_weights='random' if args.random_init else 'ResNet34_Weights.IMAGENET1K_V1',
                    episode_support_per_class=min(args.episode_support, budget // 2),
                    episode_query_per_class=min(args.episode_query, budget - min(args.episode_support, budget // 2)))
    start = time.perf_counter()
    history, best_f1, best_state, best_step, stale = [], -1.0, None, 0, 0
    labels = [task_label(row, task) for row in support]
    images = open_images(support, root)
    if method != 'frozen':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
    for step in range(0 if method == 'frozen' else 1, 1 if method == 'frozen' else args.steps + 1):
        loss_value = None
        if method != 'frozen':
            model.train()
            optimizer.zero_grad(set_to_none=True)
            si, qi = episode_indices(labels, seed, step, args.episode_support, args.episode_query)
            if method == 'protonet':
                xs = tensor_images([images[i] for i in si], config, training=True).to(args.device)
                xq = tensor_images([images[i] for i in qi], config, training=True).to(args.device)
                ys = torch.tensor([labels[i] for i in si], device=args.device)
                yq = torch.tensor([labels[i] for i in qi], device=args.device)
                protos = class_prototypes(model(xs), ys)
                loss = criterion(prototype_logits(model(xq), protos), yq)
            else:
                indices = si + qi
                inputs = tensor_images([images[i] for i in indices], config, training=True).to(args.device)
                targets = torch.tensor([labels[i] for i in indices], device=args.device)
                loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
        if method == 'frozen' or step == 1 or step % args.val_every == 0 or step == args.steps:
            artifact['prototypes'] = (None if method == 'finetune' else
                                      support_prototypes(model, support, root, task, config, args.device, args.batch_size))
            val_scores = evaluate_scores(model, artifact, validation, root, args.device, args.batch_size)
            val_truth = [task_label(row, task) for row in validation]
            metrics = binary_metrics(val_truth, val_scores, 0.5)
            history.append(dict(step=step, train_loss=loss_value, **metrics))
            if metrics['macro_f1'] > best_f1 + 1e-6:
                best_f1, best_step, stale = metrics['macro_f1'], step, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            print(f'{method} {task} K={budget} seed={seed} step={step}: val macro-F1={metrics["macro_f1"]:.4f}', flush=True)
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    artifact['state_dict'] = best_state
    artifact['prototypes'] = (None if method == 'finetune' else
                              support_prototypes(model, support, root, task, config, args.device, args.batch_size))
    val_scores = evaluate_scores(model, artifact, validation, root, args.device, args.batch_size)
    val_truth = [task_label(row, task) for row in validation]
    artifact['threshold'] = choose_threshold(val_truth, val_scores, args.target_recall)
    artifact['validation_metrics'] = binary_metrics(val_truth, val_scores, artifact['threshold'])
    artifact.update(best_step=best_step, completed_steps=step, training_seconds=time.perf_counter() - start,
                    target_recall=args.target_recall)
    return model, artifact, history


def error_gallery(rows, root, output):
    if not rows:
        return
    selected = rows[:12]
    gallery = Image.new('RGB', (4 * 240, ((len(selected) + 3) // 4) * 280), 'white')
    draw = ImageDraw.Draw(gallery)
    for index, row in enumerate(selected):
        image = open_images([row], root)[0]
        image.thumbnail((224, 224))
        x, y = (index % 4) * 240, (index // 4) * 280
        gallery.paste(image, (x, y))
        draw.text((x, y + 228), f'{row["true_label"]} -> {row["predicted_label"]}', fill='black')
        draw.text((x, y + 244), row['error_stage'], fill='black')
    gallery.save(output)


def save_predictions(folder, rows, root):
    write_csv(folder / 'predictions.csv', rows, PREDICTION_FIELDS)
    errors = [row for row in rows if row['true_label'] != row['predicted_label']]
    write_csv(folder / 'errors.csv', errors, PREDICTION_FIELDS)
    error_gallery(errors, root, folder / 'errors.png')


def stage_test(model, artifact, rows, root, args, folder):
    scores = evaluate_scores(model, artifact, rows, root, args.device, args.batch_size)
    truth = [task_label(row, artifact['task']) for row in rows]
    predictions = (scores >= artifact['threshold']).astype(int)
    metrics = binary_metrics(truth, scores, artifact['threshold'])
    metrics.update(label_provenance=artifact['label_provenance'], evidence_status=artifact['evidence_status'],
                   budget_unit=artifact['budget_unit'], ci_group_unit=artifact['group_by'])
    metrics['test_groups'] = len({row['group_id'] for row in rows})
    metrics['ci95'] = patient_bootstrap(truth, scores, predictions, [r['group_id'] for r in rows],
                                      args.bootstrap, artifact['seed']) if args.bootstrap else None
    records = []
    for row, score, true, pred in zip(rows, scores, truth, predictions):
        records.append(dict(path=row['path'], patient_id=row['patient_id'], group_id=row['group_id'],
                            true_label=artifact['classes'][true], predicted_label=artifact['classes'][pred],
                            positive_score=float(score), presence_score='', subtype_score='',
                            error_stage=artifact['task'] if true != pred else ''))
    save_predictions(folder, records, root)
    return metrics


def pipeline_test(presence_model, presence, subtype_model, subtype, rows, root, args, folder):
    validate_pair(presence, subtype)
    selected, seen = [], set()
    for row in rows:
        if row['split'] == 'test' and row['label'] in PIPELINE_CLASSES and row['pixel_sha256'] not in seen:
            selected.append(row)
            seen.add(row['pixel_sha256'])
    p1 = evaluate_scores(presence_model, presence, selected, root, args.device, args.batch_size)
    p2 = evaluate_scores(subtype_model, subtype, selected, root, args.device, args.batch_size)
    truth = np.array([PIPELINE_CLASSES.index(r['label']) for r in selected])
    pred = pipeline_labels(p1, p2, presence['threshold'], subtype['threshold'])
    # Product is a ranking score, not a calibrated probability of malignancy.
    scores = p1 * p2
    metrics = binary_metrics(truth == 2, scores, 0.5, predictions=pred == 2)
    metrics.pop('threshold')
    metrics.update(label_provenance=presence['label_provenance'], evidence_status=presence['evidence_status'],
                   budget_unit=presence['budget_unit'], ci_group_unit=presence['group_by'],
                   accuracy=float(np.mean(truth == pred)),
                   macro_f1=float(f1_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
                   confusion_matrix=confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
                   classes=PIPELINE_CLASSES, presence_threshold=presence['threshold'], subtype_threshold=subtype['threshold'],
                   score_definition='presence_score * subtype_score; uncalibrated ranking score',
                   untyped_test_tumor_rows_excluded=sum(r['split'] == 'test' and r['label'] == 'tumor' for r in rows),
                   test_groups=len({r['group_id'] for r in selected}))
    metrics['ci95'] = patient_bootstrap(truth == 2, scores, pred == 2, [r['group_id'] for r in selected],
                                      args.bootstrap, presence['seed']) if args.bootstrap else None
    records = []
    for row, true, predicted, score, a, b in zip(selected, truth, pred, scores, p1, p2):
        error_stage = '' if true == predicted else ('presence' if (true == 0) != (predicted == 0) else 'subtype')
        records.append(dict(path=row['path'], patient_id=row['patient_id'], group_id=row['group_id'],
                            true_label=PIPELINE_CLASSES[true], predicted_label=PIPELINE_CLASSES[predicted],
                            positive_score=float(score), presence_score=float(a), subtype_score=float(b),
                            error_stage=error_stage))
    save_predictions(folder, records, root)
    return metrics


def summarize(results, output):
    groups = defaultdict(list)
    for result in results:
        groups[(result['method'], result['budget'], result['task'])].append(result)
    summary = []
    for (method, budget, task), runs in sorted(groups.items()):
        row = dict(method=method, budget=budget, task=task, seeds=len(runs))
        row.update(evidence_status=runs[0]['metrics']['evidence_status'], budget_unit=runs[0]['metrics']['budget_unit'])
        for metric in ('accuracy', 'recall', 'specificity', 'macro_f1', 'auc', 'fn', 'fp'):
            values = [r['metrics'][metric] for r in runs if r['metrics'][metric] is not None]
            row[metric + '_mean'] = float(np.mean(values)) if values else None
            row[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    write_csv(output / 'summary.csv', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in CLASSES.keys() | {'pipeline'}:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for method in METHODS:
            rows = sorted([r for r in summary if r['task'] == task and r['method'] == method], key=lambda r: r['budget'])
            if not rows:
                continue
            for ax, metric in zip(axes, ('recall', 'macro_f1')):
                ax.errorbar([r['budget'] for r in rows], [r[metric + '_mean'] for r in rows],
                            yerr=[r[metric + '_std'] or 0 for r in rows], marker='o', label=method)
                unit = 'Unique-pixel training images per class' if rows[0]['budget_unit'] == 'UNIQUE_PIXEL_IMAGES' else 'Labeled originals per class'
                ax.set(xlabel=unit, ylabel=metric, ylim=(0, 1.05))
                ax.grid(alpha=0.3)
                ax.legend()
        fig.suptitle(f'{task}: mean and standard deviation across seeds')
        fig.tight_layout()
        fig.savefig(output / f'{task}_budget_curve.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--budgets', nargs='+', type=int, default=[5, 10, 30])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--val-every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10, help='Number of validation checks without improvement.')
    parser.add_argument('--episode-support', type=int, default=5)
    parser.add_argument('--episode-query', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--target-recall', type=float, default=0.9)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--random-init', action='store_true', help='Integration checks only; disables ImageNet initialization.')
    parser.add_argument('--allow-image-groups', action='store_true', help='Explicit exploratory fallback without patient identities.')
    parser.add_argument('--allow-unverified-labels', action='store_true',
                        help='Explicit exploratory agreement with provided labels; not confirmed diagnoses.')
    parser.add_argument('--exploratory-files', action='store_true',
                        help='Explicit technical comparison with unknown patient/original/augmentation ancestry.')
    args = parser.parse_args()
    if any(value < 1 for value in (args.steps, args.val_every, args.patience, args.batch_size, args.threads,
                                   args.episode_support, args.episode_query)) or args.image_size < 32:
        parser.error('Step/batch/thread counts must be positive and image size >=32.')
    if args.bootstrap < 0 or args.lr <= 0 or not 0 < args.target_recall <= 1 or min(args.budgets) < 2:
        parser.error('Invalid bootstrap count, learning rate, recall target or label budget (minimum 2).')
    if any(len(values) != len(set(values)) for values in (args.methods, args.budgets, args.seeds)):
        parser.error('Methods, budgets and seeds must be unique.')
    torch.set_num_threads(args.threads)
    rows = read_csv(args.manifest)
    args.group_by = 'pixel_sha256' if args.exploratory_files else ('source_image_id' if args.allow_image_groups else 'patient_id')
    args.budget_unit = 'UNIQUE_PIXEL_IMAGES' if args.exploratory_files else 'GROUPED_ORIGINAL_IMAGES'
    verify_manifest(rows, require_patient=not (args.allow_image_groups or args.exploratory_files),
                    allow_unverified_labels=args.allow_unverified_labels or args.exploratory_files,
                    allow_unknown_ancestry=args.exploratory_files)
    args.label_provenance = 'UNVERIFIED' if any(r['label_verified'].lower() != 'true' for r in rows) else 'CHECKED_SOURCE'
    for row in rows:
        if image_hash(resolve_image(args.data_root, row['path'])) != row['pixel_sha256']:
            raise ValueError(f'Image changed after preparing the manifest: {row["path"]}')
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    supports = {(task, budget, seed): select_budget(task_rows(rows, task, 'train'), task, budget, seed,
                                                  allow_unknown_ancestry=args.exploratory_files)
                for task in CLASSES for budget in args.budgets for seed in args.seeds}
    args.code_sha256 = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in ('prepare_data.py', 'liver_ct.py', 'run_experiments.py')}
    args.out.mkdir(parents=True, exist_ok=False)
    torch.hub.set_dir(str(args.out / 'pretrained-cache'))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    import PIL
    import sklearn
    import torchvision
    limitations = ['UNKNOWN_ANCESTRY'] if args.exploratory_files else (['IMAGE_GROUPS'] if args.allow_image_groups else [])
    if args.label_provenance == 'UNVERIFIED':
        limitations.append('UNVERIFIED_LABELS')
    config.update(manifest_sha256=manifest_hash, python=platform.python_version(), torch=str(torch.__version__),
                  torchvision=torchvision.__version__, numpy=np.__version__, sklearn=sklearn.__version__, pillow=PIL.__version__,
                  device_name=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else platform.processor(),
                  cuda_runtime=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                  evidence_status='INTEGRATION_ONLY' if args.random_init else
                                  ('EXPLORATORY_' + '_AND_'.join(limitations) if limitations else 'PATIENT_GROUPED'))
    (args.out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    results = []
    for method in args.methods:
        for budget in args.budgets:
            for seed in args.seeds:
                pair = {}
                run = args.out / f'{method}_k{budget}_seed{seed}'
                run.mkdir()
                for task in CLASSES:
                    folder = run / task
                    folder.mkdir()
                    model, artifact, history = fit_stage(method, task, supports[(task, budget, seed)],
                        task_rows(rows, task, 'val'), args.data_root, args, seed, budget, manifest_hash)
                    artifact['evidence_status'] = config['evidence_status']
                    torch.save(artifact, folder / 'model.pt')
                    write_csv(folder / 'history.csv', history)
                    metrics = stage_test(model, artifact, task_rows(rows, task, 'test'), args.data_root, args, folder)
                    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
                    results.append(dict(method=method, budget=budget, seed=seed, task=task, metrics=metrics))
                    pair[task] = (model.to('cpu'), artifact)
                folder = run / 'pipeline'
                folder.mkdir()
                presence_model, presence = pair['presence']
                subtype_model, subtype = pair['subtype']
                metrics = pipeline_test(presence_model.to(args.device), presence, subtype_model.to(args.device),
                                        subtype, rows, args.data_root, args, folder)
                (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
                results.append(dict(method=method, budget=budget, seed=seed, task='pipeline', metrics=metrics))
                (args.out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
                del pair, presence_model, subtype_model, model
    summarize(results, args.out)
    print(f'Done: {args.out}; evidence={config["evidence_status"]}', flush=True)


if __name__ == '__main__':
    main()
