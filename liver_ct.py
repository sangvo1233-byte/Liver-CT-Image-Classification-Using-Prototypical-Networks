"""Shared preprocessing, models, scores and inference for experiments and the app."""
import random

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from torch import nn
from torchvision import models, transforms

METHODS = ('frozen', 'finetune', 'protonet')
CLASSES = {'presence': ['no_tumor', 'tumor'], 'subtype': ['benign', 'malignant']}
PIPELINE_CLASSES = ['no_tumor', 'benign', 'malignant']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocessing(image_size=224):
    return dict(image_size=image_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                crop='none', distance='squared_euclidean')


def image_transform(config, training=False):
    if config['crop'] != 'none' or config['distance'] != 'squared_euclidean':
        raise ValueError('Unsupported preprocessing or prototype distance.')
    operations = [transforms.Resize((config['image_size'], config['image_size']))]
    if training:
        operations += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                       transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    operations += [transforms.ToTensor(), transforms.Normalize(config['mean'], config['std'])]
    return transforms.Compose(operations)


def tensor_images(images, config, training=False):
    transform = image_transform(config, training)
    return torch.stack([transform(image.convert('RGB')) for image in images])


class ProtoNet(nn.Module):
    """ResNet-34 and the existing v3 projection head."""
    def __init__(self, pretrained=True, dropout=0.15):
        super().__init__()
        self.encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, images):
        return self.projection(self.encoder(images))


def build_model(method, pretrained=True):
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    if method == 'protonet':
        return ProtoNet(pretrained)
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2) if method == 'finetune' else nn.Identity()
    if method == 'frozen':
        model.requires_grad_(False)
    return model


def prototype_logits(embeddings, prototypes):
    return -torch.cdist(embeddings.float(), prototypes.float()).square()


def class_prototypes(embeddings, labels):
    if set(labels.detach().cpu().tolist()) != {0, 1}:
        raise ValueError('Support must cover both classes.')
    return torch.stack([embeddings[labels == label].mean(0) for label in (0, 1)])


def task_label(row, task):
    if task == 'presence':
        return int(row['label'] != 'no_tumor')
    return CLASSES['subtype'].index(row['label'])


def task_rows(rows, task, split):
    selected = [row for row in rows if row['split'] == split
                and (task == 'presence' or row['label'] in CLASSES['subtype'])]
    # Identical decoded images are one example, even if copied into another task folder.
    unique = {}
    for row in sorted(selected, key=lambda row: (row['is_original'].lower() != 'true', row['path'])):
        key = (row['pixel_sha256'], task_label(row, task))
        unique.setdefault(key, row)
    return list(unique.values())


def select_budget(rows, task, budget, seed, allow_unknown_ancestry=False):
    rng, selected, classes = random.Random(seed), [], []
    for label in (0, 1):
        grouped = {}
        for row in sorted(rows, key=lambda row: row['path']):
            eligible = row['is_original'].lower() == 'true' or (allow_unknown_ancestry and row['is_original'].lower() == 'unknown')
            if task_label(row, task) == label and eligible:
                grouped.setdefault(row['group_id'], []).append(row)
        if len(grouped) < budget:
            raise ValueError(f'{task}: class {label} has {len(grouped)} eligible groups, needs {budget}.')
        classes.append(grouped)
    if len(set(classes[0]) | set(classes[1])) < 2 * budget:
        raise ValueError('Not enough distinct groups for a disjoint two-class annotation budget.')
    # A patient can have both tumor and no-tumor slices. Reserve exclusive groups
    # first so the positive class still has K independent groups available.
    exclusive = sorted(set(classes[0]) - set(classes[1]))
    negative = rng.sample(exclusive, min(budget, len(exclusive)))
    negative += rng.sample(sorted(set(classes[0]) & set(classes[1])), budget - len(negative))
    positive = rng.sample(sorted(set(classes[1]) - set(negative)), budget)
    for grouped, chosen in zip(classes, (negative, positive)):
        selected.extend(rng.choice(grouped[group]) for group in chosen)
    return selected


def episode_indices(labels, seed, step, support_per_class=5, query_per_class=5):
    rng = random.Random(seed + step)
    support, query = [], []
    for label in (0, 1):
        indices = [i for i, value in enumerate(labels) if value == label]
        if len(indices) < 2:
            raise ValueError('Episodic training needs at least two independent samples per class.')
        rng.shuffle(indices)
        k = min(support_per_class, len(indices) // 2)
        support.extend(indices[:k])
        query.extend(indices[k:k + min(query_per_class, len(indices) - k)])
    return support, query


@torch.no_grad()
def score_tensor(model, images, method, prototypes=None):
    model.eval()
    outputs = model(images)
    logits = outputs if method == 'finetune' else prototype_logits(outputs, prototypes)
    return torch.softmax(logits.float(), dim=1)[:, 1]


@torch.no_grad()
def predict_images(model, artifact, images, device='cpu'):
    batch = tensor_images(images, artifact['preprocessing']).to(device)
    prototypes = artifact.get('prototypes')
    if prototypes is not None:
        prototypes = prototypes.to(device)
    return score_tensor(model, batch, artifact['method'], prototypes).cpu().numpy()


def choose_threshold(y_true, scores, target_recall=0.9):
    if not 0 < target_recall <= 1 or set(np.asarray(y_true).tolist()) != {0, 1}:
        raise ValueError('Threshold tuning needs both classes and a recall target in (0, 1].')
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    best = max(eligible, key=lambda i: (precision[i], thresholds[i]))
    return float(thresholds[best])


def binary_metrics(y_true, scores, threshold, predictions=None):
    truth = np.asarray(y_true, dtype=int)
    if not len(truth):
        raise ValueError('Cannot evaluate an empty set.')
    pred = (np.asarray(scores) >= threshold).astype(int) if predictions is None else np.asarray(predictions)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    return dict(n=len(truth), accuracy=float(accuracy_score(truth, pred)),
                precision=float(precision_score(truth, pred, zero_division=0)),
                recall=float(recall_score(truth, pred, zero_division=0)),
                specificity=float(tn / (tn + fp)) if tn + fp else None,
                f1=float(f1_score(truth, pred, zero_division=0)),
                macro_f1=float(f1_score(truth, pred, labels=[0, 1], average='macro', zero_division=0)),
                auc=float(roc_auc_score(truth, scores)) if len(set(truth)) == 2 else None,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(threshold))


def pipeline_labels(presence_scores, subtype_scores, presence_threshold, subtype_threshold):
    return np.where(np.asarray(presence_scores) < presence_threshold, 0,
                    np.where(np.asarray(subtype_scores) >= subtype_threshold, 2, 1))


def patient_bootstrap(truth, scores, predictions, groups, repeats=200, seed=42):
    rng, values = np.random.default_rng(seed), {'recall': [], 'auc': []}
    groups = np.asarray(groups)
    unique = np.unique(groups)
    truth, scores, predictions = map(np.asarray, (truth, scores, predictions))
    for _ in range(repeats):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        if len(set(truth[indices])) != 2:
            continue
        values['recall'].append(float(recall_score(truth[indices], predictions[indices], zero_division=0)))
        values['auc'].append(float(roc_auc_score(truth[indices], scores[indices])))
    return {name: {'low': float(np.quantile(data, 0.025)), 'high': float(np.quantile(data, 0.975)),
                   'valid_replicates': len(data)} if data else None for name, data in values.items()}


def load_artifact(path, device='cpu'):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if artifact.get('format_version') != 1 or artifact.get('method') not in METHODS:
        raise ValueError('Use a protocol artifact generated by run_experiments.py, not a legacy checkpoint.')
    if artifact.get('classes') != CLASSES.get(artifact.get('task')):
        raise ValueError('Artifact class mapping does not match its task.')
    if not np.isfinite(artifact['threshold']) or not 0 <= artifact['threshold'] <= 1:
        raise ValueError('Invalid decision threshold.')
    if artifact.get('threshold_source') != 'val' or not artifact.get('manifest_sha256'):
        raise ValueError('Artifact lacks validation-threshold/split provenance.')
    image_transform(artifact['preprocessing'])
    model = build_model(artifact['method'], pretrained=False)
    model.load_state_dict(artifact['state_dict'], strict=True)
    model.to(device).eval()
    prototypes = artifact.get('prototypes')
    if artifact['method'] != 'finetune' and (prototypes is None or tuple(prototypes.shape) != (2, 512)):
        raise ValueError('Artifact lacks the two locked support prototypes.')
    return model, artifact


def validate_pair(presence, subtype):
    if presence['task'] != 'presence' or subtype['task'] != 'subtype':
        raise ValueError('The pipeline needs presence and subtype artifacts in that order.')
    if presence.get('label_provenance') != subtype.get('label_provenance'):
        raise ValueError('Pipeline artifacts disagree on label provenance.')
    if presence.get('budget_unit') != subtype.get('budget_unit'):
        raise ValueError('Pipeline artifacts disagree on budget unit.')
    for key in ('manifest_sha256', 'group_by', 'preprocessing', 'method', 'budget_per_class', 'seed',
                'pretrained_weights', 'evidence_status'):
        if presence[key] != subtype[key]:
            raise ValueError(f'Pipeline artifacts disagree on {key}.')
