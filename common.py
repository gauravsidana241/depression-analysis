import os
import csv
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORPUS = "./Androids-Corpus/Androids-Corpus"

FEATURE_PATHS = {
    "po":  "./androids_is09_participant_clips.npz",
    "raw": "./androids_is09_02.npz",
}

SEGMENT_LENGTHS = [32, 64, 128, 256, 512, 1024]
SEEDS = (0, 1, 2, 3, 4)


# Corpus metadata

def load_metadata():
    interview_task_path = f"{CORPUS}/Interview-Task/audio"
    rows = []
    for condition in os.listdir(interview_task_path):
        if condition == ".DS_Store":
            continue
        for clip in os.listdir(f"{interview_task_path}/{condition}"):
            stem = clip.replace(".wav", "")
            uid, mid, t = stem.split("_")
            rows.append({
                "speaker_id": f"{uid}_{mid[0]}",
                "condition": mid[0],
                "gender": mid[1],
                "age": int(mid[2:]),
                "education_level": t,
                "full_recording_path": f"{interview_task_path}/{condition}/{clip}",
            })
    interview_df = pd.DataFrame(rows)

    label_of = {sid: 1 if c == "P" else 0 for sid, c in zip(interview_df.speaker_id, interview_df.condition)}
    gender_of = dict(zip(interview_df.speaker_id, interview_df.gender))

    fold_list = pd.read_csv(f"{CORPUS}/fold-lists.csv", header=None, skiprows=2)
    interview_folds = {}
    for i, col in enumerate(range(7, 12), start=1):
        interview_folds[i] = fold_list[col].dropna().str.strip("'").tolist()

    return interview_df, label_of, gender_of, interview_folds


def load_turn_times():
    """Participant turn boundaries, keyed by speaker_id."""
    turns = {}
    with open(f"{CORPUS}/interview_timedata.csv", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            stem = row[0]
            vals = [float(x) for x in row[1:] if x.strip() != ""]
            assert len(vals) % 2 == 0, f"{stem} has odd count: {len(vals)}"
            turns[stem] = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]

    by_speaker = {}
    for full_stem, pairs in turns.items():
        uid, mid, _ = full_stem.split("_")
        by_speaker[f"{uid}_{mid[0]}"] = pairs
    return by_speaker


def load_features(condition):
    """Load the IS09 feature sequences for one audio condition."""
    path = FEATURE_PATHS[condition]
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def fold_speakers(interview_folds, k):
    """Test and train speaker ids for fold k."""
    test = [s[:4] for s in interview_folds[k]]
    train = [s[:4] for f, sp in interview_folds.items() if f != k for s in sp]
    return train, test


# Metrics

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    # zero_division=0 avoids warnings/crashes on folds with few positive predictions
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return {"acc": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1)}


def random_baseline(label_of, n_trials=1000, seed=0):
    """Expected performance of a classifier guessing according to the priors."""
    rng = np.random.default_rng(seed)
    n = len(label_of)
    p_patient = sum(label_of.values()) / n

    acc = {'acc': [], 'precision': [], 'recall': [], 'f1': []}
    for _ in range(n_trials):
        y_true = rng.choice([0, 1], size=n, p=[1 - p_patient, p_patient])
        y_pred = rng.choice([0, 1], size=n, p=[1 - p_patient, p_patient])
        m = compute_metrics(y_true, y_pred)
        for key in acc:
            acc[key].append(m[key])
    return {key: float(np.mean(vals)) for key, vals in acc.items()}


# Dataset and model

class AndroidsSegmentDataset(Dataset):
    def __init__(self, data_dict, speaker_ids, labels_dict, segment_length=128):
        self.segments = []
        self.labels = []
        self.speaker_ids = []
        self.segment_length = segment_length

        for sid in speaker_ids:
            if sid not in data_dict:
                continue

            X = data_dict[sid]
            y = labels_dict[sid]

            n_segments = len(X) // segment_length
            if n_segments == 0:
                continue

            X_truncated = X[:n_segments * segment_length]
            X_reshaped = X_truncated.reshape(n_segments, segment_length, -1)

            for seg in X_reshaped:
                self.segments.append(seg)
                self.labels.append(y)
                self.speaker_ids.append(sid)

        self.segments = torch.tensor(np.array(self.segments), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx], self.speaker_ids[idx]


class StackedRNN(nn.Module):
    def __init__(self, input_size=32, hidden_size=70, num_layers=2):
        super(StackedRNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=0.3  # overfitting issue
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_step_out = rnn_out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for segments, labels, _ in train_loader:
        segments = segments.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(segments)
        loss = criterion(outputs, labels)

        loss.backward()

        # gradient clipping to address sharp spikes in fold 4
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * segments.size(0)

    return running_loss / len(train_loader.dataset)


def evaluate_loss(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for segments, labels, _ in data_loader:
            segments = segments.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(segments)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * segments.size(0)

    return running_loss / len(data_loader.dataset)


def evaluate_speakers(model, data_loader):
    model.eval()

    speaker_probs = {}
    speaker_labels = {}

    with torch.no_grad():
        for segments, labels, speaker_ids in data_loader:
            segments = segments.to(device)

            logits = model(segments)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = labels.cpu().numpy()

            for i, sid in enumerate(speaker_ids):
                if sid not in speaker_probs:
                    speaker_probs[sid] = []
                    speaker_labels[sid] = int(labels[i])
                speaker_probs[sid].append(probs[i])

    y_true, y_pred_smv, y_pred_wa, sids = [], [], [], []

    for sid in speaker_probs:
        probs = np.array(speaker_probs[sid])
        y_true.append(speaker_labels[sid])
        sids.append(sid)

        # segment majority vote (smv)
        segment_classes = (probs > 0.5).astype(int)
        y_pred_smv.append(int(np.mean(segment_classes) > 0.5))

        # weighted average (wa)
        y_pred_wa.append(int(np.mean(probs) > 0.5))

    return y_true, y_pred_smv, y_pred_wa, sids


# Training routines

def run_rnn(data, label_of, interview_folds, segment_length,
            folds=5, hidden_size=70, epochs=30, verbose=False):
    smv_metrics, wa_metrics = [], []
    pooled_preds = []
    loss_curves = {}

    speaker_keys = list(data.keys())

    for k in range(1, folds + 1):
        train_speakers, test_speakers = fold_speakers(interview_folds, k)

        X_train = np.vstack([data[s] for s in train_speakers if s in data])
        scaler = StandardScaler()
        scaler.fit(X_train)
        scaled_data = {sid: scaler.transform(data[sid]) for sid in speaker_keys}

        train_dataset = AndroidsSegmentDataset(
            data_dict=scaled_data, labels_dict=label_of,
            speaker_ids=train_speakers, segment_length=segment_length
        )
        test_dataset = AndroidsSegmentDataset(
            data_dict=scaled_data, labels_dict=label_of,
            speaker_ids=test_speakers, segment_length=segment_length
        )

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"  [seg_len={segment_length}, fold={k}] skipped - empty split")
            continue

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = StackedRNN(input_size=32, hidden_size=hidden_size, num_layers=2).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

        fold_train_losses, fold_test_losses = [], []
        iterator = tqdm(range(epochs), desc=f"seg_len={segment_length} fold={k}") if verbose else range(epochs)
        for _ in iterator:
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            test_loss = evaluate_loss(model, test_loader, criterion)
            fold_train_losses.append(train_loss)
            fold_test_losses.append(test_loss)

        loss_curves[k] = {"train": fold_train_losses, "test": fold_test_losses}

        y_true, y_pred_smv, y_pred_wa, sids = evaluate_speakers(model, test_loader)
        smv_metrics.append(compute_metrics(y_true, y_pred_smv))
        wa_metrics.append(compute_metrics(y_true, y_pred_wa))

        for sid, yt, yp_smv, yp_wa in zip(sids, y_true, y_pred_smv, y_pred_wa):
            pooled_preds.append({"sid": sid, "y_true": int(yt), "y_pred_smv": int(yp_smv), "y_pred_wa": int(yp_wa)})

    return smv_metrics, wa_metrics, loss_curves, pooled_preds


def run_rnn_seeds(data, label_of, interview_folds, segment_length, seeds=SEEDS, **kw):
    metrics = ["acc", "precision", "recall", "f1"]
    per_seed_smv, per_seed_wa = [], []
    all_pooled = []
    loss_curves_seed0 = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        smv_folds, wa_folds, curves, pooled = run_rnn(
            data, label_of, interview_folds, segment_length, **kw
        )
        per_seed_smv.append(smv_folds)
        per_seed_wa.append(wa_folds)

        if seed == seeds[0]:
            loss_curves_seed0 = curves

        for p in pooled:
            all_pooled.append({**p, "seed": int(seed)})

    n_folds = len(per_seed_smv[0])
    for r in per_seed_smv + per_seed_wa:
        assert len(r) == n_folds, f"inconsistent fold count across repetitions: {len(r)} vs {n_folds}"

    def average_within_folds(per_seed):
        return [
            {m: float(np.mean([per_seed[r][k][m] for r in range(len(per_seed))])) for m in metrics}
            for k in range(n_folds)
        ]

    return average_within_folds(per_seed_smv), average_within_folds(per_seed_wa), loss_curves_seed0, all_pooled


def run_lr(data, label_of, interview_folds, segment_length, folds=5):
    smv_metrics, wa_metrics = [], []
    pooled_preds = []

    for k in range(1, folds + 1):
        train_speakers, test_speakers = fold_speakers(interview_folds, k)

        assert not set(train_speakers) & set(test_speakers)
        assert all(s in data for s in train_speakers + test_speakers)

        X_train = np.vstack([data[s] for s in train_speakers])
        Y_train = np.concatenate([np.full(len(data[s]), label_of[s]) for s in train_speakers])

        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)

        clf = LogisticRegression(max_iter=1000).fit(X_train, Y_train)

        Y_true, Y_pred_smv, Y_pred_wa, sids = [], [], [], []
        for sid in test_speakers:
            X_test = scalar.transform(data[sid])

            n_segments = len(X_test) // segment_length
            if n_segments == 0:
                continue

            feature_segments = np.array_split(X_test[:n_segments * segment_length], n_segments)

            seg_labels_smv, seg_probs_wa = [], []
            for seg in feature_segments:
                probs = clf.predict_proba(seg)[:, 1]

                frame_preds = (probs > 0.5).astype(int)
                seg_labels_smv.append(int(frame_preds.mean() > 0.5))

                seg_probs_wa.append(probs.mean())

            Y_pred_smv.append(int(np.mean(seg_labels_smv) > 0.5))
            Y_pred_wa.append(int(np.mean(seg_probs_wa) > 0.5))
            Y_true.append(label_of[sid])
            sids.append(sid)

        smv_metrics.append(compute_metrics(Y_true, Y_pred_smv))
        wa_metrics.append(compute_metrics(Y_true, Y_pred_wa))
        for sid, yt, yp_smv, yp_wa in zip(sids, Y_true, Y_pred_smv, Y_pred_wa):
            pooled_preds.append({"sid": sid, "y_true": int(yt),
                                 "y_pred_smv": int(yp_smv), "y_pred_wa": int(yp_wa)})

    return smv_metrics, wa_metrics, pooled_preds


def run_svm(data, label_of, interview_folds, folds=5):
    fold_metrics, pooled_preds = [], []

    for k in range(1, folds + 1):
        train_speakers, test_speakers = fold_speakers(interview_folds, k)

        X_train = np.vstack([data[s].mean(axis=0) for s in train_speakers])
        Y_train = np.array([label_of[s] for s in train_speakers])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        clf = SVC(kernel="linear").fit(X_train, Y_train)

        X_test = scaler.transform(np.vstack([data[s].mean(axis=0) for s in test_speakers]))
        Y_pred = clf.predict(X_test)
        Y_true = [label_of[s] for s in test_speakers]

        fold_metrics.append(compute_metrics(Y_true, Y_pred))
        for sid, yt, yp in zip(test_speakers, Y_true, Y_pred):
            pooled_preds.append({"sid": sid, "y_true": int(yt), "y_pred": int(yp)})

    return fold_metrics, pooled_preds

RESULTS_DIR = "results"


def save_results(obj, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"Saved {path}")
    return path


def load_results(name):
    with open(os.path.join(RESULTS_DIR, f"{name}.json")) as f:
        return json.load(f)
