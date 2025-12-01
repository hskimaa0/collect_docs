"""
KoBERT 기반 문서 타입 분류기 학습 스크립트

- 지원 데이터 포맷:
  - CSV:   id,text,label
  - JSONL: {"id": ..., "text": "...", "label": "..."}
  - JSON:  [{"id": ..., "text": "...", "label": "..."}, ...]

사용 예시:

python train.py \
  --train_file ./data/train.csv \
  --val_file ./data/val.csv \
  --output_dir ./saved_kobert_doc_classifier \
  --num_epochs 3 \
  --batch_size 16

val_file 를 생략하면 train_file 에서 자동으로 train/val split(9:1) 합니다.
"""

import os
import json
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import BertModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm


# ======================
# KoBERT 분류 모델 정의
# ======================

class KoBERTClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_classes)  # KoBERT hidden size = 768

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] 토큰의 hidden state 사용
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        x = self.dropout(cls_output)
        logits = self.classifier(x)  # (batch, num_classes)
        return logits

    def expand_classifier(self, new_num_classes: int):
        """새 라벨이 추가됐을 때 분류기 레이어 확장 (기존 가중치 유지)"""
        old_num_classes = self.classifier.out_features
        if new_num_classes <= old_num_classes:
            return  # 확장 필요 없음
        
        old_weight = self.classifier.weight.data  # (old_num_classes, 768)
        old_bias = self.classifier.bias.data      # (old_num_classes,)
        
        # 새 분류기 생성
        self.classifier = nn.Linear(768, new_num_classes)
        
        # 기존 가중치 복사
        self.classifier.weight.data[:old_num_classes] = old_weight
        self.classifier.bias.data[:old_num_classes] = old_bias
        
        print(f"[INFO] 분류기 확장: {old_num_classes} -> {new_num_classes} 클래스")


# =======================
# 데이터셋 / 로딩 유틸
# =======================

class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


def load_dataset(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
) -> Tuple[List[str], List[str]]:
    """
    주어진 파일에서 text, label 리스트를 읽어온다.
    CSV / JSON / JSONL 모두 지원.
    """
    ext = os.path.splitext(path)[1].lower()

    texts: List[str] = []
    labels: List[str] = []

    if ext == ".csv":
        df = pd.read_csv(path)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"CSV 파일에 '{text_col}', '{label_col}' 컬럼이 필요합니다.")
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(str).tolist()

    elif ext in [".json", ".jsonl"]:
        records = []
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        else:  # .json
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)

        for rec in records:
            if text_col not in rec or label_col not in rec:
                raise ValueError(f"JSON 레코드에 '{text_col}', '{label_col}' 필드가 필요합니다.")
            texts.append(str(rec[text_col]))
            labels.append(str(rec[label_col]))
    else:
        raise ValueError(f"지원하지 않는 파일 확장자: {ext}")

    return texts, labels


def build_label_mapping(labels: List[str]) -> Dict[str, int]:
    """
    문자열 라벨 리스트에서 label -> id 매핑 생성
    정렬된 순서로 id 부여 (재현성 위해)
    """
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    return label2id


# ================
# 학습 / 평가 루프
# ================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch=None,
    num_epochs=None,
):
    model.train()
    total_loss = 0.0

    desc = f"[Epoch {epoch}/{num_epochs}] Training" if epoch else "Training"
    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def evaluate(
    model,
    dataloader,
    device,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# ============
# main 함수
# ============

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="학습 데이터 파일 경로 (csv/json/jsonl)",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="검증 데이터 파일 경로 (csv/json/jsonl). 없으면 train_file에서 자동 split(9:1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="모델과 라벨 매핑을 저장할 디렉토리",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="토크나이저 max_length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="배치 크기",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="에폭 수",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="전체 step 대비 warmup step 비율",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="이전에 학습한 모델 디렉토리 (추가 학습 시 사용). 새 라벨이 추가되면 자동으로 분류기 확장",
    )

    return parser.parse_args()


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. 데이터 로딩
    train_texts, train_labels_str = load_dataset(args.train_file)

    if args.val_file is not None:
        val_texts, val_labels_str = load_dataset(args.val_file)
    else:
        # train_file 에서 자동으로 train/val 분리
        train_texts, val_texts, train_labels_str, val_labels_str = train_test_split(
            train_texts,
            train_labels_str,
            test_size=0.1,
            random_state=args.seed,
            stratify=train_labels_str,  # 라벨 비율 유지
        )

    # 2. 라벨 매핑 생성 (기존 모델이 있으면 기존 매핑 로드 후 확장)
    all_labels_str = list(train_labels_str) + list(val_labels_str)
    
    old_label2id = None
    if args.resume_from is not None:
        old_label_map_path = os.path.join(args.resume_from, "label_mapping.json")
        if os.path.exists(old_label_map_path):
            with open(old_label_map_path, "r", encoding="utf-8") as f:
                old_mapping = json.load(f)
                old_label2id = old_mapping["label2id"]
            print(f"[INFO] 기존 라벨 매핑 로드: {list(old_label2id.keys())}")
    
    if old_label2id is not None:
        # 기존 라벨 매핑 유지하면서 새 라벨만 추가
        label2id = dict(old_label2id)
        new_labels = []
        for lbl in sorted(set(all_labels_str)):
            if lbl not in label2id:
                new_id = len(label2id)
                label2id[lbl] = new_id
                new_labels.append(lbl)
        if new_labels:
            print(f"[INFO] 새로 추가된 라벨: {new_labels}")
    else:
        label2id = build_label_mapping(all_labels_str)
    
    id2label = {v: k for k, v in label2id.items()}

    print("[INFO] Label mapping (label -> id):")
    for k, v in label2id.items():
        print(f"  {k}: {v}")

    # 3. 문자열 라벨 -> 정수 라벨 변환
    train_labels = [label2id[lbl] for lbl in train_labels_str]
    val_labels = [label2id[lbl] for lbl in val_labels_str]

    # 4. 토크나이저 / 데이터셋 / 데이터로더
    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")

    train_dataset = TextDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = TextDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 5. 모델 / 옵티마이저 / 스케줄러
    num_classes = len(label2id)
    
    if args.resume_from is not None:
        # 기존 모델 로드
        old_model_path = os.path.join(args.resume_from, "best_model.pt")
        if os.path.exists(old_model_path):
            old_num_classes = len(old_label2id) if old_label2id else num_classes
            model = KoBERTClassifier(num_classes=old_num_classes)
            model.load_state_dict(torch.load(old_model_path, map_location=device))
            print(f"[INFO] 기존 모델 로드: {old_model_path}")
            
            # 새 라벨이 추가됐으면 분류기 확장
            if num_classes > old_num_classes:
                model.expand_classifier(num_classes)
        else:
            print(f"[경고] 기존 모델 파일을 찾을 수 없음: {old_model_path}")
            model = KoBERTClassifier(num_classes=num_classes)
    else:
        model = KoBERTClassifier(num_classes=num_classes)
    
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 6. 학습 루프
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.num_epochs} ==========")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            args.num_epochs,
        )
        print(f"[Train] loss: {train_loss:.4f}")

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
        )
        print(f"[Val]   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # 베스트 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model updated. Saved to {best_model_path}")

    print(f"\n[INFO] Training finished. Best val acc = {best_val_acc:.4f}")

    # 7. 최종 아티팩트 저장 (라벨 매핑, 설정 등)
    label_map_path = os.path.join(args.output_dir, "label_mapping.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Saved label mapping to {label_map_path}")

    # 토크나이저/컨피그도 같이 저장해두면 inference 때 편함
    tokenizer.save_pretrained(args.output_dir)
    # config.json 은 BertModel 쪽에서 자동 저장 가능
    # (원하면 model.config.to_json_file(...) 로 따로 저장도 가능)


if __name__ == "__main__":
    main()
