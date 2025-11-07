#!/usr/bin/env python3
"""Distributed Data Parallel fine‑tuning of LLaMA‑style models with AdaLoRA.

AdaLoRA extends LoRA by dynamically reallocating the rank budget across layers
during training.  This script is similar to `train_ddp_lora.py` but uses
`AdaLoraConfig` and calls `model.update_and_allocate()` at each step.  It
supports DDP training on GLUE tasks, evaluation, checkpointing and optional
S3 backup.
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import AdaLoraConfig, get_peft_model

from utils.glue_dataloader import load_glue, GLUE_TASK_TO_KEYS, GlueProcessed
from utils import metrics as glue_metrics
try:
    from utils import s3_backup
except Exception:
    s3_backup = None


def parse_args():
    parser = argparse.ArgumentParser(description="DDP AdaLoRA fine‑tuning on GLUE")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--output_dir", type=str, default="./runs/llama_ddp_adalora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--adalora_r", type=int, default=8)
    parser.add_argument("--adalora_target_rank", type=int, default=8)
    parser.add_argument("--adalora_init_r", type=int, default=12)
    parser.add_argument("--adalora_alpha", type=int, default=32)
    parser.add_argument("--adalora_dropout", type=float, default=0.05)
    parser.add_argument("--adalora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--s3_bucket", type=str, default=None)
    parser.add_argument("--s3_prefix", type=str, default="")
    parser.add_argument("--s3_interval", type=int, default=0)
    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script must be launched with torchrun")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int = 0) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)


def get_num_labels(task_name: str) -> int:
    if task_name == "stsb":
        return 1
    if task_name in ["sst2", "mrpc", "qqp", "rte", "qnli"]:
        return 2
    if task_name == "mnli":
        return 3
    if task_name == "cola":
        return 2
    raise ValueError(f"Unsupported task: {task_name}")


@torch.no_grad()
def evaluate(model: DDP, eval_loader: DataLoader, device: torch.device, task_name: str) -> Dict[str, float]:
    model.eval()
    preds = []
    labels = []
    for batch in eval_loader:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        outputs = model(**batch)
        logits = outputs.logits
        if task_name == "stsb":
            preds.extend(logits.view(-1).cpu().tolist())
            labels.extend(batch["labels"].view(-1).cpu().tolist())
        else:
            pred_labels = torch.argmax(logits, dim=-1)
            preds.extend(pred_labels.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    metrics = {}
    if task_name == "stsb":
        pearson, spearman = glue_metrics.stsb_corr(preds, labels)
        metrics["pearson"] = pearson
        metrics["spearman"] = spearman
    else:
        acc = glue_metrics.accuracy(preds, labels)
        metrics["accuracy"] = acc
    model.train()
    return metrics


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed, rank)
    device = torch.device("cuda", local_rank)
    if is_main_process(rank):
        print(f"[Init] rank={rank}, world_size={world_size}, local_rank={local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    processed: GlueProcessed = load_glue(args.task_name, tokenizer, args.max_length)
    num_labels = get_num_labels(args.task_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16 if not args.fp16 else torch.float16,
    )

    target_modules = [m.strip() for m in args.adalora_target_modules.split(",") if m.strip()]
    adalora_config = AdaLoraConfig(
        init_r=args.adalora_init_r,
        target_r=args.adalora_target_rank,
        beta1=0.85,
        beta2=0.85,
        tinit=100,
        tfinal=200,
        deltaT=10,
        lora_alpha=args.adalora_alpha,
        lora_dropout=args.adalora_dropout,
        target_modules=target_modules,
    )
    if is_main_process(rank):
        print(f"[AdaLoRA] Attaching to modules: {target_modules}")
    model = get_peft_model(base_model, adalora_config)
    model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    train_sampler = DistributedSampler(
        processed.train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        processed.train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_batch_size,
        collate_fn=processed.data_collator,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        processed.eval_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=processed.data_collator,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    num_training_steps = (len(train_loader) * args.epochs) // args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    backup_thread = None
    if args.s3_bucket and args.s3_interval > 0 and s3_backup is not None and is_main_process(rank):
        from threading import Thread

        def backup_loop():
            while True:
                s3_backup.upload_directory(args.output_dir, args.s3_bucket, args.s3_prefix)
                time.sleep(args.s3_interval)
        backup_thread = Thread(target=backup_loop, daemon=True)
        backup_thread.start()
        print(f"[S3] Backup thread started: bucket={args.s3_bucket}, prefix={args.s3_prefix}, interval={args.s3_interval}s")

    global_step = 0
    ddp_model.train()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if is_main_process(rank):
            print(f"\n[Train] Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        steps_in_epoch = 0
        for step, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = ddp_model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            # AdaLoRA: call update_and_allocate() each step to adjust ranks
            ddp_model.module.update_and_allocate(args.adalora_r)

            epoch_loss += loss.item()
            steps_in_epoch += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if is_main_process(rank) and global_step % args.log_every == 0:
                    avg_loss = (epoch_loss * args.gradient_accumulation_steps) / steps_in_epoch
                    lr = scheduler.get_last_lr()[0]
                    print(f"[Step {global_step}/{num_training_steps}] loss={avg_loss:.4f}, lr={lr:.6f}")

        # End of epoch: evaluate and save
        dist.barrier()
        if is_main_process(rank):
            metrics = evaluate(ddp_model, eval_loader, device, args.task_name)
            print(f"[Eval] Epoch {epoch+1}: {metrics}")
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            ddp_model.module.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            with open(os.path.join(epoch_dir, "metrics.json"), "w") as f:
                import json
                json.dump(metrics, f, indent=2)
            print(f"[Checkpoint] Saved adapter and tokenizer to {epoch_dir}")
        dist.barrier()

    if is_main_process(rank) and backup_thread is not None:
        print("[S3] Backup thread will continue running in the background")
    cleanup_distributed()


if __name__ == "__main__":
    main()
