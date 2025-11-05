#!/usr/bin/env python3
"""Distributed Data Parallel fine‑tuning of LLaMA‑style models with LoRA.

This script uses Hugging Face Transformers together with the PEFT library to
attach LoRA adapters to a base model and fine‑tune it on a GLUE task.  It
initializes PyTorch distributed training via `torchrun`, wraps the model in
DistributedDataParallel and performs gradient accumulation.  After each epoch
it evaluates the model on the validation set (on rank 0 only), saves the
adapter weights and optionally uploads checkpoints to S3.

The script is intentionally self‑contained and does not rely on any
proprietary code.  It is meant as a realistic example for sharing your
distributed fine‑tuning experience with recruiters.
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

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model

from utils.glue_dataloader import load_glue, GLUE_TASK_TO_KEYS, GlueProcessed
from utils import metrics as glue_metrics
try:
    from utils import s3_backup
except Exception:
    s3_backup = None  # Optional dependency


def parse_args():
    parser = argparse.ArgumentParser(description="DDP LoRA fine‑tuning on GLUE")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--task_name", type=str, default="sst2",
                        help="GLUE task (sst2, mrpc, qqp, rte, mnli, qnli, cola, stsb)")
    parser.add_argument("--output_dir", type=str, default="./runs/llama_ddp_lora")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)

    # Optional S3 backup
    parser.add_argument("--s3_bucket", type=str, default=None, help="S3 bucket to upload checkpoints")
    parser.add_argument("--s3_prefix", type=str, default="", help="Prefix within the bucket")
    parser.add_argument("--s3_interval", type=int, default=0, help="Backup interval in seconds (0 disables)")

    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize torch.distributed from torchrun environment and return (rank, world_size, local_rank)."""
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
    """Evaluate the model on the validation set.  Only call on rank 0."""
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

    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    processed: GlueProcessed = load_glue(args.task_name, tokenizer, args.max_length)

    # Load model
    num_labels = get_num_labels(args.task_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16 if not args.fp16 else torch.float16,
    )

    # Setup LoRA
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=target_modules,
    )
    if is_main_process(rank):
        print(f"[LoRA] Attaching to modules: {target_modules}")
    model = get_peft_model(base_model, lora_config)

    # Move to device and wrap with DDP
    model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    # Create data loaders
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
    # Eval loader without sampler (only rank 0 will evaluate)
    eval_loader = DataLoader(
        processed.eval_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=processed.data_collator,
        pin_memory=True,
    )

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    num_training_steps = (len(train_loader) * args.epochs) // args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Optionally start S3 backup thread
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

    # Training loop
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
            # Save LoRA adapter weights only
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            # Save adapter config and weights
            ddp_model.module.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            # Save metrics
            with open(os.path.join(epoch_dir, "metrics.json"), "w") as f:
                import json
                json.dump(metrics, f, indent=2)
            print(f"[Checkpoint] Saved adapter and tokenizer to {epoch_dir}")
        dist.barrier()

    # Final cleanup
    if is_main_process(rank) and backup_thread is not None:
        print("[S3] Backup thread will continue running in the background")
    cleanup_distributed()


if __name__ == "__main__":
    main()
