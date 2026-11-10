#!/usr/bin/env python3
"""
vram-guard: Production-grade VRAM profiling, hardware auditing, and 
memory-bounded training/inference configuration injection.
"""

import argparse
import sys
import json
import math
from typing import Dict, Any, List, Optional

try:
    import torch
except ImportError:
    print("Error: PyTorch is required for CUDA hardware detection. Install via torch.org", file=sys.stderr)
    sys.exit(1)


class HardwareProfiler:
    @staticmethod
    def detect_devices() -> List[Dict[str, Any]]:
        devices = []
        if not torch.cuda.is_available():
            return devices

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_vram_gb = props.total_memory / (1024 ** 3)
            
            # Free VRAM estimation via torch memory allocator
            torch.cuda.set_device(i)
            free_memory, total_memory_allocated = torch.cuda.mem_get_info(i)
            free_vram_gb = free_memory / (1024 ** 3)

            devices.append({
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_gb": round(total_vram_gb, 2),
                "free_vram_gb": round(free_vram_gb, 2),
                "multi_processor_count": props.multi_processor_count
            })
        return devices


class MemoryCalculator:
    @staticmethod
    def estimate_memory(
        model_size_b: float,
        quantization: str,
        seq_len: int,
        batch_size: int,
        mode: str = "train",
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculates memory consumption in GB for weights, gradients, optimizer states, and activations.
        """
        # Default architectural heuristics if not specified (approximate standard transformer bounds)
        hidden_dim = hidden_dim or int(4096 * (model_size_b / 7.0) ** 0.5)
        num_layers = num_layers or int(32 * (model_size_b / 7.0) ** 0.33)

        # 1. Model Weights Memory
        bits_per_param = {
            "fp16": 2.0,
            "bf16": 2.0,
            "int8": 1.0,
            "int4": 0.5,
            "qlora": 0.5  # base model in 4-bit, adapters in fp16/bf16
        }.get(quantization.lower(), 2.0)

        weights_gb = (model_size_b * 1e9 * bits_per_param) / (1024 ** 3)
        
        if quantization.lower() == "qlora":
            # Add overhead for unfrozen LoRA adapters (~0.5% to 2% of parameters in FP16)
            weights_gb += (model_size_b * 1e9 * 0.02 * 2.0) / (1024 ** 3)

        # 2. Gradients Memory (Only for training)
        gradients_gb = 0.0
        if mode == "train":
            if quantization.lower() == "qlora":
                # LoRA gradients only for adapter parameters
                gradients_gb = (model_size_b * 1e9 * 0.02 * 2.0) / (1024 ** 3)
            else:
                grad_bits = 2.0 if quantization in ["fp16", "bf16"] else 4.0
                gradients_gb = (model_size_b * 1e9 * grad_bits) / (1024 ** 3)

        # 3. Optimizer States Memory (Only for training)
        optimizer_gb = 0.0
        if mode == "train":
            if quantization.lower() == "qlora":
                # 8-bit AdamW for LoRA parameters
                optimizer_gb = (model_size_b * 1e9 * 0.02 * 2.0) / (1024 ** 3)
            else:
                opt_multiplier = {
                    "adamw": 16.0,      # FP32 master weights (4) + Momentum (4) + Variance (4) + FP16 weights/grads (4)
                    "adamw_8bit": 6.0,  # 8-bit quantization states
                    "sgd": 8.0
                }["adamw_8bit" if "8bit" in quantization else "adamw"]
                optimizer_gb = (model_size_b * 1e9 * opt_multiplier / 16.0) / (1024 ** 3)

        # 4. Activations Memory (Batch size * Seq Len * Layers * Hidden Dim scaling heuristic)
        # Activation memory per token per layer ~ 2-4 bytes * hidden_dim
        activation_bytes_per_token = 4.0
        if mode == "train":
            activation_bytes_per_token = 12.0  # intermediate saves for backprop without full checkpointing
            
        activations_gb = (batch_size * seq_len * num_layers * hidden_dim * activation_bytes_per_token) / (1024 ** 3)

        # Safety & CUDA Overhead Buffer (~15% baseline fragmentation buffer)
        overhead_gb = 1.5 + (weights_gb * 0.1)

        total_estimated_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb + overhead_gb

        return {
            "weights_gb": round(weights_gb, 3),
            "gradients_gb": round(gradients_gb, 3),
            "optimizer_gb": round(optimizer_gb, 3),
            "activations_gb": round(activations_gb, 3),
            "overhead_gb": round(overhead_gb, 3),
            "total_estimated_gb": round(total_estimated_gb, 3)
        }


class FallbackEngine:
    @staticmethod
    def compute_safe_fallbacks(
        current_args: Dict[str, Any],
        available_vram: float
    ) -> Dict[str, Any]:
        """
        Iteratively adjusts sequence length, quantization, and batch size to fit within available VRAM.
        """
        adjusted = current_args.copy()
        
        # Iteration 1: Enforce QLoRA if standard FP16/INT8 overflows
        if adjusted["quantization"].lower() not in ["int4", "qlora"]:
            test_mem = MemoryCalculator.estimate_memory(
                adjusted["model_size_b"], "qlora", adjusted["seq_len"], adjusted["batch_size"], adjusted["mode"]
            )
            if test_mem["total_estimated_gb"] <= available_vram:
                adjusted["quantization"] = "qlora"
                adjusted["note"] = "Switched to QLoRA to fit VRAM ceiling."
                return adjusted

        # Iteration 2: Reduce batch size down to 1
        while adjusted["batch_size"] > 1:
            adjusted["batch_size"] -= 1
            test_mem = MemoryCalculator.estimate_memory(
                adjusted["model_size_b"], adjusted["quantization"], adjusted["seq_len"], adjusted["batch_size"], adjusted["mode"]
            )
            if test_mem["total_estimated_gb"] <= available_vram:
                adjusted["note"] = f"Reduced batch_size to {adjusted['batch_size']} to fit VRAM ceiling."
                return adjusted

        # Iteration 3: Step-down sequence length scaling
        while adjusted["seq_len"] > 512:
            adjusted["seq_len"] = int(adjusted["seq_len"] * 0.75)
            test_mem = MemoryCalculator.estimate_memory(
                adjusted["model_size_b"], adjusted["quantization"], adjusted["seq_len"], adjusted["batch_size"], adjusted["mode"]
            )
            if test_mem["total_estimated_gb"] <= available_vram:
                adjusted["note"] = f"Scaled max_seq_len down to {adjusted['seq_len']} to prevent OOM."
                return adjusted

        adjusted["note"] = "Critical: Hardware VRAM is insufficient even under maximum structural compression."
        return adjusted


def generate_training_arguments_snippet(config: Dict[str, Any]) -> str:
    """Generates a plug-and-play Python dictionary for Hugging Face TrainingArguments / PEFT."""
    use_qlora = config["quantization"].lower() in ["int4", "qlora"]
    
    snippet = f"""# --- Hugging Face PEFT / Transformers Auto-Injected Configuration ---
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

training_args = TrainingArguments(
    output_dir="./vram-guard-output",
    per_device_train_batch_size={config['batch_size']},
    gradient_accumulation_steps={max(1, 4 // config['batch_size'])},
    learning_rate=2e-4,
    fp16={config['quantization'].lower() == 'fp16'},
    bf16={config['quantization'].lower() == 'bf16'},
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit" if {str(use_qlora)} else "adamw_torch",
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
) if {str(use_qlora)} else None
"""
    return snippet


def main():
    parser = argparse.ArgumentParser(description="vram-guard: CUDA VRAM Profiling & Memory-Bounding Engine")
    parser.add_argument("--model-size", type=float, required=True, help="Model parameter size in billions (e.g., 1.5, 3.0, 7.0)")
    parser.add_argument("--target-quantization", type=str, default="fp16", choices=["fp16", "bf16", "int8", "int4", "qlora"], help="Quantization tier")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Maximum sequence length context window")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size per device")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"], help="Execution mode")
    parser.add_argument("--device-id", type=int, default=0, help="Target CUDA device index")
    parser.add_argument("--export-format", type=str, default="json", choices=["json", "yaml", "dict"], help="Output config format")

    args = parser.parse_args()

    # 1. Hardware Detection
    devices = HardwareProfiler.detect_devices()
    if not devices:
        print(f"[{'ERROR'}] No CUDA devices detected via PyTorch.", file=sys.stderr)
        sys.exit(2)

    if args.device_id >= len(devices):
        print(f"[{'ERROR'}] Device ID {args.device_id} out of range. Found {len(devices)} device(s).", file=sys.stderr)
        sys.exit(2)

    selected_device = devices[args.device_id]
    available_vram = selected_device["free_vram_gb"]

    # 2. Memory Estimation
    memory_breakdown = MemoryCalculator.estimate_memory(
        model_size_b=args.model_size,
        quantization=args.target_quantization,
        seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        mode=args.mode
    )

    estimated_total = memory_breakdown["total_estimated_gb"]
    is_feasible = estimated_total <= available_vram

    # 3. Fallback Handling if OOM Risk Detected
    final_config = {
        "model_size_b": args.model_size,
        "quantization": args.target_quantization,
        "seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "mode": args.mode,
        "note": "Configuration fits within hardware VRAM boundaries."
    }

    if not is_feasible:
        final_config = FallbackEngine.compute_safe_fallbacks(final_config, available_vram)
        # Recalculate with fallback parameters
        memory_breakdown = MemoryCalculator.estimate_memory(
            model_size_b=final_config["model_size_b"],
            quantization=final_config["quantization"],
            seq_len=final_config["seq_len"],
            batch_size=final_config["batch_size"],
            mode=final_config["mode"]
        )
        is_feasible = memory_breakdown["total_estimated_gb"] <= available_vram

    # 4. Construct Output Report
    report = {
        "status": "PASS" if is_feasible else "FAIL",
        "hardware": selected_device,
        "memory_breakdown_gb": memory_breakdown,
        "configuration": final_config,
        "huggingface_snippet": generate_training_arguments_snippet(final_config)
    }

    # 5. Serialization & Logging
    if args.export_format == "json":
        print(json.dumps(report, indent=2))
    elif args.export_format == "dict":
        print(repr(report))
    else:
        # Simple YAML-like text printout
        print(f"=== VRAM-GUARD AUDIT REPORT ===")
        print(f"Status: {report['status']}")
        print(f"Device: {selected_device['name']} ({selected_device['free_vram_gb']} GB Free / {selected_device['total_vram_gb']} GB Total)")
        print(f"Estimated Usage: {memory_breakdown['total_estimated_gb']} GB")
        print(f"Config: {final_config}")
        print(f"\n--- Python Snippet ---\n{report['huggingface_snippet']}")

    sys.exit(0 if is_feasible else 1)


if __name__ == "__main__":
    main()
