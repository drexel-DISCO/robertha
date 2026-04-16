# ============================================================================
# RoBERTha: Robust Eigenspectrum Regularized Transformer Architecture
#            Using Iterative Hopfield Attention
#
# Authors:     Andreia Podasca and Anup Das
# Affiliation: Electrical and Computer Engineering
#              Drexel University
#              Philadelphia, PA
#
# License:     MIT License
#
# Copyright (c) 2026 Andreia Podasca and Anup Das
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ============================================================================

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json
import math
import random
from copy import deepcopy

# Import benchmark data loader
from benchmark_dataloader import (
    load_data,
    compute_metrics,
    TASK_CONFIGS
)

# Import noise injection utilities
from noise_utils import add_noise_to_embeddings


# ============================================================================
# AdvGLUE Helper Functions
# ============================================================================

def is_advglue_task(task_name):
    return task_name in TASK_CONFIGS and TASK_CONFIGS[task_name].get('benchmark') == 'advglue'

def get_base_task(task_name):
    if is_advglue_task(task_name):
        return TASK_CONFIGS[task_name].get('base_task', task_name)
    return task_name


# ============================================================================
# Set seed
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


# ============================================================================
# Model configuration (only tiny version is implemented)
# ============================================================================

MODEL_CONFIGS = {
    'tiny': {
        'hidden_size': 128,
        'num_hopfield_layers': 2,
        'intermediate_size': 512,
        'vocab_size': 30522,
        'max_position_embeddings': 512,
        'beta': 50.0,
        'max_iterations': 50,
        'convergence_threshold': 1e-4,
        'dropout': 0.3,
        'patience': 10
    }
}


def get_model_config(model_size='tiny'):
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_size '{model_size}'. Choose from: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_size]


# ============================================================================
# Regularization function
# ============================================================================

def eigenspectrum_regularization(keys, target_entropy_ratio=0.35):
    batch_size, seq_len, hidden_dim = keys.shape

    keys_centered = keys - keys.mean(dim=1, keepdim=True)
    cov = torch.matmul(keys_centered.transpose(-2, -1), keys_centered) / seq_len

    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.abs()

    eigenvalues_norm = eigenvalues / (eigenvalues.sum(dim=-1, keepdim=True) + 1e-8)
    eigenvalue_entropy = -(eigenvalues_norm * torch.log(eigenvalues_norm + 1e-10)).sum(dim=-1)

    max_entropy = torch.log(torch.tensor(hidden_dim, dtype=torch.float, device=keys.device))
    target_entropy = target_entropy_ratio * max_entropy

    loss = ((eigenvalue_entropy - target_entropy) ** 2).mean()

    return loss


# ============================================================================
# Hopfield Layer
# ============================================================================

class TrueHopfieldLayer(nn.Module):
    """Iterative Hopfield Network"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.beta = config['beta']
        self.max_iterations = config['max_iterations']
        self.convergence_threshold = config['convergence_threshold']

        self.W_query = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.W_key = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.W_value = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.W_out = nn.Linear(config['hidden_size'], config['hidden_size'])

        self.dropout = nn.Dropout(config['dropout'])

    def hopfield_update(self, query, keys):
        """Single Hopfield update step"""
        scores = torch.matmul(query, keys.transpose(-2, -1))
        scores = self.beta * scores / math.sqrt(self.hidden_size)
        attn_weights = F.softmax(scores, dim=-1)
        updated_query = torch.matmul(attn_weights, keys)
        return updated_query, attn_weights

    def forward(self, x, attention_mask=None):
        """Forward pass"""
        query = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        state = query.clone()
        iterations = 0
        converged = False

        for iteration in range(self.max_iterations):
            prev_state = state.clone()
            state, attn_weights = self.hopfield_update(state, keys)
            iterations = iteration + 1

            delta = torch.norm(state - prev_state) / (torch.norm(prev_state) + 1e-8)
            if delta < self.convergence_threshold:
                converged = True
                break

        # Compute output
        scores = torch.matmul(state, keys.transpose(-2, -1))
        scores = self.beta * scores / math.sqrt(self.hidden_size)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, values)
        output = self.W_out(output)
        output = self.dropout(output)

        return output, iterations, converged


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class FeedForward(nn.Module):
    """Standard FFN with GELU activation"""
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.linear2 = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class HopfieldPooling(nn.Module):
    """Mean pooling with attention mask support"""
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = summed / counts
        return pooled


class IterativeHopfieldModel(nn.Module):
    """Complete model with Hopfield layers for GLUE tasks"""

    def __init__(self, task_config, model_config):
        super().__init__()
        self.num_labels = task_config['num_labels']

        # Embeddings
        self.token_embeddings = nn.Embedding(
            model_config['vocab_size'],
            model_config['hidden_size'],
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            model_config['max_position_embeddings'],
            model_config['hidden_size']
        )

        self.layer_norm = nn.LayerNorm(model_config['hidden_size'])
        self.dropout = nn.Dropout(model_config['dropout'])

        # Hopfield layers
        self.hopfield_layers = nn.ModuleList([
            TrueHopfieldLayer(model_config)
            for _ in range(model_config['num_hopfield_layers'])
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(model_config['hidden_size'])
            for _ in range(model_config['num_hopfield_layers'])
        ])

        self.ffn_layers = nn.ModuleList([
            FeedForward(model_config)
            for _ in range(model_config['num_hopfield_layers'])
        ])

        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(model_config['hidden_size'])
            for _ in range(model_config['num_hopfield_layers'])
        ])

        # Pooling and classification
        self.hopfield_pool = HopfieldPooling()
        self.classifier = nn.Linear(model_config['hidden_size'], self.num_labels)

        # For storing last forward pass stats (used by regularization)
        self.last_layer_stats = []

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def get_embeddings(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape

        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)

        return embeddings

    def forward(self, input_ids=None, attention_mask=None, embeddings=None):
        if embeddings is not None:
            hidden_states = embeddings
        else:
            hidden_states = self.get_embeddings(input_ids, attention_mask)
            hidden_states = self.dropout(hidden_states)

        # Through Hopfield layers
        layer_iterations = []

        for hopfield, norm, ffn, ffn_norm in zip(
            self.hopfield_layers, self.layer_norms,
            self.ffn_layers, self.ffn_norms
        ):
            # Hopfield attention with residual
            residual = hidden_states
            hidden_states = norm(hidden_states)
            hopfield_out, iterations, converged = hopfield(hidden_states, attention_mask)
            layer_iterations.append((iterations, converged))
            hidden_states = residual + hopfield_out

            # FFN with residual
            residual = hidden_states
            hidden_states = ffn_norm(hidden_states)
            ffn_out = ffn(hidden_states)
            hidden_states = residual + ffn_out

        # Store for regularization
        self.last_layer_stats = layer_iterations

        # Pool and classify
        pooled = self.hopfield_pool(hidden_states, attention_mask)
        logits = self.classifier(pooled)

        return logits, layer_iterations

    def print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nModel Architecture:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Hopfield layers: {len(self.hopfield_layers)}")
        print(f"  Beta: {self.hopfield_layers[0].beta}")
        print(f"  Max iterations: {self.hopfield_layers[0].max_iterations}")
        print(f"  Output labels: {self.num_labels}")


# ============================================================================
# Training with regularization
# ============================================================================

def compute_regularization_loss(model, input_ids, attention_mask, reg_config):
    reg_types = reg_config['types']
    total_reg_loss = torch.tensor(0.0, device=input_ids.device)
    reg_breakdown = {}

    for reg_type in reg_types:
        reg_loss = torch.tensor(0.0, device=input_ids.device)

        if reg_type == 'esr':
            embeddings = model.get_embeddings(input_ids, attention_mask)
            current_hidden = embeddings

            for i, hopfield_layer in enumerate(model.hopfield_layers):
                normed_hidden = model.layer_norms[i](current_hidden)
                keys = hopfield_layer.W_key(normed_hidden)

                loss = eigenspectrum_regularization(
                    keys,
                    target_entropy_ratio=reg_config['esr_target_entropy_ratio']
                )
                reg_loss += loss

                with torch.no_grad():
                    hopfield_out, _, _ = hopfield_layer(normed_hidden, attention_mask)
                    current_hidden = current_hidden + hopfield_out
                    current_hidden = current_hidden + model.ffn_layers[i](model.ffn_norms[i](current_hidden))

            reg_loss *= reg_config['lambda_esr']
            reg_breakdown['esr'] = reg_loss.item()

        total_reg_loss += reg_loss

    return total_reg_loss, reg_breakdown


def train_model_with_regularization(model, train_loader, val_loader, task_name, model_size,
                                    epochs, device, model_dir, reg_config,
                                    model_config, collect_stats=False):
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    if TASK_CONFIGS[task_name]['num_labels'] == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    best_metric = -float('inf')
    patience_counter = 0

    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_task_loss = 0
        train_reg_loss = 0
        train_preds = []
        train_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)

            if TASK_CONFIGS[task_name]['num_labels'] == 1:
                task_loss = criterion(logits.squeeze(), labels)
                preds = logits.squeeze()
            else:
                task_loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=-1)

            reg_loss, _ = compute_regularization_loss(model, input_ids, attention_mask, reg_config)
            loss = task_loss + reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_task_loss += task_loss.item()
            train_reg_loss += reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({
                'loss': loss.item(),
                'task': task_loss.item(),
                'reg': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            })

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                logits, _ = model(input_ids, attention_mask)

                if TASK_CONFIGS[task_name]['num_labels'] == 1:
                    preds = logits.squeeze()
                else:
                    preds = torch.argmax(logits, dim=-1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        train_metadata = getattr(train_loader.dataset, 'metadata', None)
        val_metadata = getattr(val_loader.dataset, 'metadata', None)
        train_metrics = compute_metrics(task_name, train_preds, train_labels, metadata=train_metadata)
        val_metrics = compute_metrics(task_name, val_preds, val_labels, metadata=val_metadata)

        print(f"\nEpoch {epoch+1}")
        print(f"  Train loss: {train_loss/len(train_loader):.4f} (task: {train_task_loss/len(train_loader):.4f}, reg: {train_reg_loss/len(train_loader):.4f})")
        print(f"  Train metrics: {train_metrics}")
        print(f"  Val metrics: {val_metrics}")

        primary_metric_name = list(val_metrics.keys())[0]
        current_metric = val_metrics[primary_metric_name]

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0

            reg_str = '+'.join(reg_config['types']) if reg_config['types'] else 'none'
            model_size_dir = os.path.join(model_dir, model_size)
            os.makedirs(model_size_dir, exist_ok=True)
            model_path = os.path.join(model_size_dir, f'robertha_{task_name}_{reg_str}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'epoch': epoch,
                'reg_config': reg_config
            }, model_path)
            print(f"New best model saved! ({primary_metric_name}: {best_metric:.4f}) in path {model_path}")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{model_config['patience']})")

            if patience_counter >= model_config['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    return model


# ============================================================================
# Evaluation with noise
# ============================================================================

def evaluate_with_noise(model, val_loader, task_name, noise_levels, noise_type, device, seed):
    model = model.to(device)
    model.eval()

    num_layers = len(model.hopfield_layers)
    results = {}

    for noise_level in noise_levels:
        print(f"\n{'='*70}")
        print(f"Evaluating with noise level: {noise_level} ({noise_type})")
        print(f"{'='*70}")

        all_preds = []
        all_labels = []

        # Per-layer iteration tracking
        iterations_per_layer = [[] for _ in range(num_layers)]
        converged_per_layer = [[] for _ in range(num_layers)]

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Noise={noise_level}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # Get clean embeddings, inject noise, forward
                embeddings = model.get_embeddings(input_ids, attention_mask)
                noisy_embeddings = add_noise_to_embeddings(embeddings, noise_level, noise_type, seed)

                logits, layer_iters = model(
                    attention_mask=attention_mask,
                    embeddings=noisy_embeddings
                )

                # Record per-layer iterations
                for layer_idx, (iters, conv) in enumerate(layer_iters):
                    iterations_per_layer[layer_idx].append(iters)
                    converged_per_layer[layer_idx].append(conv)

                # Collect predictions
                if TASK_CONFIGS[task_name]['num_labels'] == 1:
                    preds = logits.squeeze()
                else:
                    preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute task metrics
        metadata = getattr(val_loader.dataset, 'metadata', None)
        metrics = compute_metrics(task_name, all_preds, all_labels, metadata=metadata, include_accuracy=True)

        # Aggregate iteration stats
        total_iterations = sum(sum(iters) for iters in iterations_per_layer)
        total_converged = sum(sum(c) for c in converged_per_layer)
        num_batches = len(iterations_per_layer[0])
        num_layer_calls = num_layers * num_batches

        metrics['avg_iterations'] = total_iterations / num_layer_calls if num_layer_calls > 0 else 0
        metrics['convergence_rate'] = total_converged / num_layer_calls if num_layer_calls > 0 else 0

        # Per-layer stats
        for layer_idx in range(num_layers):
            iters = iterations_per_layer[layer_idx]
            convs = converged_per_layer[layer_idx]
            prefix = f'layer{layer_idx}'
            metrics[f'{prefix}_iterations_mean'] = np.mean(iters)
            metrics[f'{prefix}_iterations_std'] = np.std(iters)
            metrics[f'{prefix}_iterations_min'] = int(np.min(iters))
            metrics[f'{prefix}_iterations_max'] = int(np.max(iters))
            metrics[f'{prefix}_iterations_median'] = float(np.median(iters))
            metrics[f'{prefix}_convergence_rate'] = np.mean(convs)

        # Raw per-layer per-batch data for downstream analysis
        metrics['_iterations_per_layer'] = {
            layer_idx: iters for layer_idx, iters in enumerate(iterations_per_layer)
        }

        results[noise_level] = metrics
        print(f"\n{task_name.upper()} Metrics: {metrics}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='RoBERTha with ESR Regularization')

    parser.add_argument('--model_size', type=str, default='tiny',
                       choices=['tiny', 'mobile', 'base'])
    parser.add_argument('--task', default='sst2', choices=list(TASK_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=None,
                       help='Batch size for evaluation. Defaults to --batch_size if not set.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', type=str, default='False')
    parser.add_argument('--load', type=str, default='False',
                       help='Load existing model before training (True/False). Use with --train True')
    parser.add_argument('--noise_type', default='absolute', choices=['absolute', 'percentage'])
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda, cuda:0, cuda:1, cpu, etc. (default: cuda)')
    parser.add_argument('--beta', type=float, default=15.0)
    parser.add_argument('--max_iterations', type=int, default=50)
    parser.add_argument('--max_train_samples', type=int, default=None)

    # Regularization args
    parser.add_argument('--regularization', default='none',
                       help='Regularization type(s): none/esr.')
    parser.add_argument('--esr_target_entropy_ratio', type=float, default=0.35,
                       help='Lower = more structured, Higher = more flexible')
    parser.add_argument('--target_entropy', type=float, default=2.8)
    parser.add_argument('--lambda_esr', type=float, default=0.05,
                       help='ESR regularization strength.')

    args = parser.parse_args()

    model_config = get_model_config(args.model_size)
    model_config['beta'] = args.beta
    model_config['max_iterations'] = args.max_iterations

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    # Set seed
    print(f"\n{'='*70}")
    print(f"Setting seed: {args.seed}")
    print(f"{'='*70}\n")
    seed_worker = set_seed(args.seed)

    # Setup device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Print configuration
    print("\n" + "="*70)
    print("RoBERTha configuration")
    print("="*70)
    print(f"Model Size: {args.model_size}")
    print(f"Hidden Size: {model_config['hidden_size']}")
    print(f"Num Layers: {model_config['num_hopfield_layers']}")
    print(f"FFN Size: {model_config['intermediate_size']}")
    print(f"Beta (Hopfield): {model_config['beta']}")
    print(f"Max Iterations: {model_config['max_iterations']}")
    print(f"Dropout: {model_config['dropout']}")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("="*70 + "\n")

    # Regularization config
    reg_types = args.regularization.split('+') if args.regularization != 'none' else []

    reg_config = {
        'types': reg_types,
        'target_entropy': args.target_entropy,
        'current_noise_level': 0.0,
        'esr_target_entropy_ratio': args.esr_target_entropy_ratio,
        'lambda_esr': args.lambda_esr,
    }

    print(f"Regularization Configuration:")
    print(f"  Types: {reg_types if reg_types else 'None'}")
    if 'esr' in reg_types:
        print(f"  ESR target entropy ratio: {args.esr_target_entropy_ratio} (normalized eigenvalue entropy)")

    # Load tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # AdvGLUE handling
    is_adv = is_advglue_task(args.task)
    base_task = get_base_task(args.task)
    eval_task = args.task

    if is_adv:
        print(f"\n{'='*70}")
        print(f"AdvGLUE Task: {args.task}")
        print(f"Base task for model loading: {base_task}")
        print(f"Evaluation dataset: {eval_task}")
        print(f"{'='*70}\n")

    # Load data
    print(f"Loading {eval_task} dataset...")
    train_dataset, val_dataset = load_data(eval_task, tokenizer, args.max_length, args.max_train_samples)

    g = torch.Generator()
    g.manual_seed(args.seed)

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        train_loader = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    if train_dataset is not None:
        print(f"Train samples: {len(train_dataset)}")
    else:
        print(f"Train samples: N/A (inference-only task)")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    task_config = TASK_CONFIGS[eval_task]
    model = IterativeHopfieldModel(task_config, model_config)
    model.print_model_info()

    # Train or load
    train_flag = args.train.lower() in ['true', '1', 'yes']
    load_flag = args.load.lower() in ['true', '1', 'yes']

    if is_adv and train_flag:
        print("\n" + "="*70)
        print("AdvGLUE is an inference-only benchmark.")
        print("="*70)
        return

    reg_str = '+'.join(reg_types) if reg_types else 'none'

    if load_flag and train_flag:
        print("\n" + "="*70)
        print("Loading existing model to continue training")
        print("="*70)

        model_task = base_task if is_adv else eval_task
        model_path = os.path.join(args.model_dir, args.model_size, f'robertha_{model_task}_{reg_str}.pt')

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
            print(f"Previous best metrics: {checkpoint.get('metrics', 'N/A')}")
            print(f"Continuing training from this checkpoint...")
        else:
            print(f"Model not found at {model_path}")
            print(f"Starting training from scratch instead.")
        print("="*70)

    if train_flag:
        print("\n" + "="*70)
        if load_flag:
            print(f"Continuing training with regularization: {reg_str.upper()}")
        else:
            print(f"Training model with regularization: {reg_str.upper()}")
        print("="*70)
        model = train_model_with_regularization(
            model, train_loader, val_loader, eval_task, args.model_size,
            args.epochs, device, args.model_dir, reg_config, model_config
        )

    # Load best model
    print("\n" + "="*70)
    print("Loading best model for evaluation")
    print("="*70)

    model_task = base_task if is_adv else eval_task
    model_path = os.path.join(args.model_dir, args.model_size, f'robertha_{model_task}_{reg_str}.pt')

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {model_path}")
        if is_adv:
            print(f"(Model trained on base task: {base_task})")
        print(f"Best validation metrics: {checkpoint.get('metrics', 'N/A')}")
    else:
        if not train_flag:
            print(f"No saved model found at {model_path}.")
            if is_adv:
                print(f"Please train on base task '{base_task}' first:")
                print(f"  python robertha.py --task {base_task} --train True")
            return

    # Evaluation
    print("\n" + "="*70)
    if is_adv:
        print(f"AdvGLUE Evaluation (no noise injection)")
    else:
        print("Noisy inference")
    print("="*70)

    noise_levels = [0.0] if is_adv else [0.0, 0.5, 1.0, 2.0, 5.0]

    results = evaluate_with_noise(model, val_loader, eval_task,
                                 noise_levels, args.noise_type, device, args.seed)

    # Save results
    model_size_dir = os.path.join(args.model_dir, args.model_size)
    os.makedirs(model_size_dir, exist_ok=True)
    results_file = os.path.join(model_size_dir, f'robertha_{eval_task}_{reg_str}.json')

    json_results = {}
    for k, metrics in results.items():
        serialized = {}
        for m, v in metrics.items():
            if m == '_iterations_per_layer':
                serialized[m] = {str(layer): iters for layer, iters in v.items()}
            elif isinstance(v, (int, np.integer)):
                serialized[m] = int(v)
            else:
                serialized[m] = float(v)
        json_results[str(k)] = serialized

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "="*70)
    if is_adv:
        print(f"AdvGLUE evaluation results: {eval_task}")
        print(f"(Using model trained on base task: {base_task})")
    print("="*70)

    # Identify task-specific metrics
    internal_keys = {'avg_iterations', 'convergence_rate'}
    task_metrics = [k for k in results[noise_levels[0]].keys()
                   if k not in internal_keys
                   and not k.startswith('layer')
                   and not k.startswith('_')]

    primary_metric = task_metrics[0]

    if is_adv:
        print(f"\nAdvGLUE task performance:")
    else:
        print(f"\nTask performance vs noise level ({args.noise_type}):")
    print("-" * 110)

    header = f"{'Noise':<10} "
    for metric in task_metrics:
        header += f"{metric.upper():<15} "
        header += f"{metric.upper()+'_DEG%':<15} "
    header += f"{'Avg Iters':<15} {'Conv Rate':<15}"
    print(header)
    print("-" * (10 + 30 * len(task_metrics) + 30))

    baseline_values = {metric: results[noise_levels[0]][metric] for metric in task_metrics}

    for noise_level in noise_levels:
        row = f"{noise_level:<10} "

        for metric in task_metrics:
            metric_value = results[noise_level][metric]
            row += f"{metric_value:<15.4f} "

            if noise_level == 0:
                degradation_str = "baseline"
            else:
                baseline = baseline_values[metric]
                if baseline != 0:
                    degradation = ((baseline - metric_value) / baseline) * 100
                    degradation_str = f"{degradation:+.2f}%"
                else:
                    degradation_str = "N/A"
            row += f"{degradation_str:<15} "

        avg_iters = results[noise_level].get('avg_iterations', 0)
        conv_rate = results[noise_level].get('convergence_rate', 0)
        row += f"{avg_iters:<15.2f} {conv_rate:<15.2%}"
        print(row)

    # Per-layer convergence summary
    num_layers = len(model.hopfield_layers)
    print(f"\n{'='*70}")
    print("Per-layer convergence iterations")
    print("="*70)
    header = f"{'Noise':<10} "
    for l in range(num_layers):
        header += f"{'L'+str(l)+' Mean':<12} {'L'+str(l)+' Std':<12} {'L'+str(l)+' Med':<12} {'L'+str(l)+' Min':<10} {'L'+str(l)+' Max':<10} {'L'+str(l)+' Conv%':<12} "
    print(header)
    print("-" * (10 + 68 * num_layers))

    for noise_level in noise_levels:
        row = f"{noise_level:<10} "
        for l in range(num_layers):
            prefix = f'layer{l}'
            m = results[noise_level]
            row += f"{m[f'{prefix}_iterations_mean']:<12.2f} "
            row += f"{m[f'{prefix}_iterations_std']:<12.2f} "
            row += f"{m[f'{prefix}_iterations_median']:<12.1f} "
            row += f"{m[f'{prefix}_iterations_min']:<10d} "
            row += f"{m[f'{prefix}_iterations_max']:<10d} "
            row += f"{m[f'{prefix}_convergence_rate']:<12.2%} "
        print(row)

if __name__ == '__main__':
    main()
