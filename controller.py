#!/usr/bin/env python3
"""
Brain Passage Retrieval Controller
Supports dual-encoder and cross-encoder architectures with multiple pooling strategies
Compatible with paper reproduction experiments
"""

import torch
import numpy as np
import random
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers import AutoTokenizer
import json
from datetime import datetime

# Import custom modules
from mv_dataloader import (DynamicMaskingDataloader, simple_collate_fn,
                           compute_combined_eeg_dimensions, load_combined_datasets)
from mv_models import create_model
from mv_training import train_model, test_model, finish_wandb, evaluate_expanded_candidate_index
import wandb


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set random seed to {seed}")


def save_experiment_config(config, output_dir):
    """Save experiment configuration to JSON file"""
    config_path = output_dir / "experiment_config.json"

    # Convert non-serializable values
    serializable_config = {
        k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v))
        for k, v in config.items()
    }

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    print(f"Saved experiment config to: {config_path}")


def handle_multiple_datasets(data_paths, max_eeg_len):
    """Handle loading and combining multiple datasets"""
    from mv_dataloader import load_combined_datasets, compute_combined_eeg_dimensions

    print(f"Loading {len(data_paths)} datasets for combination...")
    all_ict_pairs, combined_metadata = load_combined_datasets(data_paths, ['auto'] * len(data_paths))
    global_eeg_dims = compute_combined_eeg_dimensions(all_ict_pairs, max_eeg_len)

    print(f"Combined dataset ready: {len(all_ict_pairs)} total pairs")
    combined_dataset = {'ict_pairs': all_ict_pairs, 'metadata': combined_metadata}

    return combined_dataset, global_eeg_dims


def create_dataloaders(data_path, tokenizer, batch_size, max_text_len, max_eeg_len,
                       train_ratio, holdout_subjects, training_masking_level,
                       global_eeg_dims=None, combined_dataset=None,
                       pair_split_seed=42, balance_sources=False,
                       split_strategy='pair', train_sources=None):
    """Create training, validation, and test dataloaders"""

    # Load each source once. Reopening a very large .npy three times for train,
    # validation, and test retains multiple full copies and can OOM host RAM.
    if combined_dataset is None:
        if data_path is None:
            raise ValueError("data_path or combined_dataset must be provided")
        all_pairs, source_metadata = load_combined_datasets([data_path], ['auto'])
        combined_dataset = {'ict_pairs': all_pairs, 'metadata': source_metadata}
        data_path = None
        if global_eeg_dims is None:
            global_eeg_dims = compute_combined_eeg_dimensions(all_pairs, max_eeg_len)
    elif global_eeg_dims is None:
        global_eeg_dims = compute_combined_eeg_dimensions(combined_dataset['ict_pairs'], max_eeg_len)
    print(f"Computed shared EEG dimensions: {global_eeg_dims[0]}x{global_eeg_dims[1]}x{global_eeg_dims[2]}")

    training_masking_prob = training_masking_level / 100.0

    # Create datasets
    train_dataset = DynamicMaskingDataloader(
        data_path=None, combined_dataset=combined_dataset, tokenizer=tokenizer,
        max_text_len=max_text_len, max_eeg_len=max_eeg_len, split='train',
        train_ratio=train_ratio, global_eeg_dims=global_eeg_dims,
        dataset_type='original', holdout_subjects=holdout_subjects,
        initial_masking_probability=training_masking_prob, pair_split_seed=pair_split_seed,
        split_strategy=split_strategy
    )
    # Validation and test keep every loaded source so all ablation arms share them.
    if train_sources:
        allowed = set(train_sources)
        train_dataset.processed_pairs = [pair for pair in train_dataset.processed_pairs
                                         if pair.get('dataset_source') in allowed]
        if not train_dataset.processed_pairs:
            raise ValueError(f"No training pairs remain for train_sources={sorted(allowed)}")

    val_dataset = DynamicMaskingDataloader(
        data_path=None, combined_dataset=combined_dataset, tokenizer=tokenizer,
        max_text_len=max_text_len, max_eeg_len=max_eeg_len, split='val',
        train_ratio=train_ratio, global_eeg_dims=global_eeg_dims,
        dataset_type='original', holdout_subjects=holdout_subjects,
        initial_masking_probability=training_masking_prob, pair_split_seed=pair_split_seed,
        split_strategy=split_strategy
    )

    test_dataset = DynamicMaskingDataloader(
        data_path=None, combined_dataset=combined_dataset, tokenizer=tokenizer,
        max_text_len=max_text_len, max_eeg_len=max_eeg_len, split='test',
        train_ratio=train_ratio, global_eeg_dims=global_eeg_dims,
        dataset_type='original', holdout_subjects=holdout_subjects,
        initial_masking_probability=training_masking_prob, pair_split_seed=pair_split_seed,
        split_strategy=split_strategy
    )

    train_sampler = None
    train_sources = [pair.get('dataset_source', 'unknown') for pair in train_dataset.processed_pairs]
    train_groups = [pair.get('training_source', source)
                    for pair, source in zip(train_dataset.processed_pairs, train_sources)]
    source_counts = {source: train_sources.count(source) for source in set(train_sources)}
    group_sources = {group: {source for source, pair_group in zip(train_sources, train_groups)
                             if pair_group == group} for group in set(train_groups)}
    if balance_sources and (len(set(train_groups)) > 1 or len(set(train_sources)) > 1):
        weights = torch.tensor([
            1.0 / (len(group_sources) * len(group_sources[group]) * source_counts[source])
            for source, group in zip(train_sources, train_groups)
        ], dtype=torch.double)
        generator = torch.Generator().manual_seed(pair_split_seed)
        train_sampler = WeightedRandomSampler(weights, num_samples=len(train_sources), replacement=True,
                                              generator=generator)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler,
        collate_fn=simple_collate_fn, num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=simple_collate_fn, num_workers=0
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=simple_collate_fn, num_workers=0
    )

    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples from {len(train_dataset.unique_subjects)} subjects")
    print(f"  Val: {len(val_dataset)} samples from {len(val_dataset.unique_subjects)} subjects")
    print(f"  Test: {len(test_dataset)} samples from {len(test_dataset.unique_subjects)} subjects")

    return train_dataloader, val_dataloader, test_dataloader, global_eeg_dims


class MaskingAwareSubset(Subset):
    """Subset view that forwards the dynamic masking controls used by test_model."""

    @property
    def unique_subjects(self):
        base = self.dataset
        return {base.processed_pairs[index].get('participant_id', 'unknown') for index in self.indices}

    def get_current_masking_probability(self):
        return self.dataset.get_current_masking_probability()

    def set_masking_probability(self, probability):
        self.dataset.set_masking_probability(probability)


def create_source_test_loaders(test_dataloader):
    """Return test loaders with a separate candidate pool for each source."""
    dataset = test_dataloader.dataset
    local_indices_by_source = {}
    for local_index, pair in enumerate(dataset.processed_pairs):
        source = pair.get('dataset_source', 'unknown')
        local_indices_by_source.setdefault(source, []).append(local_index)
    return {
        source: DataLoader(
            MaskingAwareSubset(dataset, local_indices), batch_size=test_dataloader.batch_size,
            shuffle=False, collate_fn=simple_collate_fn, num_workers=0
        )
        for source, local_indices in sorted(local_indices_by_source.items())
        if local_indices
    }


def create_subset_loader(dataloader, keep):
    dataset = dataloader.dataset
    local_indices = [index for index, pair in enumerate(dataset.processed_pairs) if keep(pair)]
    if not local_indices:
        return None
    return DataLoader(MaskingAwareSubset(dataset, local_indices), batch_size=dataloader.batch_size,
                      shuffle=False, collate_fn=simple_collate_fn, num_workers=0)


def select_primary_validation_loader(val_dataloader, primary_source=None):
    if primary_source is None:
        return val_dataloader
    loader = create_subset_loader(
        val_dataloader,
        lambda pair: primary_source in (pair.get('dataset_source'), pair.get('training_source')))
    if loader is None:
        raise ValueError(f"Primary validation source {primary_source!r} is absent from validation data")
    return loader


def save_test_metrics(path, results):
    """Write source-specific test metrics in JSON-friendly form."""
    path.write_text(json.dumps(results, indent=2, default=float) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='Brain Passage Retrieval - Paper Reproduction')

    # Data arguments
    parser.add_argument('--data_path', type=str, help='Path to single ICT pairs .npy file')
    parser.add_argument('--data_paths', nargs='+', help='Paths to multiple ICT pairs .npy files')

    # Model arguments
    parser.add_argument('--pooling_strategy', type=str, default='multi',
                        choices=['cls', 'mean', 'max', 'multi'],
                        help='Semantic aggregation strategy')
    parser.add_argument('--encoder_type', type=str, default='dual',
                        choices=['dual', 'cross'],
                        help='Encoder architecture type')
    parser.add_argument('--query_type', type=str, default='eeg',
                        choices=['eeg', 'text'],
                        help='Query representation type')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    # Data split arguments
    parser.add_argument('--holdout_subjects', action='store_true',
                        help='Use subject-based train/val/test split')
    parser.add_argument('--balance_sources', action='store_true',
                        help='Sample mixed-source training pairs with equal source probability')
    parser.add_argument('--split-strategy', choices=('pair', 'target', 'trial', 'original'), default='pair',
                        help='Pair-level, target-grouped, Task trial-grouped, or original Audio BPR random split')
    parser.add_argument('--train-sources', nargs='+', default=None,
                        help='dataset_source values used for training; val/test keep all loaded sources')
    parser.add_argument('--multi-positive-loss', action='store_true',
                        help='Treat repeated in-batch examples of the same target document as positives')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient-norm clipping threshold')
    parser.add_argument('--min-epochs', type=int, default=0,
                        help='Do not early-stop before this many epochs')

    # Masking arguments
    parser.add_argument('--training_masking_level', type=int, default=90,
                        help='Masking probability during training (0-100)')
    parser.add_argument('--enable_multi_masking_validation', action='store_true',
                        help='Enable validation across multiple masking levels')
    parser.add_argument('--validation_masking_levels', nargs='+', type=int,
                        default=[0, 25, 50, 75, 90, 100],
                        help='Masking levels to evaluate during validation')

    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Named ablation condition used in W&B metadata')
    parser.add_argument('--ablation-sources', nargs='+', default=None,
                        help='Named data sources included in this ablation')
    parser.add_argument('--wandb-project', type=str, default='ECIR2026',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Explicit Weights & Biases run name')
    parser.add_argument('--primary-eval-source', type=str, default=None,
                        help='Source used for validation and early stopping in ablation runs')
    parser.add_argument('--expanded-eval-index', type=str, default=None,
                        help='JSON index of full questions and displayed responses for expanded retrieval metrics')

    # Additional arguments (with defaults for paper reproduction)
    parser.add_argument('--max_text_len', type=int, default=256, help='Max text sequence length')
    parser.add_argument('--max_eeg_len', type=int, default=50, help='Max EEG sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension size')
    parser.add_argument('--num_vectors', type=int, default=32, help='Number of vectors for multi pooling')
    parser.add_argument('--eeg_arch', type=str, default='simple',
                        choices=['simple', 'complex', 'transformer'],
                        help='EEG encoder architecture')
    parser.add_argument('--colbert_model_name', type=str, default='colbert-ir/colbertv2.0',
                        help='ColBERT model name')
    parser.add_argument('--no_lora', action='store_true', help='Disable LoRA adaptation')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--use_pretrained_text', action='store_true',
                        help='Use pretrained ColBERT for text encoding')
    parser.add_argument('--multi_masking_frequency', type=int, default=3,
                        help='Run multi-masking validation every N epochs')
    parser.add_argument('--primary_masking_level', type=int, default=90,
                        help='Primary masking level for early stopping')

    args = parser.parse_args()

    # Validate inputs
    if not args.data_path and not args.data_paths:
        raise ValueError("Must specify either --data_path or --data_paths")

    if args.data_path and args.data_paths:
        raise ValueError("Cannot specify both --data_path and --data_paths")

    # Set random seeds
    set_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.colbert_model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.colbert_model_name)
    except:
        print(f"ColBERT tokenizer not found, using bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Add special tokens
    special_tokens = ['[Q]', '[D]', '[MASK]'] if '[MASK]' not in tokenizer.get_vocab() else ['[Q]', '[D]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Handle dataset(s)
    combined_dataset = None
    global_eeg_dims = None
    data_path = None

    if args.data_paths and len(args.data_paths) > 1:
        # Multiple datasets
        print(f"\n=== LOADING MULTIPLE DATASETS ===")
        combined_dataset, global_eeg_dims = handle_multiple_datasets(
            args.data_paths, args.max_eeg_len
        )
        dataset_name = "combined"
    else:
        # Single dataset
        data_path = args.data_path or args.data_paths[0]
        print(f"\n=== LOADING DATASET ===")
        print(f"Data path: {data_path}")

        # Determine dataset name from filename
        filename = Path(data_path).name.lower()
        if 'nieuwland' in filename:
            dataset_name = "nieuwland"
        elif 'alice' in filename:
            dataset_name = "alice"
        else:
            dataset_name = Path(data_path).stem

    # Create dataloaders
    print(f"\n=== CREATING DATALOADERS ===")
    train_dataloader, val_dataloader, test_dataloader, global_eeg_dims = create_dataloaders(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
        max_eeg_len=args.max_eeg_len,
        train_ratio=args.train_ratio,
        holdout_subjects=args.holdout_subjects,
        training_masking_level=args.training_masking_level,
        global_eeg_dims=global_eeg_dims,
        combined_dataset=combined_dataset,
        pair_split_seed=args.seed,
        balance_sources=args.balance_sources,
        split_strategy=args.split_strategy,
        train_sources=args.train_sources,
    )

    training_val_dataloader = val_dataloader
    if args.primary_eval_source:
        training_val_dataloader = select_primary_validation_loader(val_dataloader, args.primary_eval_source)
        print(f"Early stopping and checkpoint selection use {args.primary_eval_source} only")
    expanded_eval_index = None
    if args.expanded_eval_index:
        with Path(args.expanded_eval_index).open(encoding='utf-8') as stream:
            expanded_eval_index = json.load(stream)
        if not expanded_eval_index.get('candidate_documents'):
            raise ValueError("Expanded evaluation index contains no candidate documents")

    # Create experiment configuration
    config = {
        'experiment_type': 'brain_passage_retrieval',
        'experiment_name': args.experiment_name,
        'ablation_sources': args.ablation_sources or [],
        'wandb_project': args.wandb_project,
        'wandb_run_name': args.wandb_run_name,
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'dataset_name': dataset_name,
        'pooling_strategy': args.pooling_strategy,
        'encoder_type': args.encoder_type,
        'query_type': args.query_type,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'patience': args.patience,
        'holdout_subjects': args.holdout_subjects,
        'pair_level_split': not args.holdout_subjects,
        'pair_split_seed': args.seed,
        'split_strategy': args.split_strategy,
        'balance_sources': args.balance_sources,
        'multi_positive_loss': args.multi_positive_loss,
        'grad_clip': args.grad_clip,
        'min_epochs': args.min_epochs,
        'max_eeg_len': args.max_eeg_len,
        'query_group_multi_positive': args.multi_positive_loss,
        'primary_task_source': 'authentic_query_response' if args.experiment_name else None,
        'primary_eval_source': args.primary_eval_source,
        'train_sources': args.train_sources or [],
        'expanded_eval_index_path': args.expanded_eval_index,
        'expanded_eval_candidate_count': (len(expanded_eval_index['candidate_documents'])
                          if expanded_eval_index else 0),
        'expanded_eval_policy': (
            "Full questions and displayed responses form one fixed index; either matching text is relevant."
            if expanded_eval_index else None
        ),
        'training_masking_level': args.training_masking_level,
        'enable_multi_masking_validation': args.enable_multi_masking_validation,
        'validation_masking_levels': args.validation_masking_levels,
        'seed': args.seed,
        'device': str(device),
        'train_samples': len(train_dataloader.dataset),
        'val_samples': len(training_val_dataloader.dataset),
        'all_source_val_samples': len(val_dataloader.dataset),
        'test_samples': len(test_dataloader.dataset),
        'eeg_arch': args.eeg_arch,
        'data_files': [Path(path).name for path in (args.data_paths or [args.data_path])],
        'hidden_dim': args.hidden_dim,
        'num_vectors': args.num_vectors,
        'colbert_model_name': args.colbert_model_name,
        'use_pretrained_text': args.use_pretrained_text,
        'use_lora': not args.no_lora,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
    }

    print(f"\n=== EXPERIMENT CONFIGURATION ===")
    print(f"Dataset: {dataset_name}")
    print(f"Pooling: {args.pooling_strategy}")
    print(f"Encoder: {args.encoder_type}")
    print(f"Query type: {args.query_type}")
    print(f"Training masking: {args.training_masking_level}%")
    print(f"Holdout subjects: {args.holdout_subjects}")
    if args.enable_multi_masking_validation:
        print(f"Multi-masking validation: {args.validation_masking_levels}%")

    # Create model
    print(f"\n=== MODEL CREATION ===")
    model = create_model(
        colbert_model_name=args.colbert_model_name,
        hidden_dim=args.hidden_dim,
        eeg_arch=args.eeg_arch,
        device=device,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        pooling_strategy=args.pooling_strategy,
        encoder_type=args.encoder_type,
        global_eeg_dims=global_eeg_dims,
        query_type=args.query_type,
        use_pretrained_text=args.use_pretrained_text
    )

    if not args.use_pretrained_text:
        model.set_tokenizer_vocab_size(len(tokenizer))
    model.multi_positive_loss = args.multi_positive_loss
    model.grad_clip_norm = args.grad_clip

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    config.update({'total_params': total_params, 'trainable_params': trainable_params})
    save_experiment_config(config, output_dir)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train model
    print(f"\n=== TRAINING ===")
    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=training_val_dataloader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        patience=args.patience,
        device=device,
        config=config,
        enable_multi_masking_validation=args.enable_multi_masking_validation,
        multi_masking_frequency=args.multi_masking_frequency,
        validation_masking_levels=args.validation_masking_levels,
        primary_masking_level=args.primary_masking_level,
        expanded_eval_index=expanded_eval_index,
    )

    # Save model with paper-compatible naming
    model_filename = f"model_{args.pooling_strategy}_{args.encoder_type}.pt"
    model_save_path = output_dir / model_filename
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'tokenizer_vocab_size': len(tokenizer)
    }, model_save_path)
    print(f"\nSaved trained model to: {model_save_path}")

    # Test evaluation
    if len(test_dataloader.dataset) > 0:
        print(f"\n=== TEST EVALUATION (original Audio BPR protocol) ===")
        results_dir = output_dir / 'test_by_source'
        results_dir.mkdir(exist_ok=True)
        test_results = test_model(
            model=trained_model,
            test_dataloader=test_dataloader,
            device=device,
            test_masking_levels=args.validation_masking_levels,
            primary_masking_level=args.primary_masking_level,
        )
        save_test_metrics(results_dir / 'all_sources_metrics.json', test_results)
        for source, source_loader in create_source_test_loaders(test_dataloader).items():
            print(f"Testing {source} queries against the {source} candidate pool")
            source_results = test_model(
                model=trained_model,
                test_dataloader=source_loader,
                device=device,
                test_masking_levels=args.validation_masking_levels,
                primary_masking_level=args.primary_masking_level,
                wandb_namespace=f'test_by_source/{source}',
            )
            save_test_metrics(results_dir / f'{source}_metrics.json', source_results)

        task_test_loader = create_subset_loader(
            test_dataloader, lambda pair: pair.get('training_source') == 'authentic_query')
        if expanded_eval_index is not None and task_test_loader is not None:
            print("Testing all Task queries together against the shared question+response index")
            combined_results = evaluate_expanded_candidate_index(
                trained_model, task_test_loader, device, expanded_eval_index,
                prefix='test_combined_task/index', masking_level=args.primary_masking_level)
            combined_results.update(evaluate_expanded_candidate_index(
                trained_model, task_test_loader, device, expanded_eval_index,
                prefix='test_combined_task/noise', masking_level=args.primary_masking_level,
                use_noise=True))
            wandb.log(combined_results)
            save_test_metrics(results_dir / 'combined_task_index_metrics.json', combined_results)
        print(f"Test evaluation completed; source metrics are in {results_dir}")

    finish_wandb()
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()