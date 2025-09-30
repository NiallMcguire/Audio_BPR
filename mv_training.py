#!/usr/bin/env python3

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mv_models import compute_similarity
from mv_dataloader import create_positive_negative_pairs


def compute_bce_loss(scores, labels):
    return F.binary_cross_entropy_with_logits(scores.squeeze(), labels.float())


def cross_encoder_train_step(model, batch, optimizer, device, step_num, debug=False):
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}

    positive_labels, negative_pairs, negative_labels = create_positive_negative_pairs(batch)
    positive_labels = positive_labels.to(device)
    negative_labels = negative_labels.to(device)

    if debug:
        query_type = model.query_type
        print(f"[DEBUG] Cross-encoder training step {step_num} (query_type: {query_type})")
        print(f"  Positive pairs: {len(positive_labels)}")
        print(f"  Negative pairs: {len(negative_labels)}")

    positive_scores = model(eeg_queries, text_queries, docs)
    positive_loss = compute_bce_loss(positive_scores, positive_labels)

    if len(negative_pairs) > 0:
        neg_eeg = torch.stack([eeg_queries[pair['eeg_idx']] for pair in negative_pairs])
        neg_text_queries = {
            'input_ids': torch.stack([text_queries['input_ids'][pair['eeg_idx']] for pair in negative_pairs]),
            'attention_mask': torch.stack([text_queries['attention_mask'][pair['eeg_idx']] for pair in negative_pairs])
        }
        neg_docs = {
            'input_ids': torch.stack([docs['input_ids'][pair['doc_idx']] for pair in negative_pairs]),
            'attention_mask': torch.stack([docs['attention_mask'][pair['doc_idx']] for pair in negative_pairs])
        }

        negative_scores = model(neg_eeg, neg_text_queries, neg_docs)
        negative_loss = compute_bce_loss(negative_scores, negative_labels)
        total_loss = positive_loss + negative_loss
        neg_acc = ((torch.sigmoid(negative_scores.squeeze()) > 0.5) == negative_labels.to(device)).float().mean()
    else:
        total_loss = positive_loss
        neg_acc = torch.tensor(0.0)

    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    pos_acc = ((torch.sigmoid(positive_scores.squeeze()) > 0.5) == positive_labels).float().mean()

    if debug:
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Positive accuracy: {pos_acc.item():.4f}")
        print(f"  Negative accuracy: {neg_acc.item() if torch.is_tensor(neg_acc) else neg_acc:.4f}")

    return total_loss.item(), {'accuracy': pos_acc.item(), 'negative_accuracy': neg_acc.item() if torch.is_tensor(neg_acc) else neg_acc}, grad_norm.item()


def cross_encoder_validation_step(model, batch, device):
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}

    positive_labels, negative_pairs, negative_labels = create_positive_negative_pairs(batch)
    positive_labels = positive_labels.to(device)
    negative_labels = negative_labels.to(device)

    with torch.no_grad():
        positive_scores = model(eeg_queries, text_queries, docs)
        positive_loss = compute_bce_loss(positive_scores, positive_labels)
        pos_acc = ((torch.sigmoid(positive_scores.squeeze()) > 0.5) == positive_labels).float().mean()

        total_loss = positive_loss
        neg_acc = torch.tensor(0.0)
        if len(negative_pairs) > 0:
            neg_eeg = torch.stack([eeg_queries[pair['eeg_idx']] for pair in negative_pairs])
            neg_text_queries = {
                'input_ids': torch.stack([text_queries['input_ids'][pair['eeg_idx']] for pair in negative_pairs]),
                'attention_mask': torch.stack([text_queries['attention_mask'][pair['eeg_idx']] for pair in negative_pairs])
            }
            neg_docs = {
                'input_ids': torch.stack([docs['input_ids'][pair['doc_idx']] for pair in negative_pairs]),
                'attention_mask': torch.stack([docs['attention_mask'][pair['doc_idx']] for pair in negative_pairs])
            }
            negative_scores = model(neg_eeg, neg_text_queries, neg_docs)
            negative_loss = compute_bce_loss(negative_scores, negative_labels)
            total_loss = positive_loss + negative_loss
            neg_acc = ((torch.sigmoid(negative_scores.squeeze()) > 0.5) == negative_labels.to(device)).float().mean()

    return total_loss.item(), {'accuracy': pos_acc.item(), 'negative_accuracy': neg_acc.item() if torch.is_tensor(neg_acc) else neg_acc}


def compute_contrastive_loss(query_vectors, doc_vectors, pooling_strategy, temperature=0.07):
    if pooling_strategy == 'multi':
        batch_size = len(query_vectors)
        device = query_vectors[0].device
    else:
        batch_size = query_vectors.size(0)
        device = query_vectors.device

    query_to_doc_sims = []
    for i in range(batch_size):
        if pooling_strategy == 'multi':
            query_i = query_vectors[i]
            doc_i = doc_vectors[i]
        else:
            query_i = query_vectors[i:i + 1]
            doc_i = doc_vectors[i:i + 1]

        sim = compute_similarity([query_i], [doc_i], pooling_strategy, temperature=1.0)
        query_to_doc_sims.append(sim[0])

    similarities = torch.stack(query_to_doc_sims)

    logits = torch.zeros(batch_size, batch_size, device=device)

    for i in range(batch_size):
        for j in range(batch_size):
            if pooling_strategy == 'multi':
                query_i = query_vectors[i]
                doc_j = doc_vectors[j]
            else:
                query_i = query_vectors[i:i + 1]
                doc_j = doc_vectors[j:j + 1]

            sim = compute_similarity([query_i], [doc_j], pooling_strategy, temperature=1.0)
            logits[i, j] = sim[0] / temperature

    labels = torch.arange(batch_size, device=device)
    loss = F.cross_entropy(logits, labels)

    return loss, similarities


def compute_alignment_metrics(query_vectors, doc_vectors, pooling_strategy, query_type='eeg'):
    if pooling_strategy == 'multi':
        batch_size = len(query_vectors)
    else:
        batch_size = query_vectors.size(0)

    query_doc_sims = []
    for i in range(batch_size):
        if pooling_strategy == 'multi':
            query_i = query_vectors[i]
            doc_i = doc_vectors[i]
        else:
            query_i = query_vectors[i:i + 1]
            doc_i = doc_vectors[i:i + 1]

        sim = compute_similarity([query_i], [doc_i], pooling_strategy, temperature=1.0)
        query_doc_sims.append(sim[0].item())

    metrics = {
        'query_doc_similarity': np.mean(query_doc_sims),
        'query_doc_similarity_std': np.std(query_doc_sims),
    }

    if query_type == 'eeg':
        metrics['eeg_query_similarity'] = metrics['query_doc_similarity']
        metrics['eeg_query_similarity_std'] = metrics['query_doc_similarity_std']
        metrics['eeg_doc_similarity'] = metrics['query_doc_similarity']
        metrics['eeg_doc_similarity_std'] = metrics['query_doc_similarity_std']
    else:
        metrics['text_query_similarity'] = metrics['query_doc_similarity']
        metrics['text_query_similarity_std'] = metrics['query_doc_similarity_std']
        metrics['text_doc_similarity'] = metrics['query_doc_similarity']
        metrics['text_doc_similarity_std'] = metrics['query_doc_similarity_std']

    return metrics


def dual_encoder_train_step(model, batch, optimizer, device, step_num, debug=False):
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}
    eeg_mv_masks = batch['eeg_mv_masks'].to(device)

    if debug:
        print(f"[DEBUG] Dual encoder training step {step_num} (query_type: {model.query_type})")
        print(f"  EEG queries: {eeg_queries.shape}")
        print(f"  Text query IDs: {text_queries['input_ids'].shape}")
        print(f"  Doc IDs: {docs['input_ids'].shape}")
        print(f"  Pooling strategy: {model.pooling_strategy}")

    outputs = model(eeg_queries, text_queries, docs, eeg_mv_masks)

    if debug:
        print(f"  Output types:")
        for key, value in outputs.items():
            if isinstance(value, list):
                print(f"    {key}: List with {len(value)} elements")
                if len(value) > 0:
                    print(f"      First element: {value[0].shape}")
            else:
                print(f"    {key}: {value.shape if value is not None else 'None'}")

    loss, query_sims = compute_contrastive_loss(
        outputs['query_vectors'],
        outputs['doc_vectors'],
        model.pooling_strategy
    )

    metrics = compute_alignment_metrics(
        outputs['query_vectors'],
        outputs['doc_vectors'],
        model.pooling_strategy,
        model.query_type
    )

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if debug:
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Query-Doc similarity: {metrics['query_doc_similarity']:.4f}")
        print(f"  Grad norm: {grad_norm.item():.4f}")
        meta = batch['metadata'][0]
        print(f"  Sample query: '{meta['query_text'][:50]}...'")
        print(f"  Sample doc: '{meta['document_text'][:50]}...'")
        print(f"  Participant: {meta['participant_id']}")
        print(f"  Was masked: {meta.get('was_masked', 'N/A')}")
        print(f"  Masking probability: {meta.get('current_masking_probability', 'N/A')}")

    return loss.item(), metrics, grad_norm.item()


def dual_encoder_validation_step(model, batch, device):
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}
    eeg_mv_masks = batch['eeg_mv_masks'].to(device)

    with torch.no_grad():
        outputs = model(eeg_queries, text_queries, docs, eeg_mv_masks)

        loss, query_sims = compute_contrastive_loss(
            outputs['query_vectors'],
            outputs['doc_vectors'],
            model.pooling_strategy
        )

        metrics = compute_alignment_metrics(
            outputs['query_vectors'],
            outputs['doc_vectors'],
            model.pooling_strategy,
            model.query_type
        )

    return loss.item(), metrics


def train_step(model, batch, optimizer, device, step_num, debug=False):
    if hasattr(model, 'cross_attention'):
        return cross_encoder_train_step(model, batch, optimizer, device, step_num, debug)
    else:
        return dual_encoder_train_step(model, batch, optimizer, device, step_num, debug)


def validation_step(model, batch, device):
    if hasattr(model, 'cross_attention'):
        return cross_encoder_validation_step(model, batch, device)
    else:
        return dual_encoder_validation_step(model, batch, device)


def build_document_database(val_dataloader):
    print("Building document database for ranking evaluation...")

    unique_docs = {}
    query_to_doc_mapping = {}
    query_idx = 0

    for batch in val_dataloader:
        for sample_idx, metadata in enumerate(batch['metadata']):
            doc_text = metadata['document_text'].strip()

            if doc_text:
                if doc_text not in unique_docs:
                    unique_doc_idx = len(unique_docs)
                    unique_docs[doc_text] = {
                        'idx': unique_doc_idx,
                        'text': doc_text,
                        'input_ids': batch['docs']['input_ids'][sample_idx].clone(),
                        'attention_mask': batch['docs']['attention_mask'][sample_idx].clone()
                    }
                else:
                    unique_doc_idx = unique_docs[doc_text]['idx']

                query_to_doc_mapping[query_idx] = unique_doc_idx
            query_idx += 1

    doc_list = [None] * len(unique_docs)
    for text, doc_info in unique_docs.items():
        doc_list[doc_info['idx']] = doc_info

    print(f"Found {len(doc_list)} unique documents for {len(query_to_doc_mapping)} queries")
    return doc_list, query_to_doc_mapping


def generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size=100, seed=42):
    print(f"Generating consistent document subsets (subset_size={subset_size}, seed={seed})...")

    random.seed(seed)
    query_subsets = {}

    for query_idx, correct_doc_idx in query_to_doc_mapping.items():
        doc_subset_indices = [correct_doc_idx]
        negative_candidates = [i for i in range(len(doc_list)) if i != correct_doc_idx]
        if negative_candidates:
            random_negatives = random.sample(negative_candidates, min(subset_size - 1, len(negative_candidates)))
            doc_subset_indices.extend(random_negatives)

        query_subsets[query_idx] = doc_subset_indices

    print(f"Generated {len(query_subsets)} consistent subsets")
    return query_subsets


def batch_similarity_computation(eeg_vectors, doc_vectors_list, pooling_strategy):
    similarities = []

    if pooling_strategy == 'multi':
        for doc_vectors in doc_vectors_list:
            sim = compute_similarity([eeg_vectors], [doc_vectors], pooling_strategy, temperature=1.0)
            similarities.append(sim[0].item())

    elif pooling_strategy == 'cls':
        if len(doc_vectors_list) > 0:
            doc_stack = torch.stack([doc_vec[0] for doc_vec in doc_vectors_list])
            eeg_batch = eeg_vectors.repeat(len(doc_vectors_list), 1, 1)
            batch_similarities = compute_similarity(eeg_batch, doc_stack, pooling_strategy, temperature=1.0)
            similarities = batch_similarities.tolist()

    return similarities


def rank_documents_for_query(model, eeg_query, eeg_mv_mask, doc_vectors_list, pooling_strategy):
    model.eval()

    with torch.no_grad():
        eeg_vectors = model.encode_eeg(eeg_query, eeg_mv_mask)

        if isinstance(eeg_vectors, list):
            eeg_vectors = eeg_vectors[0]
        else:
            eeg_vectors = eeg_vectors[0:1]

        doc_scores = batch_similarity_computation(eeg_vectors, doc_vectors_list, pooling_strategy)

    doc_indices_and_scores = list(enumerate(doc_scores))
    doc_indices_and_scores.sort(key=lambda x: x[1], reverse=True)

    ranked_doc_indices = [idx for idx, score in doc_indices_and_scores]
    ranked_scores = [score for idx, score in doc_indices_and_scores]

    return ranked_doc_indices, ranked_scores


def collect_eeg_queries(val_dataloader, device):
    print("Collecting EEG and text queries from validation set...")

    queries = []
    query_idx = 0

    for batch in val_dataloader:
        batch_size = batch['eeg_queries'].size(0)
        for sample_idx in range(batch_size):
            eeg_query = batch['eeg_queries'][sample_idx:sample_idx + 1].to(device)
            eeg_mv_mask = batch['eeg_mv_masks'][sample_idx:sample_idx + 1].to(device)
            text_query = {
                'input_ids': batch['text_queries']['input_ids'][sample_idx:sample_idx + 1].to(device),
                'attention_mask': batch['text_queries']['attention_mask'][sample_idx:sample_idx + 1].to(device)
            }
            queries.append((eeg_query, eeg_mv_mask, text_query))
            query_idx += 1

    print(f"Collected {len(queries)} query sets (EEG + text)")
    return queries


def compute_ranking_metrics(ranked_doc_indices, correct_doc_idx, k_values=[1, 5, 10, 20]):
    metrics = {}

    try:
        correct_rank = ranked_doc_indices.index(correct_doc_idx) + 1
    except ValueError:
        correct_rank = len(ranked_doc_indices) + 1

    metrics['rr'] = 1.0 / correct_rank if correct_rank <= len(ranked_doc_indices) else 0.0
    metrics['rank_of_correct'] = correct_rank

    for k in k_values:
        if k <= len(ranked_doc_indices):
            hit_at_k = 1.0 if correct_rank <= k else 0.0
            metrics[f'hit_at_{k}'] = hit_at_k
            metrics[f'precision_at_{k}'] = hit_at_k / k
            metrics[f'recall_at_{k}'] = hit_at_k
        else:
            metrics[f'hit_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'recall_at_{k}'] = 0.0

    return metrics


def validate_single_masking_level(model, val_dataloader, device, masking_probability, masking_level_name, is_primary=False):
    original_probability = val_dataloader.dataset.get_current_masking_probability()
    val_dataloader.dataset.set_masking_probability(masking_probability)

    try:
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        is_cross_encoder = hasattr(model, 'cross_attention')

        if is_cross_encoder:
            val_accuracies = []
            val_neg_accuracies = []
        else:
            val_similarities = []
            val_query_doc_sims = []
            val_eeg_doc_sims = []

        for batch_idx, batch in enumerate(val_dataloader):
            val_loss, val_metrics = validation_step(model, batch, device)

            total_val_loss += val_loss
            num_val_batches += 1

            if is_cross_encoder:
                val_accuracies.append(val_metrics['accuracy'])
                val_neg_accuracies.append(val_metrics['negative_accuracy'])
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    similarity_key = 'text_query_similarity'
                else:
                    similarity_key = 'eeg_query_similarity'

                val_similarities.append(val_metrics[similarity_key])
                val_query_doc_sims.append(val_metrics['query_doc_similarity'])

                doc_similarity_key = 'text_doc_similarity' if model.query_type == 'text' else 'eeg_doc_similarity'
                val_eeg_doc_sims.append(val_metrics[doc_similarity_key])

        avg_val_loss = total_val_loss / num_val_batches

        if is_cross_encoder:
            avg_val_accuracy = np.mean(val_accuracies)
            avg_val_neg_accuracy = np.mean(val_neg_accuracies)
            main_metric = avg_val_accuracy

            metrics_dict = {
                'loss': avg_val_loss,
                'accuracy': avg_val_accuracy,
                'negative_accuracy': avg_val_neg_accuracy
            }
        else:
            avg_val_similarity = np.mean(val_similarities)
            avg_query_doc_sim = np.mean(val_query_doc_sims)
            avg_eeg_doc_sim = np.mean(val_eeg_doc_sims)
            main_metric = avg_val_similarity

            metrics_dict = {
                'loss': avg_val_loss,
                'query_doc_similarity': avg_query_doc_sim,
                'eeg_doc_similarity': avg_eeg_doc_sim
            }

            if hasattr(model, 'query_type') and model.query_type == 'text':
                metrics_dict.update({
                    'text_query_similarity': avg_val_similarity,
                    'text_query_similarity_std': np.std(val_similarities),
                    'text_doc_similarity': avg_eeg_doc_sim,
                })
            else:
                metrics_dict.update({
                    'eeg_query_similarity': avg_val_similarity,
                    'eeg_query_similarity_std': np.std(val_similarities),
                    'eeg_doc_similarity': avg_eeg_doc_sim,
                })

        return metrics_dict, main_metric

    finally:
        val_dataloader.dataset.set_masking_probability(original_probability)


def perform_ranking_validation(model, val_dataloader, device, epoch_num, subset_size=None):
    dataset_sources_info = detect_dataset_sources(val_dataloader)

    print(f"\n=== VALIDATION RANKING (Epoch {epoch_num}) ===")

    doc_list, query_to_doc_mapping = build_document_database(val_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        print("Warning: No valid query-document pairs found for ranking evaluation")
        return {}

    if subset_size is None:
        subset_size = len(doc_list)
        print(f"No subset_size specified, using all {subset_size} documents")
    elif len(doc_list) < subset_size:
        print(f"Warning: Only {len(doc_list)} documents available, using all")
        subset_size = len(doc_list)

    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(val_dataloader, device)

    print(f"Dual-encoder ranking: {len(queries)} queries against {subset_size} documents each")
    print(f"Query type: {model.query_type}")

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_dual_encoder_subset(
            model, eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{len(queries)} queries...")

    if not all_metrics:
        print("Warning: No metrics computed")
        return {}

    main_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        main_metrics[f'val_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            main_metrics[f'val_ranking/main/{metric_name}_std'] = np.std(values)

    main_metrics.update({
        'val_ranking/main/num_unique_documents': len(doc_list),
        'val_ranking/main/num_queries_evaluated': len(all_metrics),
        'val_ranking/main/epoch_num': epoch_num,
        'val_ranking/main/subset_size': subset_size,
        'val_ranking/main/query_type': model.query_type
    })

    print(f"Main Ranking Results:")
    print(f"  Query Type: {model.query_type}")
    print(f"  Queries Evaluated: {len(all_metrics)}")
    print(f"  MRR: {main_metrics['val_ranking/main/rr']:.4f}")
    print(f"  Hit@1: {main_metrics['val_ranking/main/hit_at_1']:.4f}")
    print(f"  Hit@5: {main_metrics['val_ranking/main/hit_at_5']:.4f}")
    print(f"  Hit@10: {main_metrics['val_ranking/main/hit_at_10']:.4f}")
    print(f"  Hit@20: {main_metrics['val_ranking/main/hit_at_20']:.4f}")

    dataset_metrics = {}
    if dataset_sources_info['has_multiple']:
        print(f"\n=== DATASET-SPECIFIC VALIDATION RANKING ===")
        dataset_metrics = perform_dataset_specific_ranking(
            model, val_dataloader, device, epoch_num, subset_size, prefix='val_ranking/dataset'
        )

    all_wandb_metrics = {**main_metrics, **dataset_metrics}

    print("=" * 60)
    return all_wandb_metrics


def split_validation_data_by_dataset(val_dataloader):
    dataset_batches = {}

    print("Splitting validation data by dataset source...")

    for batch in val_dataloader:
        batch_size = len(batch['metadata'])

        for sample_idx in range(batch_size):
            metadata = batch['metadata'][sample_idx]
            dataset_source = metadata.get('dataset_source', 'unknown')

            if 'nieuwland' in dataset_source.lower():
                dataset_name = 'nieuwland'
            elif 'alice' in dataset_source.lower():
                dataset_name = 'alice'
            elif 'dataset_1' in dataset_source.lower():
                dataset_name = 'nieuwland'
            elif 'dataset_2' in dataset_source.lower():
                dataset_name = 'alice'
            else:
                print(f"DEBUG: Unknown dataset_source: '{dataset_source}'")
                dataset_name = 'unknown'

            if dataset_name not in dataset_batches:
                dataset_batches[dataset_name] = []

            sample = {
                'eeg_query': batch['eeg_queries'][sample_idx:sample_idx + 1],
                'text_query': {
                    'input_ids': batch['text_queries']['input_ids'][sample_idx:sample_idx + 1],
                    'attention_mask': batch['text_queries']['attention_mask'][sample_idx:sample_idx + 1]
                },
                'doc': {
                    'input_ids': batch['docs']['input_ids'][sample_idx:sample_idx + 1],
                    'attention_mask': batch['docs']['attention_mask'][sample_idx:sample_idx + 1]
                },
                'eeg_mv_mask': batch['eeg_mv_masks'][sample_idx:sample_idx + 1],
                'metadata': [metadata]
            }

            dataset_batches[dataset_name].append(sample)

    for dataset_name, samples in dataset_batches.items():
        print(f"  {dataset_name}: {len(samples)} samples")

    return dataset_batches


def validate_dataset_specific(model, dataset_batches, device, primary_masking_level=90):
    primary_masking_prob = primary_masking_level / 100.0
    is_cross_encoder = hasattr(model, 'cross_attention')

    dataset_results = {}

    print(f"\n=== DATASET-SPECIFIC VALIDATION (Masking: {primary_masking_level}%) ===")

    for dataset_name, samples in dataset_batches.items():
        if len(samples) == 0:
            continue

        print(f"  Validating on {dataset_name} dataset ({len(samples)} samples)...")

        model.eval()
        total_loss = 0
        num_samples = 0

        if is_cross_encoder:
            accuracies = []
            neg_accuracies = []
        else:
            similarities = []
            query_doc_sims = []
            eeg_doc_sims = []

        batch_size = 16
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]

            mini_batch = {
                'eeg_queries': torch.cat([s['eeg_query'] for s in batch_samples], dim=0),
                'text_queries': {
                    'input_ids': torch.cat([s['text_query']['input_ids'] for s in batch_samples], dim=0),
                    'attention_mask': torch.cat([s['text_query']['attention_mask'] for s in batch_samples], dim=0)
                },
                'docs': {
                    'input_ids': torch.cat([s['doc']['input_ids'] for s in batch_samples], dim=0),
                    'attention_mask': torch.cat([s['doc']['attention_mask'] for s in batch_samples], dim=0)
                },
                'eeg_mv_masks': torch.cat([s['eeg_mv_mask'] for s in batch_samples], dim=0),
                'metadata': [s['metadata'][0] for s in batch_samples]
            }

            loss, metrics = validation_step(model, mini_batch, device)

            total_loss += loss * len(batch_samples)
            num_samples += len(batch_samples)

            if is_cross_encoder:
                accuracies.append(metrics['accuracy'])
                neg_accuracies.append(metrics['negative_accuracy'])
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    similarity_key = 'text_query_similarity'
                    doc_similarity_key = 'text_doc_similarity'
                else:
                    similarity_key = 'eeg_query_similarity'
                    doc_similarity_key = 'eeg_doc_similarity'

                similarities.append(metrics[similarity_key])
                query_doc_sims.append(metrics['query_doc_similarity'])
                eeg_doc_sims.append(metrics[doc_similarity_key])

        avg_loss = total_loss / num_samples

        if is_cross_encoder:
            avg_accuracy = np.mean(accuracies)
            avg_neg_accuracy = np.mean(neg_accuracies)
            main_metric = avg_accuracy

            result_metrics = {
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'negative_accuracy': avg_neg_accuracy,
                'num_samples': num_samples
            }

            print(f"    {dataset_name}: Loss {avg_loss:.4f}, Accuracy {avg_accuracy:.4f}")

        else:
            avg_similarity = np.mean(similarities)
            avg_query_doc_sim = np.mean(query_doc_sims)
            avg_eeg_doc_sim = np.mean(eeg_doc_sims)
            main_metric = avg_similarity

            result_metrics = {
                'loss': avg_loss,
                'query_doc_similarity': avg_query_doc_sim,
                'num_samples': num_samples
            }

            if hasattr(model, 'query_type') and model.query_type == 'text':
                result_metrics.update({
                    'text_query_similarity': avg_similarity,
                    'text_doc_similarity': avg_eeg_doc_sim
                })
                print(f"    {dataset_name}: Loss {avg_loss:.4f}, Text-Query Sim {avg_similarity:.4f}")
            else:
                result_metrics.update({
                    'eeg_query_similarity': avg_similarity,
                    'eeg_doc_similarity': avg_eeg_doc_sim
                })
                print(f"    {dataset_name}: Loss {avg_loss:.4f}, EEG-Query Sim {avg_similarity:.4f}")

        dataset_results[dataset_name] = {
            'metrics': result_metrics,
            'main_metric': main_metric
        }

    print("=" * 60)
    return dataset_results


def rank_dual_encoder_subset(model, eeg_query, eeg_mv_mask, doc_list, doc_indices, device, text_query=None):
    model.eval()

    with torch.no_grad():
        if model.query_type == 'eeg':
            eeg_output = model.encode_eeg(eeg_query, eeg_mv_mask)
            query_vectors = eeg_output
        else:
            if text_query is None:
                raise ValueError("text_query is required when model.query_type='text'")
            query_vectors = model.encode_text(text_query['input_ids'], text_query['attention_mask'])

        if isinstance(query_vectors, list):
            query_vectors = query_vectors[0]
        else:
            query_vectors = query_vectors[0:1]

        doc_input_ids = torch.stack([doc_list[doc_idx]['input_ids'] for doc_idx in doc_indices]).to(device)
        doc_attention_masks = torch.stack([doc_list[doc_idx]['attention_mask'] for doc_idx in doc_indices]).to(device)

        batch_doc_vectors = model.encode_text(doc_input_ids, doc_attention_masks)

        if isinstance(batch_doc_vectors, list):
            doc_vectors_list = batch_doc_vectors
        else:
            doc_vectors_list = [batch_doc_vectors[i:i + 1] for i in range(batch_doc_vectors.size(0))]

        scores = []
        for i, doc_vectors in enumerate(doc_vectors_list):
            sim = compute_similarity([query_vectors], [doc_vectors], model.pooling_strategy, temperature=1.0)
            scores.append((doc_indices[i], sim[0].item()))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_indices = [doc_idx for doc_idx, score in scores]
    ranked_scores = [score for doc_idx, score in scores]

    return ranked_indices, ranked_scores


def perform_cross_encoder_ranking_validation(model, val_dataloader, device, epoch_num, subset_size=100):
    dataset_sources_info = detect_dataset_sources(val_dataloader)

    print(f"\n=== CROSS-ENCODER VALIDATION RANKING (Epoch {epoch_num}) ===")

    doc_list, query_to_doc_mapping = build_document_database(val_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        print("Warning: No valid query-document pairs found for ranking evaluation")
        return {}

    if len(doc_list) < subset_size:
        print(f"Warning: Only {len(doc_list)} documents available, using all")
        subset_size = len(doc_list)

    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(val_dataloader, device)

    print(f"Cross-encoder ranking: {len(queries)} queries against {subset_size} documents each")

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_cross_encoder_subset(
            model, eeg_query, text_query, doc_list, doc_subset_indices, device
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{len(queries)} queries...")

    if not all_metrics:
        print("Warning: No metrics computed")
        return {}

    main_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        main_metrics[f'val_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            main_metrics[f'val_ranking/main/{metric_name}_std'] = np.std(values)

    main_metrics.update({
        'val_ranking/main/num_unique_documents': len(doc_list),
        'val_ranking/main/num_queries_evaluated': len(all_metrics),
        'val_ranking/main/epoch_num': epoch_num,
        'val_ranking/main/subset_size': subset_size,
        'val_ranking/main/encoder_type': 'cross'
    })

    print(f"Cross-Encoder Main Ranking Results:")
    print(f"  Queries Evaluated: {len(all_metrics)}")
    print(f"  MRR: {main_metrics['val_ranking/main/rr']:.4f}")
    print(f"  Hit@1: {main_metrics['val_ranking/main/hit_at_1']:.4f}")
    print(f"  Hit@5: {main_metrics['val_ranking/main/hit_at_5']:.4f}")
    print(f"  Hit@10: {main_metrics['val_ranking/main/hit_at_10']:.4f}")
    print(f"  Hit@20: {main_metrics['val_ranking/main/hit_at_20']:.4f}")

    dataset_metrics = {}
    if dataset_sources_info['has_multiple']:
        print(f"\n=== DATASET-SPECIFIC VALIDATION RANKING ===")
        dataset_metrics = perform_dataset_specific_ranking(
            model, val_dataloader, device, epoch_num, subset_size, prefix='val_ranking/dataset'
        )

    all_wandb_metrics = {**main_metrics, **dataset_metrics}

    print("=" * 60)
    return all_wandb_metrics


def rank_cross_encoder_subset(model, eeg_query, text_query, doc_list, doc_indices, device):
    model.eval()

    with torch.no_grad():
        doc_input_ids = torch.stack([doc_list[doc_idx]['input_ids'] for doc_idx in doc_indices]).to(device)
        doc_attention_masks = torch.stack([doc_list[doc_idx]['attention_mask'] for doc_idx in doc_indices]).to(device)

        batch_size = len(doc_indices)

        batch_eeg_queries = eeg_query.repeat(batch_size, 1, 1, 1)
        batch_text_queries = {
            'input_ids': text_query['input_ids'].repeat(batch_size, 1),
            'attention_mask': text_query['attention_mask'].repeat(batch_size, 1)
        }
        batch_docs = {
            'input_ids': doc_input_ids,
            'attention_mask': doc_attention_masks
        }

        batch_scores = model(batch_eeg_queries, batch_text_queries, batch_docs)
        scores = batch_scores.squeeze().cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

    doc_score_pairs = list(zip(doc_indices, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    ranked_indices = [doc_idx for doc_idx, score in doc_score_pairs]
    ranked_scores = [score for doc_idx, score in doc_score_pairs]

    return ranked_indices, ranked_scores


def detect_dataset_sources(val_dataloader):
    dataset_sources = set()

    for batch in val_dataloader:
        for metadata in batch['metadata']:
            dataset_source = metadata.get('dataset_source', 'unknown')
            dataset_sources.add(dataset_source)

    dataset_sources = list(dataset_sources)
    print(f"Detected dataset sources: {dataset_sources}")

    has_multiple_datasets = len(dataset_sources) > 1

    return {
        'all_sources': dataset_sources,
        'has_multiple': has_multiple_datasets,
    }


def train_epoch(model, dataloader, optimizer, device, epoch_num, total_epochs, debug=False):
    model.train()
    total_loss = 0
    num_batches = 0

    is_cross_encoder = hasattr(model, 'cross_attention')

    if is_cross_encoder:
        epoch_accuracies = []
    else:
        epoch_similarities = []

    epoch_grad_norms = []

    for batch_idx, batch in enumerate(dataloader):
        debug_this_batch = debug and epoch_num == 1 and batch_idx == 0

        step_num = (epoch_num - 1) * len(dataloader) + batch_idx

        loss, metrics, grad_norm = train_step(
            model, batch, optimizer, device, step_num, debug=debug_this_batch
        )

        total_loss += loss
        num_batches += 1

        if is_cross_encoder:
            epoch_accuracies.append(metrics['accuracy'])
        else:
            if hasattr(model, 'query_type') and model.query_type == 'text':
                similarity_key = 'text_query_similarity'
                display_name = 'Text-Query Sim'
            else:
                similarity_key = 'eeg_query_similarity'
                display_name = 'EEG-Query Sim'

            epoch_similarities.append(metrics[similarity_key])

        epoch_grad_norms.append(grad_norm)

        if batch_idx % 20 == 0:
            if is_cross_encoder:
                print(f"Epoch {epoch_num}/{total_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    sim_display = f"Text-Query Sim: {metrics['text_query_similarity']:.4f}"
                else:
                    sim_display = f"EEG-Query Sim: {metrics['eeg_query_similarity']:.4f}"
                print(f"Epoch {epoch_num}/{total_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Loss: {loss:.4f}, {sim_display}")

    avg_loss = total_loss / num_batches
    if is_cross_encoder:
        avg_metric = np.mean(epoch_accuracies)
        metric_name = "Accuracy"
    else:
        avg_metric = np.mean(epoch_similarities)
        if hasattr(model, 'query_type') and model.query_type == 'text':
            metric_name = "Text-Query Sim"
        else:
            metric_name = "EEG-Query Sim"

    avg_grad_norm = np.mean(epoch_grad_norms)

    print(f"Epoch {epoch_num} training completed. "
          f"Avg Loss: {avg_loss:.4f}, Avg {metric_name}: {avg_metric:.4f}")

    return avg_loss, avg_metric, avg_grad_norm


def validate_epoch(model, val_dataloader, device, epoch_num,
                   enable_multi_masking_validation=False,
                   validation_masking_levels=[0, 25, 50, 75, 90, 100],
                   multi_masking_frequency=3,
                   primary_masking_level=90):
    print(f"Running standard validation for epoch {epoch_num}...")

    primary_masking_prob = primary_masking_level / 100.0
    primary_metrics, primary_main_metric = validate_single_masking_level(
        model, val_dataloader, device, primary_masking_prob, f"masking_{primary_masking_level}%", is_primary=True
    )

    is_cross_encoder = hasattr(model, 'cross_attention')

    if is_cross_encoder:
        print(f"Standard validation completed. Val Loss: {primary_metrics['loss']:.4f}, "
              f"Val Accuracy: {primary_metrics['accuracy']:.4f}")
    else:
        if hasattr(model, 'query_type') and model.query_type == 'text':
            display_name = "Text-Query Sim"
            similarity_key = 'text_query_similarity'
        else:
            display_name = "EEG-Query Sim"
            similarity_key = 'eeg_query_similarity'

        print(f"Standard validation completed. Val Loss: {primary_metrics['loss']:.4f}, "
              f"{display_name}: {primary_metrics[similarity_key]:.4f}")

    dataset_sources_info = detect_dataset_sources(val_dataloader)
    if dataset_sources_info['has_multiple']:
        print(f"\nDetected multiple datasets - running dataset-specific validation...")

        dataset_batches = split_validation_data_by_dataset(val_dataloader)
        dataset_results = validate_dataset_specific(model, dataset_batches, device, primary_masking_level)

        print(f"Dataset-specific validation completed")
    else:
        print(f"Single dataset detected - skipping dataset-specific validation")

    if (enable_multi_masking_validation and epoch_num % multi_masking_frequency == 0):

        print(f"\n=== DYNAMIC MULTI-MASKING VALIDATION (Epoch {epoch_num}) ===")
        print(f"Testing masking levels: {validation_masking_levels}%")
        print(f"Using SINGLE dataloader with dynamic masking probability")

        all_masking_metrics = {}

        for masking_level in validation_masking_levels:
            masking_probability = masking_level / 100.0
            masking_level_name = f"{masking_level}%"

            print(f"  Validating on {masking_level_name} masking...")

            masking_metrics, masking_main_metric = validate_single_masking_level(
                model, val_dataloader, device, masking_probability, masking_level_name
            )

            all_masking_metrics[masking_level_name] = {
                'metrics': masking_metrics,
                'main_metric': masking_main_metric
            }

            if is_cross_encoder:
                print(f"    {masking_level_name}: Loss {masking_metrics['loss']:.4f}, "
                      f"Accuracy {masking_metrics['accuracy']:.4f}")
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    sim_key = 'text_query_similarity'
                else:
                    sim_key = 'eeg_query_similarity'

                print(f"    {masking_level_name}: Loss {masking_metrics['loss']:.4f}, "
                      f"Similarity {masking_metrics[sim_key]:.4f}")

        print(f"\n  DYNAMIC multi-masking validation summary:")
        for masking_level, data in all_masking_metrics.items():
            metrics = data['metrics']
            main_metric = data['main_metric']

            if is_cross_encoder:
                print(f"    {masking_level}: Accuracy {metrics['accuracy']:.4f}")
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    sim_key = 'text_query_similarity'
                else:
                    sim_key = 'eeg_query_similarity'
                print(f"    {masking_level}: Similarity {metrics[sim_key]:.4f}")

        print("=" * 60)

    if epoch_num % 3 == 0:
        if is_cross_encoder:
            print(f"Running cross-encoder ranking validation...")
            ranking_metrics = perform_cross_encoder_ranking_validation(
                model, val_dataloader, device, epoch_num, subset_size=100
            )
        else:
            print(f"Running dual-encoder ranking validation...")
            ranking_metrics = perform_ranking_validation(
                model, val_dataloader, device, epoch_num, subset_size=None
            )

    return primary_metrics['loss'], primary_main_metric


def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs,
                patience=10, device='cuda', debug=False, config=None,
                enable_multi_masking_validation=False, multi_masking_frequency=3,
                validation_masking_levels=[0, 25, 50, 75, 90, 100],
                primary_masking_level=90):
    is_cross_encoder = hasattr(model, 'cross_attention')
    encoder_type = 'Cross-Encoder' if is_cross_encoder else 'Dual-Encoder'
    pooling_info = model.pooling_strategy if hasattr(model, 'pooling_strategy') else 'unknown'

    print(f"Starting {encoder_type} training with early stopping (patience={patience})...")
    print(f"Pooling strategy: {pooling_info}")
    print(f"Training masking level: {config.get('training_masking_level', 90)}%")

    if enable_multi_masking_validation:
        print(f"DYNAMIC multi-masking validation: ENABLED")
        print(f"  Validation masking levels: {validation_masking_levels}%")
        print(f"  Multi-masking frequency: every {multi_masking_frequency} epochs")
        print(f"  Primary masking level: {primary_masking_level}% (for early stopping)")
        print(f"  Memory efficient: Uses SINGLE dataloader with dynamic masking")
    else:
        print(f"Multi-masking validation: DISABLED")

    best_val_loss = float('inf')
    best_val_metric = -1.0 if not is_cross_encoder else 0.0
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = 0
    early_stopped = False

    for epoch in range(num_epochs):
        epoch_num = epoch + 1

        train_loss, train_metric, train_grad_norm = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch_num,
            total_epochs=num_epochs,
            debug=debug
        )

        val_loss, val_metric = validate_epoch(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            epoch_num=epoch_num,
            enable_multi_masking_validation=enable_multi_masking_validation,
            validation_masking_levels=validation_masking_levels,
            multi_masking_frequency=multi_masking_frequency,
            primary_masking_level=primary_masking_level
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metric = val_metric
            best_epoch = epoch_num
            epochs_without_improvement = 0

            best_model_state = model.state_dict().copy()

            print(f"New best validation loss: {best_val_loss:.4f} (epoch {epoch_num})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch_num} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
                early_stopped = True

        metric_name = 'accuracy' if is_cross_encoder else 'similarity'
        metric_display = 'Accuracy' if is_cross_encoder else 'EEG-Query Sim'
        print(f"\nEpoch {epoch_num}/{num_epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, {metric_display}: {train_metric:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, {metric_display}: {val_metric:.4f}")
        print(f"  Best  - Val Loss: {best_val_loss:.4f} (epoch {best_epoch})")
        print(f"  Early Stopping: {epochs_without_improvement}/{patience} epochs without improvement")

        if (enable_multi_masking_validation and epoch_num % multi_masking_frequency == 0):
            print(f"  Note: DYNAMIC multi-masking validation performed this epoch ({validation_masking_levels}%)")
        elif epoch_num % 1 == 0:
            print(f"  Note: Standard ranking validation performed this epoch")

        print("-" * 70)

        if early_stopped:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch}")

    print(f"\n{encoder_type} training completed!")
    if early_stopped:
        print(f"Stopped early after {epoch_num} epochs")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    if enable_multi_masking_validation:
        print(f"DYNAMIC multi-masking validation was performed every {multi_masking_frequency} epochs")
        print(f"Memory efficient: Used single dataloader with dynamic masking")

    return model


def test_model(model, test_dataloader, device, debug=False,
               test_masking_levels=[0, 25, 50, 75, 90, 100],
               primary_masking_level=90):
    print(f"=== COMPREHENSIVE MULTI-MASKING TEST SET EVALUATION ===")
    print(f"Test set size: {len(test_dataloader.dataset)} samples")
    print(f"Test subjects: {len(test_dataloader.dataset.unique_subjects)} unique subjects")
    print(f"Test masking levels: {test_masking_levels}%")
    print(f"Primary masking level: {primary_masking_level}%")

    model.eval()
    is_cross_encoder = hasattr(model, 'cross_attention')

    original_probability = test_dataloader.dataset.get_current_masking_probability()
    all_test_metrics = {}

    try:
        for masking_level in test_masking_levels:
            masking_probability = masking_level / 100.0
            is_primary = (masking_level == primary_masking_level)

            print(f"\n=== TESTING AT {masking_level}% MASKING {'(PRIMARY)' if is_primary else ''} ===")

            test_dataloader.dataset.set_masking_probability(masking_probability)

            if is_cross_encoder:
                main_ranking_results = perform_cross_encoder_test_ranking_at_masking_level(
                    model, test_dataloader, device, masking_level
                )
            else:
                main_ranking_results = perform_dual_encoder_test_ranking_at_masking_level(
                    model, test_dataloader, device, masking_level
                )

            dataset_sources_info = detect_dataset_sources(test_dataloader)
            dataset_ranking_results = {}

            if dataset_sources_info['has_multiple']:
                print(f"  Running dataset-specific evaluation at {masking_level}% masking...")
                dataset_ranking_results = perform_dataset_specific_ranking(
                    model, test_dataloader, device, epoch_num=0, subset_size=100,
                    prefix=f'test_ranking/masking_{masking_level}/dataset'
                )

            masking_prefix = f'test_ranking/masking_{masking_level}'

            main_masking_results = {}
            for key, value in main_ranking_results.items():
                new_key = key.replace('test_ranking/main', masking_prefix)
                main_masking_results[new_key] = value

            level_results = {**main_masking_results, **dataset_ranking_results}
            all_test_metrics.update(level_results)

            if main_ranking_results:
                main_mrr = main_ranking_results.get('test_ranking/main/rr', 0)
                main_hit1 = main_ranking_results.get('test_ranking/main/hit_at_1', 0)
                main_hit10 = main_ranking_results.get('test_ranking/main/hit_at_10', 0)
                print(f"  {masking_level}% Masking: MRR {main_mrr:.4f}, Hit@1 {main_hit1:.4f}, Hit@10 {main_hit10:.4f}")

        print(f"\n=== NOISE BASELINE AT {primary_masking_level}% MASKING ===")
        test_dataloader.dataset.set_masking_probability(primary_masking_level / 100.0)
        noise_ranking_results = perform_noise_ranking_evaluation(
            model, test_dataloader, device, primary_masking_level
        )
        all_test_metrics.update(noise_ranking_results)

        masking_comparison_metrics = calculate_masking_performance_trends(
            all_test_metrics, test_masking_levels
        )
        all_test_metrics.update(masking_comparison_metrics)

        dataset_sources_info = detect_dataset_sources(test_dataloader)
        summary_metrics = {
            'test_ranking/summary/test_masking_levels': test_masking_levels,
            'test_ranking/summary/primary_masking_level': primary_masking_level,
            'test_ranking/summary/num_test_samples': len(test_dataloader.dataset),
            'test_ranking/summary/num_test_subjects': len(test_dataloader.dataset.unique_subjects),
            'test_ranking/summary/has_multiple_datasets': dataset_sources_info['has_multiple'],
            'test_ranking/summary/encoder_type': 'cross' if is_cross_encoder else 'dual'
        }

        if hasattr(model, 'query_type'):
            summary_metrics['test_ranking/summary/query_type'] = model.query_type

        all_test_metrics.update(summary_metrics)

        print_multi_masking_test_summary(all_test_metrics, test_masking_levels, primary_masking_level)

        return all_test_metrics

    finally:
        test_dataloader.dataset.set_masking_probability(original_probability)


def perform_dual_encoder_test_ranking_at_masking_level(model, test_dataloader, device, masking_level):
    doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        return {}

    subset_size = len(doc_list)
    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(test_dataloader, device)

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_dual_encoder_subset(
            model, eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

    if not all_metrics:
        return {}

    test_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        test_metrics[f'test_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            test_metrics[f'test_ranking/main/{metric_name}_std'] = np.std(values)

    test_metrics.update({
        'test_ranking/main/num_unique_documents': len(doc_list),
        'test_ranking/main/num_queries_evaluated': len(all_metrics),
        'test_ranking/main/subset_size': subset_size,
        'test_ranking/main/masking_level': masking_level,
        'test_ranking/main/query_type': model.query_type
    })

    return test_metrics


def perform_cross_encoder_test_ranking_at_masking_level(model, test_dataloader, device, masking_level):
    doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        return {}

    subset_size = len(doc_list)
    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(test_dataloader, device)

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_cross_encoder_subset(
            model, eeg_query, text_query, doc_list, doc_subset_indices, device
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

    if not all_metrics:
        return {}

    test_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        test_metrics[f'test_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            test_metrics[f'test_ranking/main/{metric_name}_std'] = np.std(values)

    test_metrics.update({
        'test_ranking/main/num_unique_documents': len(doc_list),
        'test_ranking/main/num_queries_evaluated': len(all_metrics),
        'test_ranking/main/subset_size': subset_size,
        'test_ranking/main/masking_level': masking_level,
        'test_ranking/main/encoder_type': 'cross'
    })

    return test_metrics


def calculate_masking_performance_trends(all_metrics, test_masking_levels):
    trend_metrics = {}

    mrr_values = []
    hit1_values = []
    hit10_values = []

    for masking_level in test_masking_levels:
        mrr_key = f'test_ranking/masking_{masking_level}/rr'
        hit1_key = f'test_ranking/masking_{masking_level}/hit_at_1'
        hit10_key = f'test_ranking/masking_{masking_level}/hit_at_10'

        if mrr_key in all_metrics:
            mrr_values.append(all_metrics[mrr_key])
        if hit1_key in all_metrics:
            hit1_values.append(all_metrics[hit1_key])
        if hit10_key in all_metrics:
            hit10_values.append(all_metrics[hit10_key])

    if mrr_values:
        trend_metrics.update({
            'test_ranking/trends/mrr_mean': np.mean(mrr_values),
            'test_ranking/trends/mrr_std': np.std(mrr_values),
            'test_ranking/trends/mrr_min': np.min(mrr_values),
            'test_ranking/trends/mrr_max': np.max(mrr_values),
            'test_ranking/trends/mrr_range': np.max(mrr_values) - np.min(mrr_values)
        })

    if hit1_values:
        trend_metrics.update({
            'test_ranking/trends/hit1_mean': np.mean(hit1_values),
            'test_ranking/trends/hit1_std': np.std(hit1_values),
            'test_ranking/trends/hit1_min': np.min(hit1_values),
            'test_ranking/trends/hit1_max': np.max(hit1_values),
            'test_ranking/trends/hit1_range': np.max(hit1_values) - np.min(hit1_values)
        })

    if hit10_values:
        trend_metrics.update({
            'test_ranking/trends/hit10_mean': np.mean(hit10_values),
            'test_ranking/trends/hit10_std': np.std(hit10_values),
            'test_ranking/trends/hit10_min': np.min(hit10_values),
            'test_ranking/trends/hit10_max': np.max(hit10_values),
            'test_ranking/trends/hit10_range': np.max(hit10_values) - np.min(hit10_values)
        })

    return trend_metrics


def print_multi_masking_test_summary(all_metrics, test_masking_levels, primary_masking_level):
    print(f"\n=== MULTI-MASKING TEST EVALUATION SUMMARY ===")
    print(f"Test masking levels: {test_masking_levels}%")
    print(f"Primary masking level: {primary_masking_level}%")

    print(f"\nPerformance across masking levels:")
    print(f"{'Masking':>8} {'MRR':>8} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8}")
    print(f"{'-' * 45}")

    for masking_level in test_masking_levels:
        mrr = all_metrics.get(f'test_ranking/masking_{masking_level}/rr', 0)
        hit1 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_1', 0)
        hit5 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_5', 0)
        hit10 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_10', 0)

        primary_marker = " *" if masking_level == primary_masking_level else ""
        print(f"{masking_level:>6}%{primary_marker:>2} {mrr:>8.4f} {hit1:>8.4f} {hit5:>8.4f} {hit10:>8.4f}")

    noise_mrr = all_metrics.get('test_ranking/noise/rr', 0)
    noise_hit1 = all_metrics.get('test_ranking/noise/hit_at_1', 0)

    if noise_mrr > 0:
        primary_mrr = all_metrics.get(f'test_ranking/masking_{primary_masking_level}/rr', 0)
        primary_hit1 = all_metrics.get(f'test_ranking/masking_{primary_masking_level}/hit_at_1', 0)

        print(f"\nNoise Baseline: MRR {noise_mrr:.4f}, Hit@1 {noise_hit1:.4f}")
        print(f"Primary vs Noise: MRR +{primary_mrr - noise_mrr:.4f}, Hit@1 +{primary_hit1 - noise_hit1:.4f}")

    if 'test_ranking/trends/mrr_range' in all_metrics:
        mrr_range = all_metrics['test_ranking/trends/mrr_range']
        hit1_range = all_metrics['test_ranking/trends/hit1_range']
        print(f"\nRobustness Analysis:")
        print(f"  MRR range across masking levels: {mrr_range:.4f}")
        print(f"  Hit@1 range across masking levels: {hit1_range:.4f}")

    print("=" * 60)


def perform_dataset_specific_ranking(model, dataloader, device, epoch_num, subset_size, prefix='ranking/dataset'):
    print(f"Running dataset-specific ranking evaluation...")

    dataset_batches = split_validation_data_by_dataset(dataloader)

    all_dataset_metrics = {}
    is_cross_encoder = hasattr(model, 'cross_attention')

    for dataset_name, samples in dataset_batches.items():
        if len(samples) == 0:
            continue

        print(f"  Ranking evaluation on {dataset_name} ({len(samples)} samples)...")

        dataset_doc_list = []
        dataset_query_to_doc_mapping = {}

        for query_idx, sample in enumerate(samples):
            doc_text = sample['metadata'][0]['document_text'].strip()

            if doc_text:
                doc_info = {
                    'idx': query_idx,
                    'text': doc_text,
                    'input_ids': sample['doc']['input_ids'].squeeze(0),
                    'attention_mask': sample['doc']['attention_mask'].squeeze(0)
                }
                dataset_doc_list.append(doc_info)
                dataset_query_to_doc_mapping[query_idx] = query_idx

        if len(dataset_doc_list) == 0:
            continue

        actual_subset_size = min(subset_size, len(dataset_doc_list))
        dataset_query_subsets = generate_consistent_subsets(
            dataset_doc_list, dataset_query_to_doc_mapping, actual_subset_size
        )

        dataset_queries = []
        for sample in samples:
            eeg_query = sample['eeg_query'].to(device)
            eeg_mv_mask = sample['eeg_mv_mask'].to(device)
            text_query = {
                'input_ids': sample['text_query']['input_ids'].to(device),
                'attention_mask': sample['text_query']['attention_mask'].to(device)
            }
            dataset_queries.append((eeg_query, eeg_mv_mask, text_query))

        dataset_all_metrics = []

        for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(dataset_queries):
            if query_idx not in dataset_query_to_doc_mapping:
                continue

            correct_doc_idx = dataset_query_to_doc_mapping[query_idx]
            doc_subset_indices = dataset_query_subsets[query_idx]

            if is_cross_encoder:
                ranked_indices, scores = rank_cross_encoder_subset(
                    model, eeg_query, text_query, dataset_doc_list, doc_subset_indices, device
                )
            else:
                ranked_indices, scores = rank_dual_encoder_subset(
                    model, eeg_query, eeg_mv_mask, dataset_doc_list, doc_subset_indices, device, text_query=text_query
                )

            query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
            dataset_all_metrics.append(query_metrics)

        if len(dataset_all_metrics) == 0:
            continue

        dataset_metrics = {}
        metric_names = dataset_all_metrics[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in dataset_all_metrics]
            dataset_metrics[f'{prefix}_{dataset_name}_{metric_name}'] = np.mean(values)
            if metric_name != 'rank_of_correct':
                dataset_metrics[f'{prefix}_{dataset_name}_{metric_name}_std'] = np.std(values)

        dataset_metrics.update({
            f'{prefix}_{dataset_name}_num_queries': len(dataset_all_metrics),
            f'{prefix}_{dataset_name}_subset_size': actual_subset_size,
            f'{prefix}_{dataset_name}_epoch_num': epoch_num
        })

        print(f"    {dataset_name}: MRR {dataset_metrics[f'{prefix}_{dataset_name}_rr']:.4f}, "
              f"Hit@1 {dataset_metrics[f'{prefix}_{dataset_name}_hit_at_1']:.4f}, "
              f"Hit@10 {dataset_metrics[f'{prefix}_{dataset_name}_hit_at_10']:.4f}")

        all_dataset_metrics.update(dataset_metrics)

    return all_dataset_metrics


def perform_noise_ranking_evaluation(model, test_dataloader, device, primary_masking_level=90):
    print(f"\n=== NOISE BASELINE RANKING EVALUATION ===")
    print(f"Testing ranking performance with Gaussian noise instead of EEG queries...")

    original_probability = test_dataloader.dataset.get_current_masking_probability()
    test_dataloader.dataset.set_masking_probability(primary_masking_level / 100.0)

    try:
        doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

        if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
            print("Warning: No valid query-document pairs found for noise ranking")
            return {}

        subset_size = len(doc_list)
        query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
        queries = collect_eeg_queries(test_dataloader, device)

        print(f"Noise ranking: {len(queries)} noise queries against {subset_size} documents each")

        all_metrics = []
        is_cross_encoder = hasattr(model, 'cross_attention')

        for query_idx, (real_eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
            if query_idx not in query_to_doc_mapping:
                continue

            noise_eeg_query = generate_eeg_noise_baseline(real_eeg_query).to(device)

            correct_doc_idx = query_to_doc_mapping[query_idx]
            doc_subset_indices = query_subsets[query_idx]

            if is_cross_encoder:
                ranked_indices, scores = rank_cross_encoder_subset(
                    model, noise_eeg_query, text_query, doc_list, doc_subset_indices, device
                )
            else:
                ranked_indices, scores = rank_dual_encoder_subset(
                    model, noise_eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
                )

            query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
            all_metrics.append(query_metrics)

            if (query_idx + 1) % 50 == 0:
                print(f"  Processed {query_idx + 1}/{len(queries)} noise queries...")

        if not all_metrics:
            print("Warning: No noise ranking metrics computed")
            return {}

        noise_metrics = {}
        metric_names = all_metrics[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            noise_metrics[f'test_ranking/noise/{metric_name}'] = np.mean(values)
            if metric_name != 'rank_of_correct':
                noise_metrics[f'test_ranking/noise/{metric_name}_std'] = np.std(values)

        noise_metrics.update({
            'test_ranking/noise/num_unique_documents': len(doc_list),
            'test_ranking/noise/num_queries_evaluated': len(all_metrics),
            'test_ranking/noise/subset_size': subset_size,
            'test_ranking/noise/masking_level': primary_masking_level
        })

        print(f"Noise Baseline Ranking Results:")
        print(f"  Queries Evaluated: {len(all_metrics)}")
        print(f"  MRR: {noise_metrics['test_ranking/noise/rr']:.4f}")
        print(f"  Hit@1: {noise_metrics['test_ranking/noise/hit_at_1']:.4f}")
        print(f"  Hit@5: {noise_metrics['test_ranking/noise/hit_at_5']:.4f}")
        print(f"  Hit@10: {noise_metrics['test_ranking/noise/hit_at_10']:.4f}")
        print(f"  Hit@20: {noise_metrics['test_ranking/noise/hit_at_20']:.4f}")

        return noise_metrics

    finally:
        test_dataloader.dataset.set_masking_probability(original_probability)


def generate_eeg_noise_baseline(eeg_queries, noise_type='gaussian'):
    if noise_type == 'gaussian':
        noise = torch.randn_like(eeg_queries)
        eeg_std = eeg_queries.std()
        eeg_mean = eeg_queries.mean()
        noise = noise * eeg_std + eeg_mean
    return noise