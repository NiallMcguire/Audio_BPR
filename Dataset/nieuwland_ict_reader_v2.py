#!/usr/bin/env python3
"""
Nieuwland EEG ICT Pair Generation

Generates ICT (Inverse Cloze Task) pairs from Nieuwland EEG dataset
with runtime masking support for multi-level validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from typing import Dict, List, Tuple, Optional
import warnings
from collections import defaultdict
from tqdm import tqdm
import random
from dataclasses import dataclass
import json
from datetime import datetime
import argparse

warnings.filterwarnings('ignore')

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


@dataclass
class ICTPair:
    """Data class for ICT query-document pairs"""
    query_text: str
    query_eeg: np.ndarray
    query_words: List[str]
    doc_text: str
    doc_eeg: np.ndarray
    doc_words: List[str]
    participant_id: str
    sentence_id: int
    query_start_idx: int
    query_end_idx: int
    full_sentence_text: str
    full_sentence_words: List[str]
    fs: float


class NieuwlandWordEEGAligner:
    """Nieuwland EEG word-level aligner for ICT pair generation"""

    def __init__(self, data_dir: str, sentence_materials_dir: str = None, random_seed: int = 42,
                 limit_participants: Optional[int] = None, verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.sentence_materials_dir = Path(sentence_materials_dir) if sentence_materials_dir else None
        self.random_seed = random_seed
        self.limit_participants = limit_participants
        self.verbose = verbose

        self._set_seeds(random_seed)

        self.word_display_ms = 200
        self.blank_screen_ms = 300
        self.total_word_duration_ms = 500

        self.participants = []
        self.sentence_mapping = {}
        self.all_word_eeg_pairs = []

        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'random_seed': random_seed,
            'data_dir': str(data_dir),
            'sentence_materials_dir': str(sentence_materials_dir) if sentence_materials_dir else None,
            'word_display_ms': self.word_display_ms,
            'blank_screen_ms': self.blank_screen_ms,
            'total_word_duration_ms': self.total_word_duration_ms,
            'limit_participants': limit_participants,
            'version': 'NIEUWLAND_ICT_v1.0',
            'supports_runtime_masking': True,
            'masking_method': 'runtime_query_span_removal'
        }

        if verbose:
            print(f"Nieuwland EEG Aligner initialized")
            print(f"Random seed: {random_seed}")

        self._scan_participants(limit_participants)

        if self.sentence_materials_dir and self.sentence_materials_dir.exists():
            self._load_sentence_materials()

    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _scan_participants(self, limit_participants: Optional[int] = None):
        if self.verbose:
            print("Scanning for participants...")

        participant_files = defaultdict(dict)

        for file_path in sorted(self.data_dir.glob("seg_*")):
            if file_path.is_file():
                filename = file_path.name
                participant_id = filename[4:].split('.')[0]
                ext = filename.split('.')[-1]
                participant_files[participant_id][ext] = file_path

        participants_added = 0
        for pid in sorted(participant_files.keys()):
            if limit_participants is not None and participants_added >= limit_participants:
                break

            files = participant_files[pid]
            has_dat = 'dat' in files
            has_triggers = any(trigger_type in files for trigger_type in ['vmrk', 'ehst2', 'vhdr'])

            if has_dat and has_triggers:
                trigger_format = None
                if 'vmrk' in files:
                    trigger_format = 'vmrk'
                elif 'ehst2' in files:
                    trigger_format = 'ehst2'
                elif 'vhdr' in files:
                    trigger_format = 'vhdr'

                participant_data = {
                    'id': pid,
                    'files': files,
                    'trigger_format': trigger_format
                }

                self.participants.append(participant_data)
                if self.verbose:
                    print(f"Found participant: {pid} with {trigger_format} triggers")
                participants_added += 1

        if self.verbose:
            print(f"Total participants loaded: {len(self.participants)}")

        if self.participants:
            self.metadata['participants_processed'] = len(self.participants)

    def _load_sentence_materials(self):
        replication_items_path = self.sentence_materials_dir / "REPLICATION_ITEMS.xlsx"
        if replication_items_path.exists():
            try:
                replication_items = pd.read_excel(replication_items_path)
                if self.verbose:
                    print(f"Loaded sentence materials: {replication_items.shape}")

                for idx in sorted(replication_items.index):
                    row = replication_items.iloc[idx]
                    try:
                        item_num = int(row['Item Number'])

                        context = str(row['Sentence context']).strip()
                        expected_article = str(row['Expected']).strip()
                        expected_noun = str(row['Expected.1']).strip()
                        ending = str(row['Sentence Ending']).strip() if pd.notna(row['Sentence Ending']) else ""

                        full_sentence = f"{context} {expected_article} {expected_noun} {ending}".strip()
                        words = full_sentence.split()

                        self.sentence_mapping[item_num] = {
                            'full_sentence': full_sentence,
                            'words': words,
                            'word_count': len(words)
                        }
                    except Exception:
                        continue

                if self.verbose:
                    print(f"Created sentence mapping for {len(self.sentence_mapping)} items")
                self.metadata['sentence_count'] = len(self.sentence_mapping)

            except Exception as e:
                if self.verbose:
                    print(f"Error loading sentence materials: {e}")

    def _extract_sentence_triggers_vmrk(self, vmrk_path: Path) -> List[Tuple[int, int, int]]:
        sentence_triggers = []
        seen_triggers = set()

        try:
            with open(vmrk_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if line.startswith('Mk') and '=' in line:
                        parts = line.split('=')[1].split(',')
                        if len(parts) >= 3:
                            trigger_code = parts[1].strip()
                            sample_pos = int(parts[2]) if parts[2].isdigit() else 0

                            if trigger_code.startswith('S'):
                                try:
                                    code = int(trigger_code[1:])
                                    if 101 <= code <= 180:
                                        item_num = code - 100
                                        if item_num not in seen_triggers:
                                            sentence_triggers.append((sample_pos, code, item_num))
                                            seen_triggers.add(item_num)
                                except ValueError:
                                    continue
        except Exception as e:
            if self.verbose:
                print(f"Error reading .vmrk: {e}")

        return sentence_triggers

    def _extract_sentence_triggers_ehst2(self, ehst2_path: Path) -> List[Tuple[int, int, int]]:
        sentence_triggers = []
        seen_triggers = set()

        try:
            with open(ehst2_path, 'rb') as f:
                raw_data = f.read()

            int16_data = np.frombuffer(raw_data, dtype=np.int16)

            for i, val in enumerate(int16_data):
                if 101 <= val <= 180:
                    sample_pos = i * 2
                    code = int(val)
                    item_num = code - 100
                    if item_num not in seen_triggers:
                        sentence_triggers.append((sample_pos, code, item_num))
                        seen_triggers.add(item_num)

        except Exception as e:
            if self.verbose:
                print(f"Error reading .ehst2: {e}")

        return sentence_triggers

    def _extract_sentence_triggers_vhdr(self, vhdr_path: Path) -> List[Tuple[int, int, int]]:
        sentence_triggers = []
        seen_triggers = set()

        try:
            with open(vhdr_path, 'r', encoding='latin-1') as f:
                content = f.read()

            in_marker_section = False
            for line in content.split('\n'):
                line = line.strip()

                if line.startswith('[Marker Infos]'):
                    in_marker_section = True
                    continue
                elif line.startswith('[') and in_marker_section:
                    break

                if in_marker_section and '=' in line and line.startswith('Mk'):
                    try:
                        parts = line.split('=')[1].split(',')
                        if len(parts) >= 3:
                            trigger_part = parts[1].strip()
                            sample_pos = int(parts[2]) if parts[2].isdigit() else 0

                            if trigger_part.startswith('S'):
                                code = int(trigger_part[1:])
                                if 101 <= code <= 180:
                                    item_num = code - 100
                                    if item_num not in seen_triggers:
                                        sentence_triggers.append((sample_pos, code, item_num))
                                        seen_triggers.add(item_num)
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            if self.verbose:
                print(f"Error reading .vhdr: {e}")

        return sentence_triggers

    def _read_dat_file(self, dat_path: Path) -> Optional[np.ndarray]:
        channel_options = [32, 64, 128, 16, 8]
        dtype_options = [np.int16, np.float32, np.float64]

        for channels in channel_options:
            for dtype in dtype_options:
                try:
                    with open(dat_path, 'rb') as f:
                        data = np.frombuffer(f.read(), dtype=dtype)

                    if len(data) % channels == 0:
                        n_samples = len(data) // channels
                        if n_samples > 1000:
                            eeg_data = data.reshape((n_samples, channels)).T

                            if not np.all(eeg_data == 0) and np.isfinite(eeg_data).all():
                                if self.verbose:
                                    print(f"   EEG data loaded: {eeg_data.shape} ({channels}ch, {dtype})")
                                return eeg_data
                except Exception:
                    continue

        if self.verbose:
            print(f"   Failed to load EEG data")
        return None

    def _calculate_word_timings(self, sentence_start_sample: int, word_count: int, fs: float) -> List[int]:
        word_duration_samples = int(self.total_word_duration_ms * fs / 1000)
        return [sentence_start_sample + (i * word_duration_samples) for i in range(word_count)]

    def _extract_word_eeg(self, eeg_data: np.ndarray, word_start_sample: int,
                          window_ms: int = 500, fs: float = 500.0) -> Optional[np.ndarray]:
        window_samples = int(window_ms * fs / 1000)
        start_sample = max(0, word_start_sample)
        end_sample = min(eeg_data.shape[1], word_start_sample + window_samples)

        if end_sample <= start_sample:
            return None

        word_eeg = eeg_data[:, start_sample:end_sample]
        return word_eeg.T

    def align_all_words_to_eeg(self, max_sentences_per_participant: Optional[int] = None):
        if not self.participants:
            print("No participants found")
            return

        if self.verbose:
            print(f"\nWord-EEG Alignment")
            print("=" * 60)
            print(f"Participants: {len(self.participants)}")

        total_sentences_processed = 0
        total_words_aligned = 0

        for participant_idx, participant in enumerate(self.participants):
            if self.verbose:
                print(f"\nParticipant {participant_idx + 1}/{len(self.participants)}: {participant['id']}")

            fs = 500.0

            trigger_format = participant['trigger_format']
            if trigger_format == 'vmrk':
                sentence_triggers = self._extract_sentence_triggers_vmrk(participant['files']['vmrk'])
            elif trigger_format == 'ehst2':
                sentence_triggers = self._extract_sentence_triggers_ehst2(participant['files']['ehst2'])
            elif trigger_format == 'vhdr':
                sentence_triggers = self._extract_sentence_triggers_vhdr(participant['files']['vhdr'])
            else:
                if self.verbose:
                    print("Unknown trigger format")
                continue

            if self.verbose:
                print(f"Unique sentence triggers found: {len(sentence_triggers)}")

            if len(sentence_triggers) == 0:
                if self.verbose:
                    print("No triggers found - skipping participant")
                continue
            elif len(sentence_triggers) > 100:
                if self.verbose:
                    print(f"Too many triggers ({len(sentence_triggers)}) - check extraction logic")
                continue

            eeg_data = self._read_dat_file(participant['files']['dat'])
            if eeg_data is None:
                if self.verbose:
                    print("Failed to load EEG data")
                continue

            participant_sentences_processed = 0
            participant_words_aligned = 0

            for sample_pos, trigger_code, item_num in sorted(sentence_triggers):
                if max_sentences_per_participant is not None and participant_sentences_processed >= max_sentences_per_participant:
                    break

                if item_num not in self.sentence_mapping:
                    continue

                sentence_info = self.sentence_mapping[item_num]
                full_sentence = sentence_info['full_sentence']
                words = sentence_info['words']
                word_count = sentence_info['word_count']

                word_timings = self._calculate_word_timings(sample_pos, word_count, fs)

                sentence_word_data = []
                for word_idx, (word, word_start_sample) in enumerate(zip(words, word_timings)):
                    word_eeg = self._extract_word_eeg(eeg_data, word_start_sample, window_ms=500, fs=fs)

                    if word_eeg is not None:
                        word_time = word_start_sample / fs

                        word_data = {
                            'word': word,
                            'word_eeg': word_eeg,
                            'word_position': word_idx,
                            'word_time': word_time,
                            'sentence_id': item_num,
                            'full_sentence': full_sentence,
                            'participant_id': participant['id'],
                            'fs': fs
                        }
                        sentence_word_data.append(word_data)
                        participant_words_aligned += 1

                if sentence_word_data:
                    sentence_data = {
                        'sentence_id': item_num,
                        'participant_id': participant['id'],
                        'full_sentence': full_sentence,
                        'words': words,
                        'word_data': sentence_word_data,
                        'fs': fs
                    }
                    self.all_word_eeg_pairs.append(sentence_data)

                participant_sentences_processed += 1
                total_sentences_processed += 1

            total_words_aligned += participant_words_aligned
            if self.verbose:
                print(
                    f"   Participant {participant['id']}: {participant_sentences_processed} sentences, {participant_words_aligned} words")

        self.metadata['processed_sentences'] = total_sentences_processed
        self.metadata['total_word_eeg_pairs'] = total_words_aligned

        if self.verbose:
            print("\n" + "=" * 60)
            print("Word-EEG Alignment Complete")
            print(f"Total Stats:")
            print(f"   Participants processed: {len(self.participants)}")
            print(f"   Total sentences: {total_sentences_processed}")
            print(f"   Total words aligned: {total_words_aligned}")

    def generate_ict_pairs(self, min_query_length: int = 2, max_query_length: int = 50,
                           query_length_ratio: float = 0.3, min_sentence_length: int = 6,
                           use_ratio_based_queries: bool = True, max_pairs_per_sentence: int = 2,
                           max_total_pairs: Optional[int] = None, random_seed: Optional[int] = None) -> List[ICTPair]:

        if random_seed is None:
            random_seed = self.random_seed

        self._set_seeds(random_seed)

        if not self.all_word_eeg_pairs:
            print("No word-EEG data available. Run align_all_words_to_eeg() first.")
            return []

        if self.verbose:
            print(f"Generating ICT pairs from {len(self.all_word_eeg_pairs)} sentences (seed: {random_seed})")

        ict_pairs = []

        generation_params = {
            'min_query_length': min_query_length,
            'max_query_length': max_query_length,
            'query_length_ratio': query_length_ratio,
            'min_sentence_length': min_sentence_length,
            'use_ratio_based_queries': use_ratio_based_queries,
            'max_pairs_per_sentence': max_pairs_per_sentence,
            'max_total_pairs': max_total_pairs,
            'random_seed': random_seed,
            'masking_applied_at_generation': False,
            'supports_runtime_masking': True
        }
        self.metadata['ict_generation_params'] = generation_params

        for sentence_data in tqdm(sorted(self.all_word_eeg_pairs, key=lambda x: x['sentence_id']),
                                  desc="Creating ICT pairs"):
            try:
                words = sentence_data['words']

                if len(words) < min_sentence_length:
                    continue

                sentence_pairs = 0
                attempts = 0
                max_attempts = max_pairs_per_sentence * 5

                while sentence_pairs < max_pairs_per_sentence and attempts < max_attempts:
                    pair = self._create_ict_pair_from_sentence(
                        sentence_data, min_query_length, max_query_length,
                        query_length_ratio, use_ratio_based_queries
                    )

                    if pair is not None:
                        ict_pairs.append(pair)
                        sentence_pairs += 1

                        if max_total_pairs and len(ict_pairs) >= max_total_pairs:
                            break

                    attempts += 1

                if max_total_pairs and len(ict_pairs) >= max_total_pairs:
                    break

            except Exception as e:
                if self.verbose:
                    print(f"Error generating ICT pair: {e}")
                continue

        if self.verbose:
            print(f"Generated {len(ict_pairs)} ICT pairs")
            self._print_ict_statistics(ict_pairs)

        self.metadata['generated_ict_pairs'] = len(ict_pairs)
        return ict_pairs

    def _create_ict_pair_from_sentence(self, sentence_data: Dict, min_query_length: int, max_query_length: int,
                                       query_length_ratio: float, use_ratio_based_queries: bool) -> Optional[ICTPair]:
        words = sentence_data['words']
        word_data = sentence_data['word_data']
        sentence_length = len(words)

        if use_ratio_based_queries:
            query_length = max(min_query_length, int(sentence_length * query_length_ratio))
            query_length = min(query_length, sentence_length - 1)
        else:
            query_length = min(random.randint(min_query_length, max_query_length), sentence_length - 1)

        max_start_idx = sentence_length - query_length
        query_start_idx = random.randint(0, max_start_idx)
        query_end_idx = query_start_idx + query_length

        query_words = words[query_start_idx:query_end_idx]
        query_word_data = word_data[query_start_idx:query_end_idx]
        query_text = ' '.join(query_words)

        query_eegs = []
        for wd in query_word_data:
            if wd['word_eeg'] is not None:
                query_eegs.append(wd['word_eeg'])
            else:
                return None

        if not query_eegs:
            return None

        query_eeg = np.array(query_eegs)

        doc_words = words[:]
        doc_word_data = word_data[:]
        doc_text = ' '.join(doc_words)

        doc_eegs = []
        for wd in doc_word_data:
            if wd['word_eeg'] is not None:
                doc_eegs.append(wd['word_eeg'])

        if not doc_eegs:
            doc_eeg = np.zeros((len(words), query_eeg.shape[1], query_eeg.shape[2]))
        else:
            doc_eeg = np.array(doc_eegs)

        return ICTPair(
            query_text=query_text,
            query_eeg=query_eeg,
            query_words=query_words,
            doc_text=doc_text,
            doc_eeg=doc_eeg,
            doc_words=doc_words,
            participant_id=sentence_data['participant_id'],
            sentence_id=sentence_data['sentence_id'],
            query_start_idx=query_start_idx,
            query_end_idx=query_end_idx,
            full_sentence_text=doc_text,
            full_sentence_words=doc_words,
            fs=sentence_data['fs']
        )

    def _print_ict_statistics(self, ict_pairs: List[ICTPair]):
        if not ict_pairs:
            return

        participants = set(pair.participant_id for pair in ict_pairs)
        query_lengths = [len(pair.query_words) for pair in ict_pairs]
        doc_lengths = [len(pair.doc_words) for pair in ict_pairs]

        print(f"\nICT Pair Statistics:")
        print(f"  Participants: {len(participants)}")
        print(f"  Query length: {np.mean(query_lengths):.1f} +/- {np.std(query_lengths):.1f} words")
        print(f"  Document length: {np.mean(doc_lengths):.1f} +/- {np.std(doc_lengths):.1f} words")

    def save_ict_pairs_with_metadata(self, ict_pairs: List[ICTPair], save_path: str):
        save_path = Path(save_path)

        if self.verbose:
            print(f"Saving {len(ict_pairs)} ICT pairs to {save_path}")

        pairs_data = []
        for pair in ict_pairs:
            pair_dict = {
                'query_text': pair.query_text,
                'query_eeg': pair.query_eeg,
                'query_words': pair.query_words,
                'doc_text': pair.doc_text,
                'doc_eeg': pair.doc_eeg,
                'doc_words': pair.doc_words,
                'participant_id': pair.participant_id,
                'sentence_id': pair.sentence_id,
                'query_start_idx': pair.query_start_idx,
                'query_end_idx': pair.query_end_idx,
                'full_sentence_text': pair.full_sentence_text,
                'full_sentence_words': pair.full_sentence_words,
                'fs': pair.fs
            }
            pairs_data.append(pair_dict)

        dataset = {
            'ict_pairs': pairs_data,
            'metadata': self.metadata,
            'version': '2.0',
            'description': 'Nieuwland EEG-text ICT pairs with runtime masking support'
        }

        if save_path.suffix != '.npy':
            save_path = save_path.with_suffix('.npy')

        np.save(save_path, dataset)

        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        if self.verbose:
            print(f"Saved ICT pairs to: {save_path}")
            print(f"Saved metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate ICT pairs from Nieuwland EEG dataset'
    )

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed segmented data directory')
    parser.add_argument('--sentence_dir', type=str, required=True,
                        help='Path to sentence materials directory')
    parser.add_argument('--output', type=str, default='nieuwland_ict_pairs.npy',
                        help='Output file path (default: nieuwland_ict_pairs.npy)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--limit_participants', type=int, default=None,
                        help='Limit number of participants (default: all)')
    parser.add_argument('--max_sentences_per_participant', type=int, default=None,
                        help='Maximum sentences per participant (default: all)')

    parser.add_argument('--min_query_length', type=int, default=2,
                        help='Minimum query length in words (default: 2)')
    parser.add_argument('--max_query_length', type=int, default=50,
                        help='Maximum query length in words (default: 50)')
    parser.add_argument('--query_ratio', type=float, default=0.3,
                        help='Query length ratio (default: 0.3)')
    parser.add_argument('--min_sentence_length', type=int, default=6,
                        help='Minimum sentence length (default: 6)')
    parser.add_argument('--pairs_per_sentence', type=int, default=2,
                        help='ICT pairs per sentence (default: 2)')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Maximum total pairs (default: all)')

    args = parser.parse_args()

    print("Nieuwland EEG ICT Pair Generation")
    print("=" * 60)

    aligner = NieuwlandWordEEGAligner(
        data_dir=args.data_dir,
        sentence_materials_dir=args.sentence_dir,
        random_seed=args.seed,
        limit_participants=args.limit_participants,
        verbose=True
    )

    print("\nStep 1: Aligning words to EEG...")
    aligner.align_all_words_to_eeg(max_sentences_per_participant=args.max_sentences_per_participant)

    print("\nStep 2: Generating ICT pairs...")
    ict_pairs = aligner.generate_ict_pairs(
        min_query_length=args.min_query_length,
        max_query_length=args.max_query_length,
        query_length_ratio=args.query_ratio,
        min_sentence_length=args.min_sentence_length,
        use_ratio_based_queries=True,
        max_pairs_per_sentence=args.pairs_per_sentence,
        max_total_pairs=args.max_pairs,
        random_seed=args.seed
    )

    if ict_pairs:
        print("\nStep 3: Saving ICT pairs...")
        aligner.save_ict_pairs_with_metadata(ict_pairs, args.output)

        print("\nFinal Results:")
        print("=" * 40)
        participants = set(pair.participant_id for pair in ict_pairs)
        sentences = set(pair.sentence_id for pair in ict_pairs)
        print(f"Participants: {len(participants)}")
        print(f"Unique sentences: {len(sentences)}")
        print(f"Total ICT pairs: {len(ict_pairs)}")
        print(f"Saved to: {args.output}")
    else:
        print("No ICT pairs generated.")


if __name__ == "__main__":
    main()