#!/usr/bin/env python3
"""
Alice AudioBook EEG ICT Pair Generation

Generates ICT (Inverse Cloze Task) pairs from Alice AudioBook EEG dataset
with runtime masking support for multi-level validation.
"""

import random
import argparse
from datetime import datetime
import torch
from scipy import signal
import scipy.io as sio
import numpy as np
import csv
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

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


class Preprocess:
    """EEG preprocessing pipeline"""

    def __init__(self, target_freq=128, target_channels=128, Downsample=True, Padding=True,
                 Bandpass=True, AverageRef=True, low_freq=0.5, high_freq=100):
        self.target_freq = target_freq
        self.target_channels = target_channels
        self.Downsample = Downsample
        self.Padding = Padding
        self.Bandpass = Bandpass
        self.AverageRef = AverageRef
        self.low_freq = low_freq
        self.high_freq = high_freq

    def downsample(self, eeg_data, original_freq):
        if original_freq == self.target_freq:
            return eeg_data
        num_samples = int(len(eeg_data) * self.target_freq / original_freq)
        return signal.resample(eeg_data, num_samples)

    def pad_channels(self, eeg_data):
        current_channels = eeg_data.shape[1]
        if current_channels < self.target_channels:
            padding = np.zeros((eeg_data.shape[0], self.target_channels - current_channels))
            return np.hstack((eeg_data, padding))
        return eeg_data

    def bandpass_filter(self, eeg_data, fs):
        nyq = 0.5 * fs
        low = self.low_freq / nyq
        high = self.high_freq / nyq
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, eeg_data, axis=0)

    def average_reference(self, eeg_data):
        return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

    def process(self, eeg_data, time_column, original_freq):
        if self.Downsample:
            eeg_data = self.downsample(eeg_data, original_freq)
            time_column = self.downsample(time_column, original_freq)
            current_freq = self.target_freq
        else:
            current_freq = original_freq

        if self.Bandpass:
            eeg_data = self.bandpass_filter(eeg_data, current_freq)
        if self.AverageRef:
            eeg_data = self.average_reference(eeg_data)
        if self.Padding:
            eeg_data = self.pad_channels(eeg_data)

        return eeg_data, time_column


class AliceAudioBookReader:
    """Alice AudioBook EEG reader for ICT pair generation"""

    def __init__(self, text_path: str, eeg_base_path: str, preprocess: bool = False,
                 target_freq=128, target_channels=128, Downsample=True, Padding=True,
                 Bandpass=True, AverageRef=True, low_freq=0.5, high_freq=100,
                 random_seed: int = 42, limit_subjects: Optional[int] = None, verbose: bool = True):
        self.text_path = text_path
        self.eeg_base_path = eeg_base_path
        self.eeg_data = None
        self.time_column = None
        self.preprocess = preprocess
        self.random_seed = random_seed
        self.limit_subjects = limit_subjects
        self.verbose = verbose

        self._set_seeds(random_seed)

        if preprocess:
            self.preprocessor = Preprocess(target_freq, target_channels, Downsample, Padding,
                                           Bandpass, AverageRef, low_freq, high_freq)
        else:
            self.preprocessor = None

        self.all_sentences = []

        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'random_seed': random_seed,
            'text_path': str(text_path),
            'eeg_base_path': str(eeg_base_path),
            'target_freq': target_freq,
            'target_channels': target_channels,
            'preprocess': preprocess,
            'limit_subjects': limit_subjects,
            'version': 'ALICE_ICT_v1.0',
            'supports_runtime_masking': True,
            'masking_method': 'runtime_query_span_removal'
        }

        if verbose:
            print(f"Alice AudioBook EEG Reader initialized")
            print(f"Random seed: {random_seed}")
            if limit_subjects is None:
                print("Processing all subjects")
            else:
                print(f"Processing first {limit_subjects} subjects")

    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_eeg_data(self, eeg_file):
        if self.verbose:
            print(f"Loading EEG data from {eeg_file}")
        data = sio.loadmat(eeg_file, struct_as_record=False, squeeze_me=True)
        raw_data = data['raw']
        self.eeg_data = raw_data.trial.T
        self.time_column = raw_data.time

        if self.preprocess:
            original_freq = raw_data.fsample
            self.eeg_data, self.time_column = self.preprocessor.process(
                self.eeg_data, self.time_column, original_freq)

        return f"EEG data loaded. Shape: {self.eeg_data.shape}"

    def process_alice_csv(self) -> Tuple[List[List[Tuple[str, float, float]]], float]:
        sentences = defaultdict(list)
        segments = defaultdict(list)
        current_segment = 1
        segment_end_times = [0]

        with open(self.text_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                word = row['Word']
                segment = int(row['Segment'])
                onset = float(row['onset'])
                offset = float(row['offset'])
                sentence_num = int(row['Sentence'])

                if segment != current_segment:
                    if segments[current_segment]:
                        segment_end_times.append(
                            segment_end_times[-1] + max(offset for _, _, offset, _ in segments[current_segment]))
                    else:
                        segment_end_times.append(segment_end_times[-1])
                    current_segment = segment

                segments[segment].append((word, onset, offset, sentence_num))

        if segments[current_segment]:
            segment_end_times.append(
                segment_end_times[-1] + max(offset for _, _, offset, _ in segments[current_segment]))

        for segment, words in segments.items():
            segment_start = segment_end_times[segment - 1]
            for word, onset, offset, sentence_num in words:
                adjusted_onset = segment_start + onset
                adjusted_offset = segment_start + offset
                sentences[sentence_num].append((word, adjusted_onset, adjusted_offset))

        sentence_list = [sentences[i] for i in range(1, len(sentences) + 1)]
        return sentence_list, segment_end_times[-1]

    def segment_eeg(self, sentences: List[List[Tuple[str, float, float]]]) -> List[Dict]:
        segmented_data = []
        for sentence in sentences:
            if not sentence:
                continue
            start_time, end_time = sentence[0][1], sentence[-1][2]
            start_index = np.searchsorted(self.time_column, start_time)
            end_index = np.searchsorted(self.time_column, end_time)
            eeg_segment = self.eeg_data[start_index:end_index + 1]

            word_eeg_pairs = []
            for word, onset, offset in sentence:
                word_start_index = np.searchsorted(self.time_column, onset)
                word_end_index = np.searchsorted(self.time_column, offset)
                word_eeg = self.eeg_data[word_start_index:word_end_index + 1]
                word_eeg_pairs.append((word, word_eeg))

            segmented_data.append({
                'eeg': eeg_segment,
                'words': [word for word, _, _ in sentence],
                'word_onsets': [onset for _, onset, _ in sentence],
                'word_offsets': [offset for _, _, offset in sentence],
                'start_time': start_time,
                'end_time': end_time,
                'word_eeg_pairs': word_eeg_pairs
            })
        return segmented_data

    def process_subject(self, subject_file: str, subject_id: str):
        self.load_eeg_data(subject_file)
        alice_sentences, total_duration = self.process_alice_csv()
        segmented_data = self.segment_eeg(alice_sentences)

        if self.verbose:
            print(f"Subject {subject_id}: {len(segmented_data)} sentences, duration: {total_duration:.2f}s")

        return segmented_data, alice_sentences, total_duration

    def read_sentences_reproducibly(self) -> List[Dict]:
        sentences = []
        subject_files = sorted([f for f in os.listdir(self.eeg_base_path) if f.endswith('.mat')])

        if self.limit_subjects is not None:
            subject_files = subject_files[:self.limit_subjects]

        subjects_processed = 0
        for subject_file in tqdm(subject_files, desc="Processing subjects"):
            subject_path = os.path.join(self.eeg_base_path, subject_file)
            subject_id = subject_file.split('.')[0]

            try:
                segmented_data, alice_sentences, total_duration = self.process_subject(subject_path, subject_id)

                for sentence_idx, sentence_data in enumerate(segmented_data):
                    words = sentence_data['words']
                    word_eeg_pairs = sentence_data['word_eeg_pairs']

                    if len(words) > 0:
                        word_eegs = []
                        for word, word_eeg in word_eeg_pairs:
                            if word_eeg.shape[0] > 0:
                                word_eegs.append(word_eeg)
                            else:
                                dummy_eeg = np.zeros((10, sentence_data['eeg'].shape[1]))
                                word_eegs.append(dummy_eeg)

                        if len(word_eegs) == len(words):
                            sentence_info = {
                                'text': ' '.join(words),
                                'eeg': sentence_data['eeg'],
                                'words': words,
                                'word_eegs': word_eegs,
                                'word_timings': list(zip(
                                    sentence_data['word_onsets'],
                                    sentence_data['word_offsets']
                                )),
                                'subject_id': subject_id,
                                'sentence_idx': sentence_idx,
                                'fs': self.preprocessor.target_freq if self.preprocess else 128,
                                'start_time': sentence_data['start_time'],
                                'end_time': sentence_data['end_time'],
                                'unique_sentence_id': f"{subject_id}_{sentence_idx}"
                            }
                            sentences.append(sentence_info)

                subjects_processed += 1

            except Exception as e:
                if self.verbose:
                    print(f"Error processing {subject_file}: {e}")
                continue

        self.all_sentences = sentences
        self.metadata['total_sentences'] = len(sentences)
        self.metadata['subjects_processed'] = subjects_processed

        if self.verbose:
            print(f"Extracted {len(sentences)} sentences from Alice dataset")
            subjects = set(s['subject_id'] for s in sentences)
            print(f"Found {len(subjects)} subjects")

        return sentences

    def generate_ict_pairs(self, min_query_length: int = 2, max_query_length: int = 50,
                           query_length_ratio: float = 0.3, min_sentence_length: int = 6,
                           use_ratio_based_queries: bool = True, max_pairs_per_sentence: int = 2,
                           max_total_pairs: Optional[int] = None, random_seed: Optional[int] = None) -> List[ICTPair]:

        if random_seed is None:
            random_seed = self.random_seed

        self._set_seeds(random_seed)

        if not self.all_sentences:
            print("No sentence data available. Run read_sentences_reproducibly() first.")
            return []

        if self.verbose:
            print(f"Generating ICT pairs from {len(self.all_sentences)} sentences (seed: {random_seed})")

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

        for sentence_data in tqdm(sorted(self.all_sentences, key=lambda x: x['unique_sentence_id']),
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

    def _create_ict_pair_from_sentence(self, sentence_data: Dict, min_query_length: int,
                                       max_query_length: int, query_length_ratio: float,
                                       use_ratio_based_queries: bool) -> Optional[ICTPair]:
        words = sentence_data['words']
        word_eegs = sentence_data['word_eegs']
        sentence_length = len(words)

        if use_ratio_based_queries:
            query_length = max(min_query_length, int(sentence_length * query_length_ratio))
            query_length = min(query_length, sentence_length - 1)
            query_length = min(query_length, max_query_length)
        else:
            query_length = min(
                random.randint(min_query_length, max_query_length),
                sentence_length - 1
            )

        max_start_idx = sentence_length - query_length
        query_start_idx = random.randint(0, max_start_idx)
        query_end_idx = query_start_idx + query_length

        query_words = words[query_start_idx:query_end_idx]
        query_text = ' '.join(query_words)

        query_word_eegs = word_eegs[query_start_idx:query_end_idx]
        if not query_word_eegs or any(eeg.shape[0] == 0 for eeg in query_word_eegs):
            return None

        try:
            max_samples = max(eeg.shape[0] for eeg in query_word_eegs)
            channels = query_word_eegs[0].shape[1]

            padded_query_eegs = []
            for eeg in query_word_eegs:
                if eeg.shape[0] < max_samples:
                    padding = np.zeros((max_samples - eeg.shape[0], channels))
                    padded_eeg = np.vstack([eeg, padding])
                else:
                    padded_eeg = eeg[:max_samples]
                padded_query_eegs.append(padded_eeg)

            query_eeg = np.array(padded_query_eegs)
        except Exception:
            return None

        doc_words = words[:]
        doc_word_eegs = word_eegs[:]
        doc_text = ' '.join(doc_words)

        if doc_word_eegs:
            try:
                max_samples = max(eeg.shape[0] for eeg in doc_word_eegs)
                channels = doc_word_eegs[0].shape[1]

                padded_doc_eegs = []
                for eeg in doc_word_eegs:
                    if eeg.shape[0] < max_samples:
                        padding = np.zeros((max_samples - eeg.shape[0], channels))
                        padded_eeg = np.vstack([eeg, padding])
                    else:
                        padded_eeg = eeg[:max_samples]
                    padded_doc_eegs.append(padded_eeg)

                doc_eeg = np.array(padded_doc_eegs)
            except Exception:
                doc_eeg = np.zeros((len(words), query_eeg.shape[1], query_eeg.shape[2]))
        else:
            doc_eeg = np.zeros((len(words), query_eeg.shape[1], query_eeg.shape[2]))

        return ICTPair(
            query_text=query_text,
            query_eeg=query_eeg,
            query_words=query_words,
            doc_text=doc_text,
            doc_eeg=doc_eeg,
            doc_words=doc_words,
            participant_id=sentence_data['subject_id'],
            sentence_id=int(sentence_data['sentence_idx']),
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
            'description': 'Alice AudioBook EEG-text ICT pairs with runtime masking support'
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
        description='Generate ICT pairs from Alice AudioBook EEG dataset'
    )

    parser.add_argument('--text_path', type=str, required=True,
                        help='Path to AliceChapterOne-EEG.csv')
    parser.add_argument('--eeg_path', type=str, required=True,
                        help='Path to Subjects directory')
    parser.add_argument('--output', type=str, default='alice_ict_pairs.npy',
                        help='Output file path (default: alice_ict_pairs.npy)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--limit_subjects', type=int, default=None,
                        help='Limit number of subjects (default: all)')

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

    parser.add_argument('--no_preprocess', action='store_true',
                        help='Disable EEG preprocessing')
    parser.add_argument('--target_freq', type=int, default=128,
                        help='Target frequency for downsampling (default: 128)')
    parser.add_argument('--target_channels', type=int, default=128,
                        help='Target number of channels (default: 128)')

    args = parser.parse_args()

    print("Alice AudioBook ICT Pair Generation")
    print("=" * 60)

    reader = AliceAudioBookReader(
        text_path=args.text_path,
        eeg_base_path=args.eeg_path,
        preprocess=not args.no_preprocess,
        target_freq=args.target_freq,
        target_channels=args.target_channels,
        Downsample=True,
        Padding=True,
        Bandpass=True,
        AverageRef=True,
        low_freq=0.5,
        high_freq=100,
        random_seed=args.seed,
        limit_subjects=args.limit_subjects,
        verbose=True
    )

    print("\nStep 1: Reading sentences from dataset...")
    sentences = reader.read_sentences_reproducibly()

    if not sentences:
        print("No sentences found. Check data paths.")
        return

    print("\nStep 2: Generating ICT pairs...")
    ict_pairs = reader.generate_ict_pairs(
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
        reader.save_ict_pairs_with_metadata(ict_pairs, args.output)

        print("\nFinal Results:")
        print("=" * 40)
        participants = set(pair.participant_id for pair in ict_pairs)
        sentences_used = set(pair.sentence_id for pair in ict_pairs)
        print(f"Participants: {len(participants)}")
        print(f"Unique sentences: {len(sentences_used)}")
        print(f"Total ICT pairs: {len(ict_pairs)}")
        print(f"Saved to: {args.output}")
    else:
        print("No ICT pairs generated.")


if __name__ == "__main__":
    main()