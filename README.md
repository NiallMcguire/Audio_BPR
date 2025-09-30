# EEG-Text Brain Passage Retrieval

Scripts for generating Inverse Cloze Task (ICT) pairs from EEG-text datasets and training dual-encoder Brain Passage Retrieval models across auditory and visual modalities.

## Requirements

Python 3.7 or higher

Install dependencies:
```bash
pip install -r requirements.txt
```


## Datasets

### Alice AudioBook Dataset (Auditory EEG)

Download: https://openneuro.org/datasets/ds002322


### Nieuwland Dataset (Visual EEG)

Download: https://osf.io/eyzaq/


## Reproducing Paper Experiments

### Step 1: Generate ICT Pairs

Generate ICT pairs for both datasets using the paper's configuration.

**Alice Dataset (Auditory):**
```bash
python alice_ict_reader.py \
  --text_path /path/to/AliceChapterOne-EEG.csv \
  --eeg_path /path/to/Subjects \
  --output alice_ict_pairs.npy \
  --seed 42 \
  --min_query_length 2 \
  --max_query_length 50 \
  --query_ratio 0.3 \
  --pairs_per_sentence 2
```

**Nieuwland Dataset (Visual):**
```bash
python nieuwland_ict_reader.py \
  --data_dir /path/to/Processed_segmented_data \
  --sentence_dir /path/to/Sentence_Materials \
  --output nieuwland_ict_pairs.npy \
  --seed 42 \
  --min_query_length 2 \
  --max_query_length 50 \
  --query_ratio 0.3 \
  --pairs_per_sentence 2
```

### Step 2: Train Models

The paper evaluates three training regimes (individual Alice, individual Nieuwland, combined) across four pooling strategies (CLS, MEAN, MAX, MULTI).

#### Individual Training - Alice (Auditory)

**CLS Pooling:**
```bash
python controller.py \
  --data_path alice_ict_pairs.npy \
  --pooling_strategy cls \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/alice_cls
```

**MEAN Pooling:**
```bash
python controller.py \
  --data_path alice_ict_pairs.npy \
  --pooling_strategy mean \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/alice_mean
```

**MAX Pooling:**
```bash
python controller.py \
  --data_path alice_ict_pairs.npy \
  --pooling_strategy max \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/alice_max
```

**MULTI Pooling:**
```bash
python controller.py \
  --data_path alice_ict_pairs.npy \
  --pooling_strategy multi \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/alice_multi
```

#### Individual Training - Nieuwland (Visual)

**CLS Pooling:**
```bash
python controller.py \
  --data_path nieuwland_ict_pairs.npy \
  --pooling_strategy cls \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/nieuwland_cls
```

**MEAN Pooling:**
```bash
python controller.py \
  --data_path nieuwland_ict_pairs.npy \
  --pooling_strategy mean \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/nieuwland_mean
```

**MAX Pooling:**
```bash
python controller.py \
  --data_path nieuwland_ict_pairs.npy \
  --pooling_strategy max \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/nieuwland_max
```

**MULTI Pooling:**
```bash
python controller.py \
  --data_path nieuwland_ict_pairs.npy \
  --pooling_strategy multi \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/nieuwland_multi
```

#### Combined Cross-Sensory Training

**CLS Pooling:**
```bash
python controller.py \
  --data_paths alice_ict_pairs.npy nieuwland_ict_pairs.npy \
  --pooling_strategy cls \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/combined_cls
```

**MEAN Pooling:**
```bash
python controller.py \
  --data_paths alice_ict_pairs.npy nieuwland_ict_pairs.npy \
  --pooling_strategy mean \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/combined_mean
```

**MAX Pooling:**
```bash
python controller.py \
  --data_paths alice_ict_pairs.npy nieuwland_ict_pairs.npy \
  --pooling_strategy max \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/combined_max
```

**MULTI Pooling:**
```bash
python controller.py \
  --data_paths alice_ict_pairs.npy nieuwland_ict_pairs.npy \
  --pooling_strategy multi \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/combined_multi
```

### Step 3: Evaluation

Models are automatically evaluated during training across all masking levels (0%, 25%, 50%, 75%, 90%, 100%). Results are saved in the specified `--output_dir` including:

- `model_<pooling>_<arch>.pt`: Trained model checkpoint
- `experiment_config.json`: Full experiment configuration
- Console output with validation and test metrics (MRR, Hit@1, Hit@5, Hit@10)


## Command-Line Options

### ICT Generation Parameters

**Common (both scripts):**
- `--seed`: Random seed for reproducibility (default: 42)
- `--min_query_length`: Minimum query length in words (default: 2)
- `--max_query_length`: Maximum query length in words (default: 50)
- `--query_ratio`: Query length as fraction of sentence (default: 0.3)
- `--min_sentence_length`: Minimum sentence length to process (default: 6)
- `--pairs_per_sentence`: ICT pairs generated per sentence (default: 2)
- `--max_pairs`: Maximum total pairs to generate (default: unlimited)

**Alice-specific:**
- `--limit_subjects`: Limit number of subjects (default: all)
- `--target_freq`: Target sampling frequency in Hz (default: 128)
- `--target_channels`: Target number of channels (default: 128)
- `--no_preprocess`: Disable EEG preprocessing

**Nieuwland-specific:**
- `--limit_participants`: Limit number of participants (default: all)
- `--max_sentences_per_participant`: Maximum sentences per participant (default: all)

### Model Training Parameters

- `--data_path`: Path to single ICT pairs .npy file
- `--data_paths`: Paths to multiple ICT pairs .npy files (for combined training)
- `--pooling_strategy`: Semantic aggregation (choices: cls, mean, max, multi)
- `--encoder_type`: Encoder architecture (choices: dual, cross; default: dual)
- `--query_type`: Query type (choices: eeg, text; default: eeg)
- `--batch_size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 1e-5)
- `--epochs`: Maximum training epochs (default: 200)
- `--patience`: Early stopping patience (default: 10)
- `--holdout_subjects`: Use subject-based train/val/test split
- `--training_masking_level`: Masking probability during training (default: 90)
- `--enable_multi_masking_validation`: Enable multi-level masking validation
- `--validation_masking_levels`: Masking levels to evaluate (default: 0 25 50 75 90 100)
- `--seed`: Random seed (default: 42)
- `--output_dir`: Output directory for results

View all options:
```bash
python alice_ict_reader.py --help
python nieuwland_ict_reader.py --help
python controller.py --help
```

## Output Format

### ICT Pairs

Each ICT generation script produces:
- `.npy` file: ICT pairs with EEG data
- `.json` file: Metadata

Load the data:
```python
import numpy as np
data = np.load("alice_ict_pairs.npy", allow_pickle=True).item()
ict_pairs = data['ict_pairs']
metadata = data['metadata']

# Each ICT pair contains:
# - query_text, query_eeg, query_words
# - doc_text, doc_eeg, doc_words (unmasked full sentence)
# - query_start_idx, query_end_idx (for runtime masking)
# - participant_id, sentence_id, fs
```

### Model Outputs

Training produces:
- `model_<pooling>_<arch>.pt`: Trained model checkpoint
- `experiment_config.json`: Full configuration
- Console output with validation/test metrics (MRR, Hit@1, Hit@5, Hit@10)

## Data Attribution

- **Alice AudioBook**: Brennan et al. (2019) - https://openneuro.org/datasets/ds002322
- **Nieuwland**: Nieuwland et al. (2018) - https://osf.io/eyzaq/

Please cite the original dataset papers when using these resources.

