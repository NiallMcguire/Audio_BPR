# EEG-Text ICT Pair Generation

Scripts for generating Inverse Cloze Task (ICT) pairs from EEG-text datasets with runtime masking support.

## Requirements

Python 3.7 or higher

Install dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

### Alice AudioBook Dataset

**Download**: [https://osf.io/q3zws/](https://osf.io/q3zws/)

**Structure**:
```
alice_data/
├── AliceChapterOne-EEG.csv
└── Subjects/
    ├── sub01.mat
    ├── sub02.mat
    └── ...
```

### Nieuwland Dataset

**Download**: [https://osf.io/eyzaq/](https://osf.io/eyzaq/)

**Structure**:
```
nieuwland_data/
├── Processed_segmented_data/
│   ├── seg_01.dat
│   ├── seg_01.vmrk
│   └── ...
└── Sentence_Materials/
    └── REPLICATION_ITEMS.xlsx
```

## Usage

### Alice Dataset

Basic usage:
```bash
python alice_ict_reader.py \
  --text_path /path/to/AliceChapterOne-EEG.csv \
  --eeg_path /path/to/Subjects \
  --output alice_ict_pairs.npy
```

With custom parameters:
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

### Nieuwland Dataset

Basic usage:
```bash
python nieuwland_ict_reader.py \
  --data_dir /path/to/Processed_segmented_data \
  --sentence_dir /path/to/Sentence_Materials \
  --output nieuwland_ict_pairs.npy
```

With custom parameters:
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

## Command-Line Options

### Common Parameters (both scripts)
- `--seed`: Random seed for reproducibility (default: 42)
- `--min_query_length`: Minimum query length in words (default: 2)
- `--max_query_length`: Maximum query length in words (default: 50)
- `--query_ratio`: Query length as fraction of sentence (default: 0.3)
- `--min_sentence_length`: Minimum sentence length to process (default: 6)
- `--pairs_per_sentence`: ICT pairs generated per sentence (default: 2)
- `--max_pairs`: Maximum total pairs to generate (default: unlimited)

### Alice-Specific Parameters
- `--limit_subjects`: Limit number of subjects (default: all)
- `--target_freq`: Target sampling frequency in Hz (default: 128)
- `--target_channels`: Target number of channels (default: 128)
- `--no_preprocess`: Disable EEG preprocessing

### Nieuwland-Specific Parameters
- `--limit_participants`: Limit number of participants (default: all)
- `--max_sentences_per_participant`: Maximum sentences per participant (default: all)

View all options:
```bash
python alice_ict_reader.py --help
python nieuwland_ict_reader.py --help
```

## Output Format

Each script generates two files:
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



## Data Attribution

- Alice AudioBook: Brennan et al. (2019) - https://osf.io/q3zws/
- Nieuwland: Nieuwland et al. (2018) - https://osf.io/eyzaq/

Please cite the original dataset papers when using these resources.