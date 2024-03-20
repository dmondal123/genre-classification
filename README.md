# Music Genre Classification using MIDI Embeddings

## Overview
This project focuses on classifying music genres using MIDI embeddings generated from the Lakh MIDI Dataset (lmd-matched, match scores and lmd-aligned) which can be downloaded from [here](https://colinraffel.com/projects/lmd/). The data needs to be stored in a Data folder. The goal is to develop machine learning models capable of accurately categorizing musical compositions into various genres based on their MIDI representations. The project explores different model architectures and evaluates their performance using various evaluation metrics.

## Problem Statement
The task involves classifying music genres based on MIDI embeddings, which are numerical representations of musical compositions. The problem is important for applications such as personalized music recommendation systems and organizing music libraries, where accurate genre classification is crucial for enhancing user experiences.

## Dataset Description
- Original Dataset: Lakh MIDI Dataset
- Initial Statistics: 485 unique genres, 31,000+ songs
- Sampling Technique: SMOTE
- Final Dataset: Balanced dataset with genres present more than 150 times
- No. of genres in final dataset: 21
- No. of data points in final dataset: 29295

## Data Preprocessing
- MIDI Embeddings: Generated using MIDI2Vec
- One-hot Encoding: Representing genres for each song
- Data Balancing: SMOTE for imbalanced classes

## Model Architectures Explored
1. Simple LSTMs
2. Bidirectional LSTMs with Dropout Layers
3. Bidirectional LSTMs with Batch Normalization

## Evaluation Metrics
- Precision, Recall, F1 Score, AUC-ROC
- Interpretation and importance in music genre classification

## Key Findings
- Best performing model: Bidirectional LSTMs with Batch Normalization (Test Precision: 88%, Recall: 86.9%, F1 Score: 87.4%, AUC-ROC: 98.2%)
- Limitations: Dataset size and diversity, model complexity, overfitting
- Future Work: Dataset expansion, alternative data representations, model optimization, ensemble learning

## Repository Structure
- `data/`: Contains datasets used in the project.
- `models/`: Saved model files.
- `logs/`: Training logs.
- `notebooks/`: Jupyter notebooks for data preprocessing, model training, and evaluation.
- `src/` : Contains the functions for loading and cleaning the data, model architectue and model train files, and some miscallaneous functions.
- `README.md`: Overview of the project and instructions for running the code.
- `requirements.txt`: List of dependencies required to run the code.

## Usage
1. Install the project dependencies:
```pip install genreclassification```

2. Clone the repository:
```git clone https://github.com/dmondal123/genre-classification.git```

## For generating MIDI embeddings

1. Clone the [MIDI2Vec](https://github.com/midi-ld/midi2vec) repository and enter in it.
2. Generate edgelists (absolute paths to prefer)
```
cd midi2edgelist
npm install
node index.js -i <dataset_folder>
node index.js -i <dataset_folder> -o edgelist_300 -n 300
```
3. Compute embeddings
```
cd ../
pip install -r edgelist2vec/requirements.txt

python edgelist2vec/embed.py -o embeddings/<dataset>.bin
python edgelist2vec/embed.py -o embeddings/<dataset>_notes.bin --exclude notes
python edgelist2vec/embed.py -o embeddings/<dataset>_program.bin --exclude program
python edgelist2vec/embed.py -o embeddings/<dataset>_tempo.bin --exclude tempo
python edgelist2vec/embed.py -o embeddings/<dataset>_timesig.bin --exclude time.signature
python edgelist2vec/embed.py -i edgelist_300 -o embeddings/<dataset>_300.bin
```




