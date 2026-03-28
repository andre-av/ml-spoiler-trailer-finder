# Movie Trailer Spoiler Detector

An ML-powered tool that analyzes YouTube movie trailer comments to detect spoilers, helping viewers decide whether it's safe to watch the trailer.

## How It Works

1. Paste a YouTube trailer URL
2. The system scrapes the top comments
3. A fine-tuned DistilBERT model classifies each comment as spoiler or non-spoiler
4. You get an overall spoiler risk score and a list of flagged comments

## Project Structure

```
├── notebooks/                  # Jupyter notebooks for exploration and training
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb
├── src/                        # Reusable Python modules
│   ├── data/preprocessing.py   # Text cleaning and tokenization
│   ├── model/                  # Training and inference code
│   └── scraper/                # YouTube comment fetching
├── app/                        # Streamlit web application
├── data/                       # Local data storage (not tracked by git)
└── models/                     # Saved model weights (not tracked by git)
```

## Setup

```bash
# Clone the repository
git clone https://github.com/andre-av/ml-spoiler-trailer-finder.git
cd ml-spoiler-trailer-finder

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the [IMDB Spoiler Dataset](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset) by Rishabh Misra for training. The dataset contains ~573K movie reviews with binary spoiler labels, which the model uses to learn general spoiler language patterns that transfer to YouTube comments.

## Model Performance

_Results will be added after training._

## Future Work

- Trailer transcript analysis via speech-to-text
- Visual trailer analysis using multimodal models (CLIP)
- Fine-tuning on YouTube-specific comment data

## License

MIT
