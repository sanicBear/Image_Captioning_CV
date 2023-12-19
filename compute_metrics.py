import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge import Rouge
from meteor import Meteor

def compute_metrics_for_csv(csv_path):
    # Load the CSV file containing "captions" and "predicted captions" columns
    df = pd.read_csv(csv_path)

    # Extract captions and predicted captions from the DataFrame
    captions = df['captions'].tolist()
    predicted_captions = df['predicted_captions'].tolist()

    # Initialize smoothing function for BLEU
    smoothing = SmoothingFunction().method1

    # Compute BLEU-1 and BLEU-2 scores
    bleu1_scores = [sentence_bleu([caption.split()], predicted_caption.split(), smoothing_function=smoothing, weights=(1, 0)) for caption, predicted_caption in zip(captions, predicted_captions)]
    bleu2_scores = [sentence_bleu([caption.split()], predicted_caption.split(), smoothing_function=smoothing, weights=(0.5, 0.5)) for caption, predicted_caption in zip(captions, predicted_captions)]

    # Compute corpus-level BLEU-1 and BLEU-2 scores
    corpus_bleu1 = corpus_bleu([[caption.split()] for caption in captions], [predicted_caption.split() for predicted_caption in predicted_captions], smoothing_function=smoothing, weights=(1, 0))
    corpus_bleu2 = corpus_bleu([[caption.split()] for caption in captions], [predicted_caption.split() for predicted_caption in predicted_captions], smoothing_function=smoothing, weights=(0.5, 0.5))

    # Initialize ROUGE scorer
    rouge = Rouge()

    # Compute ROUGE scores
    rouge_scores = rouge.get_scores(predicted_captions, captions, avg=True)

    # Initialize METEOR scorer
    meteor = Meteor()

    # Compute METEOR scores
    meteor_scores = [meteor.single_meteor_score(predicted_caption, caption) for predicted_caption, caption in zip(predicted_captions, captions)]
    corpus_meteor = meteor.corpus_meteor_score(captions, predicted_captions)

    metrics = {
        "BLEU-1": {
            "sentence_scores": bleu1_scores,
            "corpus_score": corpus_bleu1
        },
        "BLEU-2": {
            "sentence_scores": bleu2_scores,
            "corpus_score": corpus_bleu2
        },
        "ROUGE": rouge_scores,
        "METEOR": {
            "sentence_scores": meteor_scores,
            "corpus_score": corpus_meteor
        }
    }

    return metrics

# Example usage:
csv_path = 'filter_test.csv'
metrics = compute_metrics_for_csv(csv_path)
print(metrics)
