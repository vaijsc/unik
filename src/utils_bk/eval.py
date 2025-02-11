from rouge_score import rouge_scorer

def calc_rouge(predictions, references):
    """
    input: 
        predictions (list): list of predicted summary
        references (list(list)): list of ground truths
    output (dict):
        rouge1, rouge2, rougeL
    """
    rouge_type = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_type, use_stemmer=True)
    all_r1_scores = []
    all_r2_scores = []
    all_rL_scores = []
    for prediction, reference in zip(predictions, references):
        score = scorer.score_multi(reference, prediction)
        all_r1_scores.append(score['rouge1'].fmeasure)
        all_r2_scores.append(score['rouge2'].fmeasure)
        all_rL_scores.append(score['rougeL'].fmeasure)

    # print(all_rL_scores)
    mean_r1 = sum(all_r1_scores) / len(all_r1_scores)
    mean_r2 = sum(all_r2_scores) / len(all_r2_scores)
    mean_rL = sum(all_rL_scores) / len(all_rL_scores)
    return {
        'rouge1': mean_r1,
        'rouge2': mean_r2,
        'rougeL': mean_rL,
    } 


if __name__ == "__main__":
    pass  
