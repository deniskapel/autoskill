import json
import requests
from tqdm import tqdm

with open('data/predictions/merged_preds.json', 'r', encoding='utf8') as f:
    predictions = json.load(f)

# score_map = {"isResponseComprehensible": {},
#              "isResponseErroneous": {},
#              "isResponseInteresting": {},
#              "isResponseOnTopic": {},
#              "responseEngagesUser": {}}

scored_preds = list()

for sample in tqdm(predictions):
    scored = {
        'context': sample['context'],
        'preds': sample['preds'],
        'scores': {}
    }

    algos = list(sample['preds'].keys())

    context = sample['context'].split(" __eou__ ")
    hyps = [pred.replace("  ", " ") for pred in sample['preds'].values()]
    # hyps = list(set(hyps))

    request_data = {
        "hypotheses": hyps,
        "currentUtterance": context[-1],
        "pastResponses": context[1:-1],
        "pastUtterances": context[0:1],
    }

    result = requests.post("http://0.0.0.0:8004/batch_model", json=request_data).json()[0]["batch"]

    for algo, scores in zip(algos, result):
        scores = {metric: round(score, 3) for metric, score in scores.items()}
        scored['scores'][algo] = scores

    scored_preds.append(scored)

with open('data/predictions/scored_merged_preds.json', "w", encoding="UTF-8") as f:
    json.dump(scored_preds, f, ensure_ascii=False, indent=2)
