import json

with open('./data/QUEST/predictions_Quest_BM25.jsonl', 'r') as read_file, open('QUEST/test.jsonl') as gt, open('QUEST/predictions_Quest_BM25_converted.jsonl', 'w') as write_file:
    for (pred, true) in zip(read_file, gt):
        data_pred = json.loads(pred)
        data_true = json.loads(true)

        new_data = {}
        if data_pred['query'] != data_true['query']:
            print("Queries do not match!")
            break

        new_data['query'] = data_pred['query']
        new_data['relevant'] = data_true['docs']
        retrieved = []
        for rank, (score, doc) in enumerate(sorted(zip(data_pred['scores'], data_pred['docs']), reverse=True)):
            retrieved.append({'rank': rank+1, 'score': score, 'title': doc})
        new_data['retrieved'] = retrieved

        write_file.write(json.dumps(new_data) + '\n')


with open('./data/QUEST_VAR/predictions_Quest_var_BM25.jsonl', 'r') as read_file, open('QUEST_VAR/test.jsonl') as gt, open('QUEST_VAR/predictions_Quest_var_BM25_converted.jsonl', 'w') as write_file:
    for (pred, true) in zip(read_file, gt):
        data_pred = json.loads(pred)
        data_true = json.loads(true)

        new_data = {}
        if data_pred['query'] != data_true['query']:
            print("Queries do not match!")
            break

        new_data['query'] = data_pred['query']
        new_data['relevant'] = data_true['docs']
        retrieved = []
        for rank, (score, doc) in enumerate(sorted(zip(data_pred['scores'], data_pred['docs']), reverse=True)):
            retrieved.append({'rank': rank+1, 'score': score, 'title': doc})
        new_data['retrieved'] = retrieved

        write_file.write(json.dumps(new_data) + '\n')