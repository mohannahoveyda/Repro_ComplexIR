import json
import os
import subprocess
import argparse


def make_doc_mappings(file_path, mapping_path):
    doc_mapping = {}
    with open(file_path, 'r') as read_file, open(mapping_path, 'w') as write_file:
        for line in read_file:
            data = json.loads(line)
            if data['title'] not in doc_mapping.keys():
                doc_mapping[data['title']] = len(doc_mapping)
        json.dump(doc_mapping, write_file, indent=4)


def make_query_mappings(file_path, mapping_path):
    query_mapping = {}
    with open(file_path, 'r') as read_file, open(mapping_path, 'w') as write_file:
        for line in read_file:
            data = json.loads(line)
            query_mapping[data['query']] = len(query_mapping)
        json.dump(query_mapping, write_file, indent=4)


def make_qrels(test_path, qrel_path, query_mapping, doc_mapping):
    with open(test_path, 'r') as f, open(qrel_path, 'w') as write_file:
        for item in f:
            data = json.loads(item)
            query_id = query_mapping[data['query']]
            docs = data['docs']
            for doc in docs:
                score = 1
                doc_id = doc_mapping[doc]
                write_file.write(f'{query_id}   1   {doc_id}   {score}\n')


def format_results(result_path, formatted_path, query_mapping, doc_mapping):
    with open(result_path, 'r') as read_file, open(formatted_path, 'w') as write_file:
        removed = 0
        for line in read_file:
            data = json.loads(line)
            query_id = query_mapping[data['query']]
            results = data['retrieved']
            docs_seen = []
            for res in results:
                doc_id = doc_mapping[res['title']]
                if doc_id in docs_seen:
                    removed += 1
                    continue
                else:
                    docs_seen.append(doc_id)
                    score = res['score']
                    rank = res['rank']
                    write_file.write(f'{query_id} Q0 {doc_id} {rank} {score} BM25\n')
        if removed > 0:
            print(f'Removed {removed} duplicate documents from results.')


def preprocess(results_file, folder_path):
    # make document mappings to represent documents as integers
    documents_path = folder_path + 'documents.jsonl'
    doc_mapping_path = folder_path + 'doc_mapping.json'
    if not os.path.exists(doc_mapping_path):
        print('Making document mappings...')
        make_doc_mappings(documents_path, doc_mapping_path)
    doc_mapping = json.load(open(doc_mapping_path, 'r'))

    # make query mappings to represent queries as integers
    test_path = folder_path + 'test.jsonl'
    query_mapping_path = folder_path + 'query_mapping.json'
    if not os.path.exists(query_mapping_path):
        print('Making query mappings...')
        make_query_mappings(test_path, query_mapping_path)
    query_mapping = json.load(open(query_mapping_path, 'r'))

    # make qrels in TREC format
    qrel_path = folder_path + 'qrels.test'
    if not os.path.exists(qrel_path):
        print('Making qrels...')
        make_qrels(test_path, qrel_path, query_mapping, doc_mapping)

    # convert result file to TREC format
    print('Formatting result file...')
    formatted_path = results_file.replace('.jsonl', '.test')
    format_results(results_file, formatted_path, query_mapping, doc_mapping)


def evaluate(results_file, folder_path):
    # run TREC metrics
    print('Evaluating...')
    eval = {}
    for metric, k in [('recall', 5), ('recall', 20), ('recall', 100), ('ndcg', 5), ('ndcg', 20), ('ndcg', 100)]:
        # e.g. trec_eval/trec_eval QUEST/qrels.test QUEST/predictions_Quest_BM25.test -m recall -M 5
        cmd = ['./code/evaluation/trec_eval/trec_eval',
               folder_path + 'qrels.test',
               results_file.replace('.jsonl', '.test'),
               '-m', metric,
               '-M', str(k)]
        result = subprocess.run(cmd, text=True, capture_output=True)

        if result.returncode != 0:
            raise RuntimeError(f'TREC evaluation failed: {result.stderr}')

        # only get the value we need
        target = f'{metric}'
        if metric == 'recall':
            target = target + f'_{k}'
        for line in result.stdout.splitlines():
            cols = line.split()
            if cols[0] == target:
                eval[f'{metric}_{k}'] = float(cols[-1])

    return eval


def postprocess(eval, dataset):
    print(eval)

    # \n{.000} & \n{.000} & \n{.000} & \n{.000} & \n{.000} & \n{.000}
    if dataset == 'QUEST':
        for value in eval.values():
            print(f'\\n{{{value}}} & ', end='')
    elif dataset == 'QUEST_VAR':
        for value in list(eval.values())[:-1]:
            print(f'\\n{{{value}}} & ', end='')
        print(f'\\n{{{list(eval.values())[-1]}}} \\\\', end='')
    print()


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--input', required=True, help='Results file path')
    args.add_argument('--dataset', required=False, default='QUEST', help='Dataset name (QUEST/QUEST_VAR)')
    args = args.parse_args()

    results_file = args.input
    dataset = args.dataset

    if dataset == 'QUEST':
        folder_path = './data/QUEST/'
    elif dataset == 'QUEST_VAR':
        folder_path = './data/QUEST_VAR/'
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    preprocess(results_file, folder_path)
    eval = evaluate(results_file, folder_path)
    postprocess(eval, dataset)


if __name__ == '__main__':
    main()
