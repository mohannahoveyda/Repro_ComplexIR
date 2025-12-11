import json

# make documents.jsonl
# wanted: dict_keys(['title', 'text'])
with open('./data/QUEST_VAR/quest_text_w_id_withVarients.jsonl', 'r') as read_file, open('./data/QUEST_VAR/documents.jsonl', 'w') as write_file:
    print('making documents.jsonl...')
    for line in read_file:
        data = json.loads(line)
        doc_entry = {
            'title': data['title'],
            'text': data['text']
        }
        write_file.write(json.dumps(doc_entry) + '\n')

# make title id mapping
id_mapping = {}
with open('./data/QUEST_VAR/quest_text_w_id_withVarients.jsonl', 'r') as read_file:
    print('making title id mapping...')
    for line in read_file:
        doc = json.loads(line)
        if doc['idx'] not in id_mapping.keys():
            id_mapping[doc['idx']] = doc['title']

# make test.jsonl
# wanted: dict_keys(['query', 'docs', 'original_query', 'scores', 'metadata'])
#   metadata: dict_keys(['template', 'relevance_ratings', 'evidence_ratings', 'attributions', 'domain'])
with open('./data/QUEST_VAR/quest_test.jsonl', 'r') as read_file, open('./data/QUEST_VAR/test.jsonl', 'w') as write_file:
    print('making test.jsonl...')
    i = 0
    for line in read_file:
        i += 1
        data = json.loads(line)
        doc_entry = {
            'query': data['nl_query'],
            'docs': [id_mapping[doc] for doc in data['documents']],
            'original_query': data['queries'],
            'scores': None,
            'metadata': {
                'template': data['operators'],
                'relevance_ratings': None,
                'evidence_ratings': None,
                'attributions': None,
                'domain': data['domain']
            }
        }
        write_file.write(json.dumps(doc_entry) + '\n')
