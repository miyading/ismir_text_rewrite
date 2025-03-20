import os
import json
import sys


def create_file_path_jsonl(directory_path='audio', jsonl_path='output.jsonl'):
    jsonl = []
    for file in os.listdir(directory_path):
        jsonl.append({'path': directory_path+'/'+file})
    with open(jsonl_path, 'w') as f:
        for item in jsonl:
            f.write(json.dumps(item) + '\n')
    return jsonl

def combine_results(directory_path):
    with open('output.jsonl', 'r', encoding='utf-16') as f:
        data = [json.loads(line) for line in f] 

    audios = []
    for file in os.listdir(directory_path):
        audios.append({'path': directory_path+'/'+file})

    combined = []
    for i in range(len(data)):
        combined.append({**audios[i], **data[i]})
    with open('combined.jsonl', 'w') as f:
        for item in combined:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    # create_file_path_jsonl('../audios', 'audios.jsonl')
    combine_results('../audios')