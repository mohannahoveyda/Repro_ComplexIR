import json
import argparse

# function to add unique reproducible IDs to each instance

def add_ids(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile):
            instance = json.loads(line)
            instance['id'] = idx + 1
            outfile.write(json.dumps(instance, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add unique IDs to JSONL file instances.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("output_file", type=str, help="Path to the output JSONL file")
    args = parser.parse_args()

    add_ids(args.input_file, args.output_file)

