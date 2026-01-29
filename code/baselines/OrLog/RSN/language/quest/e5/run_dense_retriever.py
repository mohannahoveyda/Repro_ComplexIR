# language/quest/e5/run_dense_retriever.py
import json
import faiss
from absl import app, flags
from sentence_transformers import SentenceTransformer
from language.quest.common import example_utils
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("examples",  None, "Path to examples jsonl")
flags.DEFINE_string("index",     None, "Path to FAISS index file")
flags.DEFINE_string("docs",      None, "Path to document corpus jsonl")
flags.DEFINE_string("output",    None, "Where to write predictions jsonl")
flags.DEFINE_integer("sample",   0,    "If >0, sample this many examples")
flags.DEFINE_integer("topk",     20,  "How many docs to retrieve per query")

def main(_):
    examples = example_utils.read_examples(FLAGS.examples)
    if FLAGS.sample > 0:
        examples = examples[:FLAGS.sample]

    documents = []
    with open(FLAGS.docs, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            documents.append({
                "title": obj.get("title", ""), 
                "text": obj.get("text", "")
            })

    index = faiss.read_index(FLAGS.index)

    model = SentenceTransformer("intfloat/e5-base-v2")

    predictions = []
    # for ex in examples:
    for ex in tqdm(examples, desc="Retrieving", unit="query"):
        q_emb = model.encode([ex.query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, ids = index.search(q_emb, FLAGS.topk)

        docs = [documents[i]["title"] for i in ids[0]]
        scs    = [float(s)        for s in scores[0]]
        predictions.append(
            example_utils.Example(
                original_query=ex.original_query,
                query=ex.query,
                docs=docs,
                scores=scs,
                metadata=ex.metadata,
                id=ex.id
            )
        )

    example_utils.write_examples(FLAGS.output, predictions)

if __name__ == "__main__":
    app.run(main)
