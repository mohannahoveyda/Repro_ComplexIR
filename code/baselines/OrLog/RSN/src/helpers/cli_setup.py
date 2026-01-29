import argparse

from pathlib import Path

DEFAULT_DATA_PATHS = {
    "BM25": Path("/home/mhoveyda1/REASON/predictions/BM25") 
                / "test_top20_sample0_2025-07-07_14-43_Filtered_Augmented_with_wikidata_and_wikipedia_metadata.jsonl",
    "E5":  Path("/home/mhoveyda1/REASON/predictions/E5")  
                / "test_top20_sample0_2025-06-11_15-22_Filtered_Augmented_with_wikidata_and_wikipedia_metadata.jsonl",
}
def parse_cli_arguments():
    cli = argparse.ArgumentParser(
            description="Run ProbLog ranking sweep for multiple LLMs & contexts"
        )
    cli.add_argument("-r", "--retriever", choices=["BM25","E5"], required=True, help="Name of the base retriever (e.g. BM25 or E5). If omitted, will be inferred from the data path.")
    cli.add_argument("-q", "--quantize", choices=["8bit","4bit","none"], default="8bit", help="Bits‑and‑bytes quantization")
    cli.add_argument("-M", "--method", choices=["tf","likelihood"], default="tf", help="Probability calculation method")

    cli.add_argument("-n", "--n-instances", type=int, default=None, help="Total number of examples to sample") # should be at least 7
    cli.add_argument("-p", "--prompts-path", default="configs/Problog/prompts.json", help="Path to prompts JSON")
    # cli.add_argument("-d", "--data-path",
    #                     # default="predictions/BM25/test_top20_sample0_2025-05-22_12-23_filtered_with_wikidata_and_wikipedia_metadata_filtered_based_on_pred_maps_sampled_equi_110.jsonl",
    #                     # default="/home/mhoveyda1/REASON/predictions/E5/test_top20_sample0_2025-06-11_15-22_filtered_more_than_one_gold_with_wikidata_and_wikipedia_metadata_filtered_based_on_pred_maps.jsonl",
    #                     path = "/home/mhoveyda1/REASON/predictions/BM25/test_top20_sample0_2025-07-07_14-43_Filtered_Augmented_with_wikidata_and_wikipedia_metadata.jsonl",
    #                     help="Path to full JSONL input data")
    cli.add_argument(
        "-d", "--data-path",
        type=Path,
        default=None,
        help="(Optional) override the JSONL path if you really want to."
    )
    cli.add_argument("-m", "--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model path")
    cli.add_argument("--start", type=int, default=0, help="Index of first sample to process (inclusive)")
    cli.add_argument("--end",   type=int, default=None, help="Index of last sample to process (exclusive)")
    cli.add_argument("--outdir", default="results",            # overridden by the job script
        help="Directory where this chunk stores its outputs"
    )
    cli.add_argument("--log_dir", default="LOGS_NEW", help="The directory to save logs for each specific run"
    )
    cli.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse all queries (populate cache), then exit"
        )
    cli.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip all reasoning; only compute metrics on the input retriever ranking."
        )
    args = cli.parse_args()
    if args.data_path is None:
        args.data_path = DEFAULT_DATA_PATHS[args.retriever]
    return args