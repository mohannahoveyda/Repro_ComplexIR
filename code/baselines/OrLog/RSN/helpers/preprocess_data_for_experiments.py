import asyncio
import aiohttp
import aiofiles
import json
import os
import logging
from aiohttp import ClientSession, ClientConnectorError, ClientResponseError, ServerTimeoutError
from aiohttp.client_exceptions import ClientError
from tqdm.asyncio import tqdm
from asyncio import as_completed  
import argparse


logging.basicConfig(
    filename='wikidata_processing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# path_to_gold = "/home/mhoveyda/SIGIR_RSN/datasets/QUEST/test_id_added.jsonl"
# path_to_documents = "/home/mhoveyda/RSN/quest_datasets/documents.jsonl"
# path_to_predictions = "/home/mhoveyda/SIGIR_RSN/predictions/BM25/test_top50_sample0_2025-01-03_17-44.jsonl"
# output_filtered_path = f"{path_to_predictions.split('.jsonl')[0]}_filtered_more_than_one_gold.jsonl"
# path_to_augmented_filtered_predictions = f"{output_filtered_path.split('.jsonl')[0]}_with_wikidata_and_wikipedia_metadata.jsonl"


class PredictionFilter:
    def __init__(self, gold_path, pred_path, output_path):
        self.gold_path = gold_path
        self.pred_path = pred_path
        self.output_path = output_path
        self.gold = []
        self.pred = []

    async def read_jsonl_async(self, path):
        """Asynchronously read a JSONL file and return a list of JSON objects."""
        data = []
        try:
            async with aiofiles.open(path, mode='r') as f:
                async for line in f:
                    data.append(json.loads(line))
            logging.info(f"Successfully read {len(data)} records from {path}.")
        except Exception as e:
            logging.error(f"Error reading {path}: {e}")
        return data

    async def write_jsonl_async(self, path, data):
        """Asynchronously write a list of JSON objects to a JSONL file."""
        try:
            async with aiofiles.open(path, mode='w') as f:
                for item in data:
                    await f.write(json.dumps(item) + '\n')
            logging.info(f"Successfully wrote {len(data)} records to {path}.")
        except Exception as e:
            logging.error(f"Error writing to {path}: {e}")

    async def load_data(self):
        """Load gold and prediction data asynchronously."""
        self.gold = await self.read_jsonl_async(self.gold_path)
        self.pred = await self.read_jsonl_async(self.pred_path)
        logging.info(f"Loaded {len(self.gold)} gold queries and {len(self.pred)} predictions.")

    # async def compare_and_filter(self):
    #     """Compare predictions against gold data and filter accordingly."""
    #     filtered_preds = []
    #     gold_dict = {g['id']: g for g in self.gold}
    #     pred_dict = {p['id']: p for p in self.pred}

    #     logging.info(f"Comparing {len(pred_dict)} predictions to {len(gold_dict)} gold queries.")

    #     for k in gold_dict:
    #         if k in pred_dict:
    #             gold_docs = set(gold_dict[k].get('docs', []))
    #             pred_docs = set(pred_dict[k].get('docs', []))
    #             pred_dict[k]['gold_docs'] = list(gold_docs)

    #             # Map indicating if each predicted doc is in gold docs
    #             doc_map = {doc: (doc in gold_docs) for doc in pred_docs}
    #             pred_dict[k]['pred_maps'] = doc_map

    #             # Count matching predictions
    #             matching_preds = sum(doc_map.values())

    #             # Keep if at least one prediction matches
    #             if matching_preds >= 1:
    #                 filtered_preds.append(pred_dict[k])

    #     logging.info(f"Filtered down to {len(filtered_preds)} predictions with at least one matching gold doc.")
    #     return filtered_preds

    async def compare_and_filter(self):
        filtered_preds = []
        # Build a lookup of gold docs by query ID
        gold_dict = {g['id']: g['docs'] for g in self.gold}

        for p in self.pred:
            pid = p['id']
            gold_docs = gold_dict.get(pid, [])
            p['gold_docs'] = gold_docs

            gold_set = set(gold_docs)
            p['pred_maps'] = {doc: (doc in gold_set) for doc in p['docs']}

            filtered_preds.append(p)

        return filtered_preds


    async def process(self):
        """Process the filtering and write the filtered predictions."""
        await self.load_data()
        filtered_preds = await self.compare_and_filter()
        await self.write_jsonl_async(self.output_path, filtered_preds)
        return filtered_preds


class WikiData:
    def __init__(self):
        self.pid_to_property_name = {}

    async def get_all_properties(self, session):
        """Asynchronously fetch all Wikidata properties."""
        sparql_url = "https://query.wikidata.org/sparql"
        query = """
        SELECT ?property ?propertyLabel WHERE {
          ?property a wikibase:Property .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """
        headers = {"Accept": "application/sparql-results+json"}

        try:
            async with session.get(sparql_url, params={'query': query}, headers=headers, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()

            properties_mapping = {}
            for result in data['results']['bindings']:
                pid = result['property']['value'].split('/')[-1]
                property_label = result['propertyLabel']['value']
                properties_mapping[pid] = property_label

            self.pid_to_property_name = properties_mapping
            logging.info("Fetched all Wikidata properties successfully.")
        except Exception as e:
            logging.error(f"Error fetching Wikidata properties: {e}")

    async def get_entity_id(self, session, search_term):
        """Asynchronously get the Wikidata entity ID from Wikipedia using the search term."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": search_term,
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "redirects": 1  # Enables automatic redirect handling
        }

        try:
            async with session.get(url, params=params, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
        except Exception as e:
            logging.error(f"Error fetching entity ID for '{search_term}': {e}")
            return None

        # Handle redirection
        if 'redirects' in data.get('query', {}):
            redirected_title = data['query']['redirects'][0]['to']
            logging.info(f"Redirected from '{search_term}' to '{redirected_title}'.")
            search_term = redirected_title
            params["titles"] = search_term
            try:
                async with session.get(url, params=params, timeout=30) as response:
                    response.raise_for_status()
                    data = await response.json()
            except Exception as e:
                logging.error(f"Error fetching entity ID after redirect for '{search_term}': {e}")
                return None

        # Extract Wikidata entity ID
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            if 'pageprops' in page_data and 'wikibase_item' in page_data['pageprops']:
                return page_data['pageprops']['wikibase_item']

        logging.warning(f"No entity found for '{search_term}'.")
        return None

    async def get_description(self, session, entity_id):
        """Asynchronously get the description of the entity."""
        if not entity_id:
            return None

        url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        try:
            async with session.get(url, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
            entity_data = data.get("entities", {}).get(entity_id, {})
            description = entity_data.get("descriptions", {}).get("en", {}).get("value", "No description available")
            return description
        except Exception as e:
            logging.error(f"Error fetching description for entity ID '{entity_id}': {e}")
            return None

    async def get_entity_statements(self, session, entity_id, max_retries=5):
        """Asynchronously get the property names and values of the entity with retry logic."""
        if not entity_id:
            return None

        sparql_url = "https://query.wikidata.org/sparql"
        query = f"""
            SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
            wd:{entity_id} ?property ?value.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}
            """
        headers = {"Accept": "application/sparql-results+json"}

        for attempt in range(1, max_retries + 1):
            try:
                async with session.get(sparql_url, params={'query': query}, headers=headers, timeout=30) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                        logging.warning(f"Rate limited. Retrying in {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        continue
                    response.raise_for_status()
                    data = await response.json()
                break  # Successful request
            except (ClientError, ServerTimeoutError, asyncio.TimeoutError) as e:
                wait_time = 2 ** attempt
                logging.warning(f"SPARQL query attempt {attempt} failed: {e}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            except json.decoder.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}. Retrying in {2 ** attempt} seconds...")
                await asyncio.sleep(2 ** attempt)
        else:
            logging.error(f"Max retries exceeded for SPARQL query of entity ID '{entity_id}'.")
            return None

        results = data.get('results', {}).get('bindings', [])
        entity_properties = {}
        for result in results:
            pid = result['property']['value'].split('/')[-1]
            value_label = result.get('valueLabel', {}).get('value', result['value']['value'])
            property_name = self.pid_to_property_name.get(pid, f"Unknown Property ({pid})")

            if any(property_name.startswith(prefix) for prefix in [
                "Unknown", "image", "Commons category", "Freebase", "ISFDB"
            ]) or value_label.startswith("statement/"):
                continue  

            if property_name not in entity_properties:
                entity_properties[property_name] = []
            entity_properties[property_name].append(value_label)

        return entity_properties

    async def get_categories(self, session, search_term):
        """Asynchronously get categories for the Wikipedia page, handling redirects."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": search_term,
            "prop": "categories",
            "cllimit": "max",
            "redirects": 1
        }

        try:
            async with session.get(url, params=params, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
        except Exception as e:
            logging.error(f"Error fetching categories for '{search_term}': {e}")
            return []

        if 'redirects' in data.get('query', {}):
            redirected_title = data['query']['redirects'][0]['to']
            logging.info(f"Redirected from '{search_term}' to '{redirected_title}'.")
            search_term = redirected_title
            params["titles"] = search_term
            try:
                async with session.get(url, params=params, timeout=30) as response:
                    response.raise_for_status()
                    data = await response.json()
            except Exception as e:
                logging.error(f"Error fetching categories after redirect for '{search_term}': {e}")
                return []

        # Extract categories
        pages = data.get('query', {}).get('pages', {})
        categories = []
        for page_data in pages.values():
            if 'categories' in page_data:
                categories = [category["title"] for category in page_data["categories"]]
                break

        if categories:
            return categories
        else:
            logging.warning(f"No categories found for '{search_term}'.")
            return []      


class WikiDataProcessor:
    def __init__(self, doc_texts):
        self.doc_texts = doc_texts
        self.wiki = WikiData()

    async def collect_doc_metadata(self, session, doc, max_retries=5):
        """Collect metadata for a single document."""
        entity_id = await self.wiki.get_entity_id(session, doc)
        if not entity_id:
            return {
                'doc': doc,
                'wikidata_description': None,
                'wikidata_properties': None,
                'wikidata_categories': None,
                'wikipedia_text': self.doc_texts.get(doc, 'No Wikipedia text available.')
            }

        description = await self.wiki.get_description(session, entity_id)
        properties = await self.wiki.get_entity_statements(session, entity_id, max_retries=max_retries)
        categories = await self.wiki.get_categories(session, doc)
        text = self.doc_texts.get(doc, 'No Wikipedia text available.')

        return {
            'doc': doc,
            'wikidata_description': description,
            'wikidata_properties': properties,
            'wikidata_categories': categories,
            'wikipedia_text': text
        }


async def read_existing_output(path):
    processed_ids = set()
    if not os.path.exists(path):
        logging.info(f"Output file '{path}' does not exist. Starting fresh.")
        return processed_ids

    try:
        async with aiofiles.open(path, mode='r') as f:
            async for line in f:
                pred = json.loads(line)
                processed_ids.add(pred['id'])
        logging.info(f"Found {len(processed_ids)} already processed predictions in '{path}'.")
    except Exception as e:
        logging.error(f"Error reading existing output file '{path}': {e}")

    return processed_ids


async def write_augmented_prediction(path, prediction):
    try:
        async with aiofiles.open(path, mode='a') as f:
            await f.write(json.dumps(prediction) + '\n')
        logging.info(f"Appended prediction ID {prediction['id']} to '{path}'.")
    except Exception as e:
        logging.error(f"Error writing prediction ID {prediction['id']} to '{path}': {e}")


async def process_single_prediction(pred, wiki_processor, session, output_path, semaphore):
    async with semaphore:
        try:
            pred_docs_metadata = []
            for doc in pred.get('docs', []):
                doc_info = await wiki_processor.collect_doc_metadata(session, doc)
                if doc_info:
                    pred_docs_metadata.append(doc_info)
                await asyncio.sleep(0.1)  

            gold_docs_metadata = []
            for doc in pred.get('gold_docs', []):
                doc_info = await wiki_processor.collect_doc_metadata(session, doc)
                if doc_info:
                    gold_docs_metadata.append(doc_info)
                await asyncio.sleep(0.1)  

            pred['pred_docs_metadata'] = pred_docs_metadata
            pred['gold_doc_metadata'] = gold_docs_metadata

            await write_augmented_prediction(output_path, pred)
        except Exception as e:
            logging.error(f"Error processing prediction ID {pred.get('id')}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter predictions against gold and augment with Wikidata metadata."
    )
    parser.add_argument(
        "--gold", "-g",
        default="./datasets/QUEST/test_id_added.jsonl",
        help="Path to the gold JSONL file (default: %(default)s)."
    )
    parser.add_argument(
        "--documents", "-d",
        default="./datasets/QUEST/tmp/documents.jsonl",
        help="Path to the documents JSONL file (default: %(default)s)."
    )
    parser.add_argument(
        "--predictions", "-p",
        required=True,
        help="Path to the input predictions JSONL file."
    )
    parser.add_argument(
        "--out-filtered", "-f",
        help="Path to write filtered predictions. Defaults to '<predictions>_filtered_more_than_one_gold.jsonl'."
    )
    parser.add_argument(
        "--out-augmented", "-a",
        help="Path to write augmented predictions. Defaults to '<out-filtered>_with_wikidata_and_wikipedia_metadata.jsonl'."
    )
    return parser.parse_args()


async def main(
    path_to_gold: str,
    path_to_documents: str,
    path_to_predictions: str,
    output_filtered_path: str,
    path_to_augmented_filtered_predictions: str
):

    filter_instance = PredictionFilter(path_to_gold, path_to_predictions, output_filtered_path)
    filtered_preds = await filter_instance.process()

    documents = await filter_instance.read_jsonl_async(path_to_documents)
    doc_texts = {d['title']: d['text'] for d in documents}
    wiki_processor = WikiDataProcessor(doc_texts)

    processed_ids = await read_existing_output(path_to_augmented_filtered_predictions)

    preds_to_process = [pred for pred in filtered_preds if pred['id'] not in processed_ids]
    total_to_process = len(preds_to_process)
    logging.info(f"Total predictions to process: {total_to_process}")

    if total_to_process == 0:
        logging.info("All predictions have already been processed.")
        print("All predictions have already been processed.")
        return

    timeout = aiohttp.ClientTimeout(total=60)  # Adjust as needed
    connector = aiohttp.TCPConnector(limit=20)  # Limit concurrent connections
    async with ClientSession(timeout=timeout, connector=connector) as session:
        await wiki_processor.wiki.get_all_properties(session)

        semaphore = asyncio.Semaphore(10)  

        tasks = [
            process_single_prediction(pred, wiki_processor, session, path_to_augmented_filtered_predictions, semaphore)
            for pred in preds_to_process
        ]

        for task in tqdm(as_completed(tasks), total=total_to_process, desc="Processing Predictions"):
            await task

    logging.info(f"All predictions have been processed and written to '{path_to_augmented_filtered_predictions}'.")
    print(f"\nUpdated predictions written to {path_to_augmented_filtered_predictions}")


if __name__ == "__main__":
    args = parse_args()

    out_filtered = args.out_filtered or args.predictions.replace(
        ".jsonl", "_Filtered.jsonl"
    )
    out_augmented = args.out_augmented or out_filtered.replace(
        ".jsonl", "_Augmented_with_wikidata_and_wikipedia_metadata.jsonl"
    )
    try:
        # asyncio.run(main())
        asyncio.run(main(
            path_to_gold=args.gold,
            path_to_documents=args.documents,
            path_to_predictions=args.predictions,
            output_filtered_path=out_filtered,
            path_to_augmented_filtered_predictions=out_augmented
        ))

    except KeyboardInterrupt:
        logging.warning("Process interrupted by user.")
        print("\nProcess interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")