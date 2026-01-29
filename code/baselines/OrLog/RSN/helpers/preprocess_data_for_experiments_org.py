# %%
import json
import requests
import json
import time
from tqdm import tqdm
import os


# Paths to the files

path_to_gold = "/home/mhoveyda/SIGIR_RSN/datasets/QUEST/test_id_added.jsonl"
path_to_documents = "/home/mhoveyda/RSN/quest_datasets/documents.jsonl"

path_to_predictions = "/home/mhoveyda/SIGIR_RSN/predictions/BM25/test_top50_sample0_2025-01-03_17-44.jsonl"
output_filterd_path = f"{path_to_predictions.split('.jsonl')[0]}_filtered_more_than_one_gold.jsonl"

path_to_augmented_filterd_predictions = f"{output_filterd_path.split('.jsonl')[0]}_with_wikidata_and_wikipedia_metadata.jsonl"


class PredictionFilter:
    def __init__(self, gold_path, pred_path, output_path):
        self.gold_path = gold_path
        self.pred_path = pred_path
        self.output_path = output_path
        self.gold = self.read_jsonl(self.gold_path)
        self.pred = self.read_jsonl(self.pred_path)
        print(f"Loaded {len(self.gold)} gold queries and {len(self.pred)} predictions")

    def read_jsonl(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f]
    
    def compare_and_filter(self):
        filtered_preds = []
        gold_dict = {g['id']: g for g in self.gold}
        pred_dict = {p['id']: p for p in self.pred}

        print(f"Comparing {len(pred_dict)} predictions to {len(gold_dict)} gold queries")
        print(f"Predictions: {list(pred_dict.keys())}")

        for k in gold_dict:
            if k in pred_dict:
                gold_docs_for_this_query = set(gold_dict[k]['docs'])  # Convert to set for fast lookup
                pred_docs_for_this_query = set(pred_dict[k]['docs'])  # Convert to set for fast lookup
                pred_dict[k]['gold_docs'] = gold_dict[k]['docs']
                
                # Create a map showing whether each predicted doc is in the gold set or not
                doc_map = {doc: (doc in gold_docs_for_this_query) for doc in pred_docs_for_this_query}
                pred_dict[k]['pred_maps'] = doc_map

                # Count how many predictions match the gold documents
                matching_preds = sum(doc_map.values())

                # If at least one prediction matches the gold docs, keep it
                if matching_preds >= 1:
                    filtered_preds.append(pred_dict[k])
                

        return filtered_preds

    def write_jsonl(self, data):
        with open(self.output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    def process(self):
        # Compare and filter predictions
        filtered_preds = self.compare_and_filter()
        # Write the filtered predictions to a file
        self.write_jsonl(filtered_preds)
        print(f"Filtered predictions written to {self.output_path}")
        return filtered_preds



class WikiData:
    def __init__(self, search_term=None):
        self.pid_to_property_name = self.get_all_properties() 
    
    def set_search_term(self, search_term):
        """Set the search term for the WikiData instance."""
        self.search_term = search_term
        self.entity_id = self.get_entity_id()


    def get_entity_id(self):
        """Get the Wikidata entity ID from Wikipedia using the search term."""
        url = f"https://en.wikipedia.org/w/api.php"
        
        # First, attempt to get the Wikidata entity ID using the API
        params = {
            "action": "query",
            "format": "json",
            "titles": self.search_term,
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "redirects": 1  # This enables automatic redirect handling in the API
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for redirected titles (if the original title was a redirect)
        if 'redirects' in data['query']:
            redirected_title = data['query']['redirects'][0]['to']
            print(f"Redirected from {self.search_term} to {redirected_title}")
            self.search_term = redirected_title  # Update the search term to the redirected title

            # Retry the request with the new redirected title
            params["titles"] = self.search_term
            response = requests.get(url, params=params)
            data = response.json()
        
        # Process the page data to extract the Wikibase entity ID
        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            if 'pageprops' in page_data and 'wikibase_item' in page_data['pageprops']:
                return page_data['pageprops']['wikibase_item']  # Return the Wikidata entity ID
        
        print(f"No entity found for {self.search_term}")
        return None
    def get_description(self):
        """Get the description of the entity."""
        if not self.entity_id:
            return None
        
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{self.entity_id}.json"
        response = requests.get(url)
        data = response.json()
        entity_data = data.get("entities", {}).get(self.entity_id, {})
        description = entity_data.get("descriptions", {}).get("en", {}).get("value", "No description available")
        return description

    # def get_entity_statements(self):
    #     """Get the property names and values of the entity."""
    #     if not self.entity_id:
    #         return None
        
    #     # print(f"Entity ID: {self.entity_id}")
        
    #     sparql_url = "https://query.wikidata.org/sparql"
    #     query = f"""
    #         SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
    #         wd:{self.entity_id} ?property ?value.
    #         SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    #         }}
    #         """
    #     headers = {"Accept": "application/sparql-results+json"}
    #     response = requests.get(sparql_url, params={'query': query}, headers=headers)

    #     try: 
    #         data = response.json()
    #     except json.decoder.JSONDecodeError as e:
    #         print(f"Error: {response.text}")
    #         print(f"response: {response}")
    #         raise e
    #     results = data['results']['bindings']
        
    #     # Extract properties and values into a dictionary using the PID-to-property name mapping
    #     entity_properties = {}
    #     for result in results:
    #         pid = result['property']['value'].split('/')[-1]  # Extract the property ID
    #         value_label = result.get('valueLabel', {}).get('value', result['value']['value'])
            
    #         # Lookup the property name using the PID mapping
    #         property_name = self.pid_to_property_name.get(pid, f"Unknown Property ({pid})")

    #         if value_label.startswith("statement/") or property_name.startswith("Unknown") or property_name.startswith("image") or property_name.startswith("Commons category") or property_name.startswith("Freebase") or property_name.startswith("ISFDB"):
    #             continue  # Skip if the value is a statement (not a direct value)

    #         # print(f"{property_name=}, {value_label=}")
    #         if not entity_properties.get(property_name):
    #             entity_properties[property_name] = []
    #         entity_properties[property_name].append(value_label)
        
    #     return entity_properties
    def get_entity_statements(self, max_retries=5):
        """Get the property names and values of the entity with retry logic."""
        if not self.entity_id:
            return None

        sparql_url = "https://query.wikidata.org/sparql"
        query = f"""
            SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
            wd:{self.entity_id} ?property ?value.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}
            """
        headers = {"Accept": "application/sparql-results+json"}

        for attempt in range(max_retries):
            response = requests.get(sparql_url, params={'query': query}, headers=headers)
            
            if response.status_code == 429:  # Handle rate limiting
                retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                print(f"Rate limited. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
                continue

            try:
                data = response.json()
                results = data['results']['bindings']
                break  # Exit the retry loop if successful
            except json.decoder.JSONDecodeError:
                print(f"Error decoding JSON: {response.text}")
                if response.status_code >= 400:
                    print(f"HTTP error: {response.status_code}")
                time.sleep(2 ** attempt)  # Exponential backoff on error
                continue
        else:
            raise Exception("Max retries exceeded for SPARQL query.")

        # Extract properties and values into a dictionary using PID-to-property name mapping
        entity_properties = {}
        for result in results:
            pid = result['property']['value'].split('/')[-1]  # Extract the property ID
            value_label = result.get('valueLabel', {}).get('value', result['value']['value'])

            property_name = self.pid_to_property_name.get(pid, f"Unknown Property ({pid})")

            if any(property_name.startswith(prefix) for prefix in [
                "Unknown", "image", "Commons category", "Freebase", "ISFDB"
            ]) or value_label.startswith("statement/"):
                continue  # Skip certain properties

            if not entity_properties.get(property_name):
                entity_properties[property_name] = []
            entity_properties[property_name].append(value_label)

        return entity_properties
    
    def get_categories(self):   
        """Get categories for the Wikipedia page, handling redirects."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": self.search_term,
            "prop": "categories",
            "cllimit": "max",
            "redirects": 1
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        # Handle redirection
        if 'redirects' in data.get('query', {}):
            redirected_title = data['query']['redirects'][0]['to']
            print(f"Redirected from {self.search_term} to {redirected_title}")
            self.search_term = redirected_title
            params["titles"] = self.search_term
            response = requests.get(url, params=params)
            data = response.json()
        
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
            print(f"No categories found for {self.search_term}")
            return []      

    def get_all_properties(self):
        """Get a mapping from PIDs to property names from Wikidata."""
        sparql_url = "https://query.wikidata.org/sparql"
        query = """
        SELECT ?property ?propertyLabel WHERE {
          ?property a wikibase:Property .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """
        headers = {"Accept": "application/sparql-results+json"}
        response = requests.get(sparql_url, params={'query': query}, headers=headers)
        data = response.json()
        
        properties_mapping = {}
        for result in data['results']['bindings']:
            pid = result['property']['value'].split('/')[-1]  # Extract the property ID from URL
            property_label = result['propertyLabel']['value']  # Get the label in English (or auto language)
            properties_mapping[pid] = property_label
        
        return properties_mapping

    def write_properties_to_file(self, filename='wikidata_properties.json'):
        """Write the PID to property name mapping to a JSON file."""
        with open(filename, 'w') as file:
            json.dump(self.pid_to_property_name, file, indent=4)
        print(f"Property mapping written to {filename}")

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]



filter_instance = PredictionFilter(path_to_gold, path_to_predictions, output_filterd_path)

filtered_preds = filter_instance.process()


wiki = WikiData()

# # Optionally write the property mapping to a file
# wiki.write_properties_to_file()


documents = read_jsonl(path_to_documents)

doc_texts = {d['title']: d['text'] for d in documents}


def collect_doc_metadata(doc, max_retries=5):
    """Collect metadata for a document."""
    wiki.set_search_term(doc)
    
    try:
        description = wiki.get_description()
        properties = wiki.get_entity_statements(max_retries=max_retries)
        categories = wiki.get_categories()
        text = doc_texts.get(doc, 'No Wikipedia text available.')
    except Exception as e:
        print(f"Error processing document {doc}: {e}")
        return None

    return {
        'doc': doc,
        'wikidata_description': description,
        'wikidata_properties': properties,
        'wikidata_categories': categories,
        'wikipedia_text': text
    }

for pred in tqdm(filtered_preds):

    pred_docs_metadata = []
    for doc in pred['docs']:
        doc_info = collect_doc_metadata(doc)
        if doc_info:
            pred_docs_metadata.append(doc_info)
        time.sleep(1)
    
    gold_docs_metadata = []
    for doc in pred['gold_docs']:
        doc_info = collect_doc_metadata(doc)
        if doc_info:
            gold_docs_metadata.append(doc_info)
        time.sleep(1) 
        
    pred['pred_docs_metadata'] = pred_docs_metadata
    pred['gold_doc_metadata'] = gold_docs_metadata



with open(path_to_augmented_filterd_predictions, 'w') as f:
    for pred in filtered_preds:
        f.write(json.dumps(pred) + '\n')
print(f"\nUpdated predictions written to {path_to_augmented_filterd_predictions}")