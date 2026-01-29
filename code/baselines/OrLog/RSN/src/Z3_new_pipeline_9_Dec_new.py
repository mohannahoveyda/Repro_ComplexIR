import json
from openai import OpenAI
import re
# from utils import *
import time
import pandas as pd
import io
from datetime import date
import contextlib
from collections import defaultdict
import random
from tqdm import tqdm
import os
random.seed(42)

DATA_SAMPLES = 1
# ENTITY_SAMPLES = 1
LINC_MAJ = 5
CONSIDER_CATEGORIES = True
MAX_ATTEMPTS = 1
SEPARATE = False

today = date.today()

# path_ = "/home/mhoveyda/SIGIR_RSN/filtered_predictions_with_wikidata_and_wikipedia_metadata_2_9_Nov.jsonl"
path_ = "/home/mhoveyda1/RSN_Z/test_top20_sample0_2025-05-22_12-23_filtered_with_wikidata_and_wikipedia_metadata_filtered_based_on_pred_maps_sampled_equi_110.jsonl"

# output_path = f"/home/mhoveyda/RSN/predictions_with_z3_results_{today}_m.jsonl"

useless_properties = [
    "IMDb ID",
    "FilmAffinity film ID",
    "Rotten Tomatoes ID",
    "MPA film rating",
    "DNF film ID",
    "Kinopoisk film ID",
    "Google Knowledge Graph ID",
    "LUMIERE film ID",
    "TMDB movie ID",
    "Letterboxd film ID",
    "iTunes movie ID",
    "Trakt.tv ID",
    "Apple TV movie ID",
    "Plex media key",
    "Schnittberichte.com title ID",
    "NientePopCorn movie ID",
    "Trakt.tv film ID",
    "FilmVandaag ID",
    "EIDR content ID",
    "Moviebuff ID",
    "AlloCiné film ID",
    "Metacritic ID",
    "Douban film ID",
    "Filmweb.pl film ID",
    "IGAC rating",
    "Kinobox film ID",
    "review score",
    "AllMovie title ID",
    "official website",
    "box office",
    "KINENOTE film ID",
    "EIRIN film rating",
    "The Numbers movie ID",
    "MYmovies movie ID",
    "Movieplayer film ID",
    "Il mondo dei doppiatori ID",
    "FilmTv.it movie ID",
    "Unconsenting Media ID",
    "Allcinema film ID",
    "Movie Walker Press film ID",
    "title",
    "cast"
    "ISAN",
    "cost",
    "duration",
]

api_key="2EbJI2KPZoAWWb3YBoxDtyGk4S33CmMo"
base_url="https://api.deepinfra.com/v1/openai"



def calculate_metrics(data, frame_work, k_values=[1, 3, 5, 10]):
    """
    Calculates MRR and Precision@K for the initial BM25 results and after verification.
    
    :param data: List of JSON objects loaded from the JSONL file.
    :param k_values: List of K values for Precision@K calculation.
    :return: A dictionary containing the metrics for BM25 and the new framework.
    """
    metrics = defaultdict(dict)

    for k in k_values:
        metrics["BM25"][f"Precision@{k}"] = 0
        metrics[frame_work][f"Precision@{k}"] = 0

    total_queries = len(data)
    bm25_rr_sum, framework_rr_sum = 0, 0

    for entry in data:
        gold_docs = set(entry["gold_docs"])
        bm25_results = entry["docs"]
        framework_results = [item["doc"] for item in sorted(entry["pred_docs_metadata"], 
                                                            key=lambda x: (x["Z3_result"] == "True", -bm25_results.index(x["doc"])),
                                                            reverse=True)]
        
        # print(f"Gold Docs: {gold_docs}")
        # print(f"BM25 Results: {bm25_results}")
        # print(f"Framework {frame_work} Results: {framework_results}")

        # Calculate MRR for BM25
        bm25_rr = 0
        for idx, doc in enumerate(bm25_results):
            if doc in gold_docs:
                bm25_rr = 1 / (idx + 1)
                break
        bm25_rr_sum += bm25_rr

        # Calculate MRR for the new framework
        framework_rr = 0
        for idx, doc in enumerate(framework_results):
            if doc in gold_docs:
                framework_rr = 1 / (idx + 1)
                break
        framework_rr_sum += framework_rr

        # Calculate Precision@K for BM25 and framework
        for k in k_values:
            bm25_precision = len([doc for doc in bm25_results[:k] if doc in gold_docs]) / k
            framework_precision = len([doc for doc in framework_results[:k] if doc in gold_docs]) / k
            metrics["BM25"][f"Precision@{k}"] += bm25_precision
            metrics[frame_work][f"Precision@{k}"] += framework_precision

    # Finalize metrics
    metrics["BM25"]["MRR"] = bm25_rr_sum / total_queries
    metrics[frame_work]["MRR"] = framework_rr_sum / total_queries
    for k in k_values:
        metrics["BM25"][f"Precision@{k}"] /= total_queries
        metrics[frame_work][f"Precision@{k}"] /= total_queries

    return metrics

class IncrementalEvaluation:
    def __init__(self):
        # Initialize confusion matrix and other metrics with zero counts
        labels_gold = ['True', 'False']
        labels_pred = ['True', 'False', 'Uncertain', 'Error']
        self.confusion_matrix = pd.DataFrame(0, index=labels_gold, columns=labels_pred)
        self.reciprocal_ranks = []
        self.precision_at_k_list = []

    def update_confusion_matrix(self, gold_label, predicted_label):
        # Ensure the predicted label is one of the expected labels
        if predicted_label not in self.confusion_matrix.columns:
            predicted_label = 'Error'
        if gold_label in self.confusion_matrix.index:
            self.confusion_matrix.loc[gold_label, predicted_label] += 1

    def update_P_at_k(self, pred_maps, docs_metadata, k=3):
        relevant_count = 0
        for i, doc_metadata in enumerate(docs_metadata[:k]):
            doc_name = doc_metadata['doc']
            if doc_name in pred_maps and pred_maps[doc_name]:  # Check if the document is relevant
                relevant_count += 1
        # Append this P@k score for the current item
        self.precision_at_k_list.append(relevant_count / k)

    def update_MRR(self, pred_maps, docs_metadata):
        for rank, doc_metadata in enumerate(docs_metadata, start=1):
            doc_name = doc_metadata['doc']
            if doc_name in pred_maps and pred_maps[doc_name]:  # Check if the document is relevant
                self.reciprocal_ranks.append(1 / rank)
                return  # Stop at the first relevant document
        # Append 0 if no relevant document was found
        self.reciprocal_ranks.append(0)

    def get_metrics(self):
        # Calculate average metrics
        avg_P_at_k = sum(self.precision_at_k_list) / len(self.precision_at_k_list) if self.precision_at_k_list else 0
        avg_MRR = sum(self.reciprocal_ranks) / len(self.reciprocal_ranks) if self.reciprocal_ranks else 0
        return {
            'Confusion Matrix': self.confusion_matrix,
            'Average P@k': avg_P_at_k,
            'Average MRR': avg_MRR
        }


class Reason:
    def __init__(self,path_to_predictions, key, url, useless_properties):
        self.path_to_predictions = path_to_predictions
        self.api_key = key
        self.base_url = url
        self.openai = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url
            )
        self.useless_properties = useless_properties
    
    def read_jsonl(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f]
    def generate(self, system_message, user_message):
        response = self.openai.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
    )

        
        print(f"{response.choices[0].message.content}")

        print(f"Prompt_Tokens_Count: {response.usage.prompt_tokens}, Result_Tokens_Count: {response.usage.completion_tokens}")
        return response.choices[0].message.content
    def collect_doc_metadata(self, doc):
        wikidata_description = doc["wikidata_description"]

        properties = doc["wikidata_properties"]
        filtered_properties = {}

        for prop, values in properties.items():
            if prop not in self.useless_properties and "ID" not in prop:
                filtered_properties[prop] = values
        text = doc["wikipedia_text"]
        
        return {
            'doc': doc,
            'wikidata_description': wikidata_description,
            'wikidata_properties': properties,
            'wikipedia_text': text
        }    
    def process_doc_name(self, doc):
        # Remove the year and any text in parentheses
        processed_name = re.sub(r'\s*\(.*?\)', '', doc)
        # Replace spaces with underscores
        processed_name = processed_name.replace(' ', '_')
        return processed_name
    
    def make_prompt(self, query, doc_metadata, doc_name, filtered_categories=None):

        conclusion = query
        premises = []
        for prop, values in doc_metadata.items():
            if prop not in self.useless_properties and "ID" not in prop:
                if prop == 'publication date':
                    values = values[0]
                    values = [values.split('-').pop(0)]
                
                
                values_list = [f"'{v}'" for v in values]
                values_str = ', '.join(values_list)
                premises.append(f"{prop}: {values_str}")
    
            
        

        if not CONSIDER_CATEGORIES:

            prompt = """
            The following is a first-order logic (FOL) problem.
            The premises are given as a set of first-order logic sentences.
            The conclusion is a single first-order logic sentence.
            The task is to translate each of the premises and the conclusion into FOL expressions suitable for evaluation by the Z3 theorem prover.
            Expressions should be in Python code using the Z3 library and in a python code block.

            The generated format should follow the example below:

            <Entity>
            name: The_Sofa_:_A_Moral_Tale
            </Entity>

            <PREMISES>
            Instance of: ['written work']
            Author: ['Claude Prosper Jolyot de Crébillon']
            Genre: ['fantasy', 'libertine novel', 'erotic novel']
            Language of work or name: ['French']
            Country of origin: ['France']
            Publication date: ['1742']
            </PREMISES>

            <CONCLUSION> 1742 or French satirical novels. </CONCLUSION>

            <EVALUATE>
            
            ```python
            # Python code using Z3

            from z3 import *
            # Declare sorts
            Work = DeclareSort('Work')
            Person = DeclareSort('Person')
            Genre = DeclareSort('Genre')
            Language = DeclareSort('Language')
            Country = DeclareSort('Country')

            # Declare constants for entities
            the_sofa = Const('The_Sofa_A_Moral_Tale', Work)
            written_work = Const('Written_Work', Work)
            crebillon = Const('Claude_Prosper_Jolyot_de_Crebillon', Person)
            fantasy = Const('Fantasy', Genre)
            libertine_novel = Const('Libertine_Novel', Genre)
            erotic_novel = Const('Erotic_Novel', Genre)
            french = Const('French', Language)
            france = Const('France', Country)
            satirical_novel = Const('Satirical_Novel', Genre)
            # Declare predicates
            InstanceOf = Function('InstanceOf', Work, Work, BoolSort())
            AuthorOf = Function('AuthorOf', Work, Person, BoolSort())
            HasGenre = Function('HasGenre', Work, Genre, BoolSort())
            LanguageOf = Function('LanguageOf', Work, Language, BoolSort())
            CountryOfOrigin = Function('CountryOfOrigin', Work, Country, BoolSort())
            PublicationDate = Function('PublicationDate', Work, IntSort(), BoolSort())

            # Initialize the solver
            s = Solver()

            # Add premises to the solver
            s.add(InstanceOf(the_sofa, written_work))
            s.add(AuthorOf(the_sofa, crebillon))
            s.add(HasGenre(the_sofa, fantasy))
            s.add(HasGenre(the_sofa, libertine_novel))
            s.add(HasGenre(the_sofa, erotic_novel))
            s.add(LanguageOf(the_sofa, french))
            s.add(CountryOfOrigin(the_sofa, france))
            s.add(PublicationDate(the_sofa, 1742))

            # Express the conclusion: "1742 or French satirical novels."
            # Define the conclusion as a Z3 expression
            conclusion = Or(
                PublicationDate(the_sofa, 1742),
                And(
                    HasGenre(the_sofa, satirical_novel),
                    CountryOfOrigin(the_sofa, france)
                )
            )
            # Add the negation of the conclusion to the solver to check for unsatisfiability
            s.add(Not(conclusion))

            # Check if the premises and the negation of the conclusion are unsatisfiable and print the result.
            if s.check() == unsat:
                print("True")
            else:
                print("False")
            ```
            </EVALUATE>
            """


            prompt += f"\n<Entity>\nname: {doc_name}\n</Entity>\n"
            prompt += f"<PREMISES>\n"
            for premise in premises:
                prompt += f"{premise}\n"

            prompt += f"\n</PREMISES>"
            prompt += f"\n<CONCLUSION> {conclusion} </CONCLUSION>"
            prompt += f"\n</EVALUATE>"
            return prompt
        elif CONSIDER_CATEGORIES:
            prompt = """
            The following is a first-order logic (FOL) problem.
            The premises are given as a set of first-order logic sentences.
            The conclusion is a single first-order logic sentence.
            The task is to translate each of the premises and the conclusion into FOL expressions suitable for evaluation by the Z3 theorem prover.
            Expressions should be in Python code using the Z3 library and in a python code block.

            The generated format should follow the example below:

            <Entity>
            name: The_Sofa_:_A_Moral_Tale
            </Entity>

            <PREMISES>
            Instance of: ['written work']
            Author: ['Claude Prosper Jolyot de Crébillon']
            Genre: ['fantasy', 'libertine novel', 'erotic novel']
            Language of work or name: ['French']
            Country of origin: ['France']
            Publication date: ['1742']
            Example of: ['1742 novels']
            </PREMISES>

            <CONCLUSION> 1742 or French satirical novels. </CONCLUSION>

            <EVALUATE>
            
            ```python
            # Python code using Z3

            from z3 import *
            # Declare sorts
            Work = DeclareSort('Work')
            Person = DeclareSort('Person')
            Genre = DeclareSort('Genre')
            Language = DeclareSort('Language')
            Country = DeclareSort('Country')

            # Declare constants for entities
            the_sofa = Const('The_Sofa_A_Moral_Tale', Work)
            written_work = Const('Written_Work', Work)
            crebillon = Const('Claude_Prosper_Jolyot_de_Crebillon', Person)
            fantasy = Const('Fantasy', Genre)
            libertine_novel = Const('Libertine_Novel', Genre)
            erotic_novel = Const('Erotic_Novel', Genre)
            french = Const('French', Language)
            france = Const('France', Country)
            satirical_novel = Const('Satirical_Novel', Genre)
            # Declare predicates
            InstanceOf = Function('InstanceOf', Work, Work, BoolSort())
            AuthorOf = Function('AuthorOf', Work, Person, BoolSort())
            HasGenre = Function('HasGenre', Work, Genre, BoolSort())
            LanguageOf = Function('LanguageOf', Work, Language, BoolSort())
            CountryOfOrigin = Function('CountryOfOrigin', Work, Country, BoolSort())
            PublicationDate = Function('PublicationDate', Work, IntSort(), BoolSort())

            # Initialize the solver
            s = Solver()

            # Add premises to the solver
            s.add(InstanceOf(the_sofa, written_work))
            s.add(AuthorOf(the_sofa, crebillon))
            s.add(HasGenre(the_sofa, fantasy))
            s.add(HasGenre(the_sofa, libertine_novel))
            s.add(HasGenre(the_sofa, erotic_novel))
            s.add(LanguageOf(the_sofa, french))
            s.add(CountryOfOrigin(the_sofa, france))
            s.add(PublicationDate(the_sofa, 1742))

            # Express the conclusion: "1742 or French satirical novels."
            # Define the conclusion as a Z3 expression
            conclusion = Or(
                PublicationDate(the_sofa, 1742),
                And(
                    HasGenre(the_sofa, satirical_novel),
                    CountryOfOrigin(the_sofa, france)
                )
            )
            # Add the negation of the conclusion to the solver to check for unsatisfiability
            s.add(Not(conclusion))

            # Check if the premises and the negation of the conclusion are unsatisfiable and print the result.
            if s.check() == unsat:
                print("True")
            else:
                print("False")
            ```
            </EVALUATE>
            """


            prompt += f"\n<Entity>\nname: {doc_name}\n</Entity>\n"
            prompt += f"<PREMISES>\n"
            for premise in premises:
                prompt += f"{premise}\n"
            for cat in filtered_categories:
                prompt += f"Example of: ['{cat}']\n"

            prompt += f"\n</PREMISES>"
            prompt += f"\n<CONCLUSION> {conclusion} </CONCLUSION>"
            prompt += f"\n</EVALUATE>"
            return prompt
 
    def split_on_commas_not_in_quotes_or_parentheses(self, s):
        result = []
        current = ''
        in_single_quote = False
        in_double_quote = False
        in_parentheses = 0
        escape = False
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\' and not escape:
                escape = True
                current += c
            elif c == "'" and not in_double_quote and not escape:
                in_single_quote = not in_single_quote
                current += c
            elif c == '"' and not in_single_quote and not escape:
                in_double_quote = not in_double_quote
                current += c
            elif c == '(' and not in_single_quote and not in_double_quote and not escape:
                in_parentheses += 1
                current += c
            elif c == ')' and not in_single_quote and not in_double_quote and not escape:
                if in_parentheses > 0:
                    in_parentheses -= 1
                current += c
            elif c == ',' and not in_single_quote and not in_double_quote and in_parentheses == 0 and not escape:
                result.append(current.strip())
                current = ''
            else:
                current += c
                escape = False
            i += 1
        if current:
            result.append(current.strip())
        return result

    def is_section_header(self, line):
        line = line.strip()
        # Check for FOL; TEXT; TEXT:
        if line.startswith('FOL;') or line.startswith('TEXT;') or line.startswith('TEXT:'):
            return True
        # Regular expression to match section headers with optional spaces
        if re.match(r'<\s*/?\s*(EVALUATE|CONCLUSION|PREMISES|Entity)\s*>', line, re.IGNORECASE):
            return True
        if line == '':
            return True
        return False

    def extract_fol_expressions(self, llm_output):
        fol_expressions = []
        # Replace curly quotes with straight quotes
        llm_output = llm_output.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        # Remove backslashes
        llm_output = llm_output.replace("\\", "")
        lines = llm_output.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('FOL;'):
                content = line[len('FOL;'):].strip()
                i += 1
                # Collect the content that may span multiple lines
                while i < len(lines):
                    next_line = lines[i].strip()
                    if self.is_section_header(next_line):
                        break
                    content += ' ' + next_line
                    i += 1
                # Now, split content into expressions
                expressions = self.split_on_commas_not_in_quotes_or_parentheses(content)
                for expr in expressions:
                    expr = expr.strip()
                    # Remove enclosing quotes if any
                    if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
                        expr = expr[1:-1]
                    if expr:
                        fol_expressions.append(expr)
                # Note: Do not increment i here because it's already pointing to the next line
            else:
                i += 1
        return fol_expressions

    # def extract_code(self,llm_output):
    #     # Use regular expression to find code blocks labeled as Python code
    #     try:
    #         code_blocks = re.findall(r'```python(.*?)```', llm_output, re.DOTALL)
    #     except:
    #         return -1
    #     # Combine all extracted code blocks
    #     code = '\n\n'.join(code_blocks).strip()
    #     return code


    def extract_code(self, llm_output):
        # Use regular expression to find code blocks labeled as Python code
        try:
            code_blocks = re.findall(r'```python(.*?)```', llm_output, re.DOTALL)
            if not code_blocks:
                return None  # Return None if no code blocks are found
            # Return the last extracted code block, stripping unnecessary whitespace
            return code_blocks[-1].strip()
        except Exception as e:
            return -1
    def extract_JSON(self, llm_output):
        json_blocks = re.findall(r'```JSON\n(.*?)\n```', llm_output, re.DOTALL)
    
        if not json_blocks:
            return None  # Return None if no JSON blocks found
        
        # Parse the last JSON block found
        try:
            last_json = json.loads(json_blocks[-1])
            return last_json
        except json.JSONDecodeError:
            return None  # Return None if the last block is not valid JSON



    def run_prover9(self, expressions):
        premises = expressions[:len(expressions) - 1]
        conclusion = expressions[-1]
        # Print the logical argument
        print("---Premises---\n")
        for premise in premises:
            print(premise)
            


        print("\n")
        # Evaluate the logical argument
        start_time = time.time()    
        result = evaluate(premises, conclusion)
        # print(f"Result: {result}")
        print("--- %s seconds ---" % (time.time() - start_time)) 
        return result
    def translate_query_to_z3(self, query):
        system_message = f"""

        The following is a first-order logic (FOL) problem. 
        The conclusion is a single first-order logic sentence. 
        The task is to translate the conclusion to an FOL expression in Z3 Python code. 
        Here is an example:

        <Query> 1742 or French satirical novels. </Query>

        Translated to Z3 Python code:

        ```python
        conclusion = Or(
            PublicationDate(entity, 1742),
            And(
                CountryOfOrigin(entity, france),
                Genre(entity, satirical),
                TypeOfWork(entity, novel)
            )
        )
        ```

        Now, translate the following query into logical code compatible with Z3:
        <Query> {query} </Query>
        """
        response = self.generate("",system_message)
        return self.extract_code(response)
    def translate_premises(self, translated_query, entity, premises):
        system_message = f"""

        The following is a first-order logic (FOL) problem. 
        The premises are given as a set of first-order logic sentences. 
        The task it to use the given translated query and the provided entity and premises, to translate each of the premises into FOL expressions in Z3 Python code.

        Here is an example:

        <Entity> The_Sofa_:_A_Moral_Tale </Entity>

        <PREMISES>
        Instance of: ['written work']
        Author: ['Claude Prosper Jolyot de Crébillon']
        Genre: ['fantasy', 'libertine novel', 'erotic novel']
        Language of work or name: ['French']
        Country of origin: ['France']
        Publication date: ['1742']
        Example of: ['1742 novels']
        </PREMISES>

        <Translated Query> 
        conclusion = Or(
            PublicationDate(entity, 1742),
            And(
                CountryOfOrigin(entity, france),
                Genre(entity, satirical),
                TypeOfWork(entity, novel)
            )
        )
        </Translated Query>

        Translated premises to Z3 Python code:

        
        ```python

        # Declare sorts
        Work = DeclareSort('Work')
        Person = DeclareSort('Person')
        Genre = DeclareSort('Genre')
        Language = DeclareSort('Language')
        Country = DeclareSort('Country')

        # Declare constants for entities
        entity = Const('The_Sofa_A_Moral_Tale', Work)
        written_work = Const('Written_Work', Work)
        crebillon = Const('Claude_Prosper_Jolyot_de_Crebillon', Person)
        fantasy = Const('Fantasy', Genre)
        libertine_novel = Const('Libertine_Novel', Genre)
        erotic_novel = Const('Erotic_Novel', Genre)
        french = Const('French', Language)
        france = Const('France', Country)
        satirical_novel = Const('Satirical_Novel', Genre)

        # Declare predicates
        InstanceOf = Function('InstanceOf', Work, Work, BoolSort())
        AuthorOf = Function('AuthorOf', Work, Person, BoolSort())
        HasGenre = Function('HasGenre', Work, Genre, BoolSort())
        LanguageOf = Function('LanguageOf', Work, Language, BoolSort())
        CountryOfOrigin = Function('CountryOfOrigin', Work, Country, BoolSort())
        PublicationDate = Function('PublicationDate', Work, IntSort(), BoolSort())

        # Initialize the solver
        s = Solver()

        # Add premises to the solver
        s.add(InstanceOf(entity, written_work))
        s.add(AuthorOf(entity, crebillon))
        s.add(HasGenre(entity, fantasy))
        s.add(HasGenre(entity, libertine_novel))
        s.add(HasGenre(entity, erotic_novel))
        s.add(LanguageOf(entity, french))
        s.add(CountryOfOrigin(entity, france))
        s.add(PublicationDate(entity, 1742))
        ```
        
        Now, given the the translated query and the premises for the entity, translate the premises into logical code compatible with Z3

        <Translated Query> {translated_query} </Translated Query>
        <Entity> {entity} </Entity>
        <PREMISES> {premises} </PREMISES>
        """

        response = self.generate("", system_message)
        return self.extract_code(response)
    def combine_code(self, query_code, premises_code):
        imports = """
from z3 import *
        """
        common_part = """
# Add negation and check satisfiability
s.add(Not(conclusion))
if s.check() == unsat:
    print("True")
else:
    print("False")
        """

        return f"{imports}\n\n{premises_code}\n\n{query_code}\n\n{common_part}"




rsn = Reason(path_, api_key, base_url, useless_properties)
data = rsn.read_jsonl(rsn.path_to_predictions)
print(f"Number of queries: {len(data)}")
# data_test = data[:DATA_SAMPLES]
# Filter out queries with missing or empty `doc_metadata` in any document
# valid_data = [
#     d for d in data 
#     if all(doc.get("wikidata_properties") for doc in d["pred_docs_metadata"])
# ]
# Check all the possible templates and their counts
template_counts = defaultdict(int)
for d in data:
    template_counts[d['metadata']['template']] += 1
print(f"Template Counts: {template_counts}")

# valid_data = [
#     d for d in data 
#     if all(doc.get("wikidata_properties") for doc in d["pred_docs_metadata"]) and d['metadata']['template'] != '_'
# ]

# valid_data = [
#     d for d in data 
#     if all(doc.get("wikidata_properties") for doc in d["pred_docs_metadata"]) # and d['metadata']['template'] != '_'
# ]

# Check all the possible templates and their counts
# template_counts = defaultdict(int)
# for d in valid_data:
#     template_counts[d['metadata']['template']] += 1
# print(f"Template Counts: {template_counts}")
valid_data = data

print(f"Number of valid queries: {len(valid_data)}")    
data_test = random.sample(valid_data, DATA_SAMPLES)
print(f"Number of queries sampled: {len(data_test)}")

# data_test = data 
# Check all the possible templates and their counts
template_counts = defaultdict(int)
for d in valid_data:
    template_counts[d['metadata']['template']] += 1
print(f"Template Counts: {template_counts}")

# data_test = data 


def run_experiment(experiment_mode, data_test, rsn):

    output_path = f"/home/mhoveyda/RSN/predictions_with_z3_results_{today}_{experiment_mode}_{DATA_SAMPLES}.jsonl"

    # if output_path already exists, load the existing results and proceed to evaluate them
    if os.path.exists(output_path):
        with open(output_path) as f:
            data_test = [json.loads(line) for line in f]
    else:

        if experiment_mode == "raw_retrieval":
            # This mode consists of retrieving the top-k documents and evaluating them directly without any further processing.

            # Initialize the IncrementalEvaluation instance
            evaluation = IncrementalEvaluation()

            for i, d in enumerate(data_test):
                for j, doc in enumerate(d["pred_docs_metadata"]):
                    print(f"\n\n----------Query #{i}: {data_test[i]['query']}, Pred #{j}: {doc['doc']}----------")

                    # Update the confusion matrix with the gold label and the predicted label
                    evaluation.update_confusion_matrix(
                        data_test[i]['pred_maps'][doc['doc']],
                        doc['Z3_result']
                    )

                    # Update the precision at k and MRR metrics
                    evaluation.update_P_at_k(
                        data_test[i]['pred_maps'],
                        d["pred_docs_metadata"],
                        k=3
                    )
                    evaluation.update_MRR(
                        data_test[i]['pred_maps'],
                        d["pred_docs_metadata"]
                    )

                    # Print the confusion matrix
                    print(evaluation.get_metrics()['Confusion Matrix'])

                    # Print the average P@k and MRR
                    print(f"Average P@3: {evaluation.get_metrics()['Average P@k']}")
                    print(f"Average MRR: {evaluation.get_metrics()['Average MRR']}")
        if experiment_mode == "LINC":
            # This mode consists of a simultaneous translation of premises and query to Z3 code for *n times followed by a majority voting funcion.

            # for i, d in enumerate(data_test):
            for i, d in enumerate(tqdm(data_test, desc="Processing Queries", unit="query")):

                # if any(not doc.get("wikidata_properties") for doc in d["pred_docs_metadata"]):
                #     print(f"\n- Skipping Query #{i}: Some documents are missing `wikidata_properties`.")
                #     continue  # S
                # for j, doc in enumerate(d["pred_docs_metadata"][0:ENTITY_SAMPLES]):
                # for j, doc in enumerate(d["pred_docs_metadata"]):
                for j, doc in enumerate(tqdm(d["pred_docs_metadata"], desc=f"Docs in Query {i}", leave=False, unit="doc")):


                    print(f"\n\n----------Query #{i}: {data_test[i]['query']}, Pred #{j}: {doc['doc']}----------")

                    attempt = 0

                    list_of_results = []
                    list_of_codes = []
                    while attempt < LINC_MAJ:
                        attempt_start = time.time()  # Start time for the attempt

                        print(f"\n- Attempt {attempt + 1} for doc {j}")

                        # Collect premises and categories
                        doc_metadata = doc["wikidata_properties"]
                        filtered_categories = [cat.replace('Category:', '') for cat in doc['wikidata_categories'] if cat.replace('Category:', '') in re.findall(r'<mark>(.*?)</mark>', data_test[i]['original_query'])]

                        doc_name = rsn.process_doc_name(doc["doc"])

                        premises = []
                        # First check if the doc_metadata is not empty and if it is ignore this query and move to the next one

                        for prop, values in doc_metadata.items():

                            if prop not in rsn.useless_properties and "ID" not in prop and "ISBN" not in prop:

                                prop = prop.replace(" ", "_")

                                if prop == 'publication_date':
                                    values = values[0]
                                    values = [values.split('-').pop(0)]

                                values = [v.replace(" ", "_") for v in values]
                                
                                if isinstance(values, list) and len(values) > 1:
                                    premises.append({"property": prop, "values": values})
                                else:
                                    premises.append({"property": prop, "values": values})
        
                        for cat in filtered_categories:
                            print(f"\n\n- Category: {cat}")
                            premises.append({"property": "example_of", "values": cat.replace(" ", "_")})
                        
                        entity_dict = {
                            "entity": doc_name,
                            "premises": premises
                        }
                
                        print(f"\n\n- Entity with Premises: \n{entity_dict}")
                        
                        # Translate the premises and query to Z3 code simultaneously
                        prompt = """
                            The following is a first-order logic (FOL) problem.
                            The premises are given as a set of first-order logic sentences.
                            The conclusion is a single first-order logic sentence.
                            The task is to translate each of the premises and the conclusion into FOL expressions suitable for evaluation by the Z3 theorem prover.
                            Expressions should be in Python code using the Z3 library and in a python code block.

                            The generated format should follow the example below:

                            <Entity>
                            {
                                'entity': 'The_Sofa_:_A_Moral_Tale',
                                'premises': [
                                    {'property': 'instance_of', 'values': ['written_work']},
                                    {'property': 'author', 'values': ['Claude_Prosper_Jolyot_de_Crébillon']},
                                    {'property': 'genre', 'values': ['fantasy', 'libertine_novel', 'erotic_novel']},
                                    {'property': 'language_of_work_or_name', 'values': ['French']},
                                    {'property': 'country_of_origin', 'values': ['France']},
                                    {'property': 'publication_date', 'values': ['1742']},
                                    {'property': 'example_of', 'values': ['1742_novels']}
                                ]
                            }
                            </Entity>

                            <Query> 1742 or French satirical novels. </Query>

                            <EVALUATE>
                
                            ```python

                            # Python code using Z3
                            from z3 import *

                            # Declare sorts
                            Work = DeclareSort('Work')
                            Person = DeclareSort('Person')
                            Genre = DeclareSort('Genre')
                            Language = DeclareSort('Language')
                            Country = DeclareSort('Country')

                            # Declare constants for entities
                            the_sofa = Const('The_Sofa_A_Moral_Tale', Work)
                            written_work = Const('Written_Work', Work)
                            crebillon = Const('Claude_Prosper_Jolyot_de_Crebillon', Person)
                            fantasy = Const('Fantasy', Genre)
                            libertine_novel = Const('Libertine_Novel', Genre)
                            erotic_novel = Const('Erotic_Novel', Genre)
                            french = Const('French', Language)
                            france = Const('France', Country)
                            satirical_novel = Const('Satirical_Novel', Genre)
                            # Declare predicates
                            InstanceOf = Function('InstanceOf', Work, Work, BoolSort())
                            AuthorOf = Function('AuthorOf', Work, Person, BoolSort())
                            HasGenre = Function('HasGenre', Work, Genre, BoolSort())
                            LanguageOf = Function('LanguageOf', Work, Language, BoolSort())
                            CountryOfOrigin = Function('CountryOfOrigin', Work, Country, BoolSort())
                            PublicationDate = Function('PublicationDate', Work, IntSort(), BoolSort())

                            # Initialize the solver
                            s = Solver()

                            # Add premises to the solver
                            s.add(InstanceOf(the_sofa, written_work))
                            s.add(AuthorOf(the_sofa, crebillon))
                            s.add(HasGenre(the_sofa, fantasy))
                            s.add(HasGenre(the_sofa, libertine_novel))
                            s.add(HasGenre(the_sofa, erotic_novel))
                            s.add(LanguageOf(the_sofa, french))
                            s.add(CountryOfOrigin(the_sofa, france))
                            s.add(PublicationDate(the_sofa, 1742))

                            # Express the conclusion: "1742 or French satirical novels."
                            # Define the conclusion as a Z3 expression
                            conclusion = Or(
                                PublicationDate(the_sofa, 1742),
                                And(
                                    HasGenre(the_sofa, satirical_novel),
                                    CountryOfOrigin(the_sofa, france)
                                )
                            )
                            # Add the negation of the conclusion to the solver to check for unsatisfiability
                            s.add(Not(conclusion))

                            # Check if the premises and the negation of the conclusion are unsatisfiable and print the result.
                            if s.check() == unsat:
                                print("True")
                            else:
                                print("False")
                            ```
                            </EVALUATE>
                        """

                        prompt += f"\n<Entity>\n{entity_dict}\n</Entity>\n"
                        prompt += f"<Query> {data_test[i]['query']} </Query>"
                        prompt += f"\n</EVALUATE>"
                        print(f"\n\n- Prompt: \n{prompt}")

                        output = rsn.generate("", prompt)

                        code = rsn.extract_code(output)
                        list_of_codes.append(code)

                        print(f"\n\n- Code: \n{code}")

                        if code != -1 and code.strip() != "":
                            try:
                                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                                    exec(code)
                                    result = buf.getvalue().strip()
                            except Exception as e:
                                print(f"\n- Error: {e}")
                                result = "CodeError"
                        else:
                            result = "NoCode"
                        
                        list_of_results.append(result)
                        attempt_end = time.time()  # End time for the attempt
                        print(f"- Time for Attempt {attempt + 1}: {attempt_end - attempt_start:.2f} seconds")
                        attempt += 1
                    
                    # Majority voting: get the result with the highest frequency and its code 
                    result = max(set(list_of_results), key = list_of_results.count)
                    complete_code = list_of_codes[list_of_results.index(result)]


                    print(f"\n\n- Final Result: {result}, Gold: {data_test[i]['pred_maps'][doc['doc']]}, List: {list_of_results}")
                    print(f"\n\n- Final Code: \n{complete_code}")

                    data_test[i]["pred_docs_metadata"][j]["Z3_result"] = result
                    data_test[i]["pred_docs_metadata"][j]["Z3_code"] = complete_code

            print(f"\n\n- Writing the results to {output_path}")
            with open(output_path, "w") as f:
                for d in data_test:
                    f.write(json.dumps(d) + "\n")

        if experiment_mode == "SEP_FOL_Z3":
            # This mode consists of 
            # 1. Translating query to FOL
            # 2. Translating premises and FOL query to Z3 code
            # 3. Running the Z3 code and in case of Error, feeding the error and the code to LLM for revising up to N times or until success
            for i, d in enumerate(tqdm(data_test, desc="Processing Queries", unit="query")):
                query_translation_prompt = """
                    You are an advanced reasoning assistant designed to translate natural language queries into structured logical representations. Follow these steps to parse and translate the query into a JSON format that reflects all atomic constraints and logical relationships.

                    1. Decompose the query into atomic constraints:
                    Identify the smallest components of the query, such as specific properties, values, and logical operators (e.g., AND, OR).

                    2. Preserve logical relationships:
                    Use the relationships from the query (e.g., AND, OR, NOT) to structure the constraints hierarchically. Ensure dependencies and groupings are maintained.

                    3. Map properties and values explicitly:
                    For each atomic constraint, define the property (e.g., publication year, genre), the operator (e.g., =, >, <), and the value (e.g., 1742, satirical).

                    4. Output the first-order-logic FOL result in JSON format:
                    Use a nested structure to represent the logical relationships and constraints. Ensure the final structure clearly distinguishes between conditions and subconditions.

                    Here is an example query and how you should format the output:

                    Query:
                    "1742 or French satirical novels"

                    Output JSON:
                    ```JSON
                    {
                        "condition": "OR",
                        "conditions": [
                            {"property": "publication_year", "operator": "=", "value": 1742},
                            {
                                "condition": "AND",
                                "conditions": [
                                    {"property": "genre", "value": "satirical"},
                                    {"property": "country", "value": "France"}
                                ]
                            }
                        ]
                    }
                    ```
                    Now translate the following query:
                """        

                query_translation_prompt += f"\n\n<Query> {data_test[i]['query']} </Query>"

                query_translation_response = rsn.generate("", query_translation_prompt)   
                print(f"\n\n- Query Translation Response: \n{query_translation_response}")         
                query_translation = rsn.extract_JSON(query_translation_response)
                
                if not query_translation:
                    query_translation_response = rsn.generate("", query_translation_prompt)
                    query_translation = rsn.extract_JSON(query_translation_response)
                

                print(f"\n\n- Query Translation: \n{query_translation}")
                
               
                for j, doc in enumerate(tqdm(d["pred_docs_metadata"], desc=f"Docs in Query {i}", leave=False, unit="doc")):
                    print(f"\n\n----------Query #{i}: {data_test[i]['query']}, Pred #{j}: {doc['doc']}----------")

                    attempt = 0

                    while attempt < MAX_ATTEMPTS:
                        attempt_start = time.time()

                        print(f"\n- Attempt {attempt + 1} for doc {j}")

                        # Collect premises and categories
                        doc_metadata = doc["wikidata_properties"]
                        filtered_categories = [cat.replace('Category:', '') for cat in doc['wikidata_categories'] if cat.replace('Category:', '') in re.findall(r'<mark>(.*?)</mark>', data_test[i]['original_query'])]

                        doc_name = rsn.process_doc_name(doc["doc"])

                        premises = []


                        for prop, values in doc_metadata.items():

                            if prop not in rsn.useless_properties and "ID" not in prop and "ISBN" not in prop:

                                prop = prop.replace(" ", "_")

                                if prop == 'publication_date':
                                    values = values[0]
                                    values = [values.split('-').pop(0)]

                                values = [v.replace(" ", "_") for v in values]
                                
                                if isinstance(values, list) and len(values) > 1:
                                    premises.append({"property": prop, "values": values})
                                else:
                                    premises.append({"property": prop, "values": values})
        
                        for cat in filtered_categories:
                            print(f"\n\n- Category: {cat}")
                            premises.append({"property": "example_of", "values": cat.replace(" ", "_")})
                        
                        entity_dict = {
                            "entity": doc_name,
                            "premises": premises
                        }
                
                        print(f"\n\n- Entity with Premises: \n{entity_dict}")
                        
                        # Translate the premises and FOL query to Z3 code

                        full_translation_prompt = """

                            The following is a first-order logic (FOL) problem.
                            The premises are given as a set of logical statements.
                            The query is translated to a first-order logic (FOL) structure.
                            The task is to translate the given premises and the FOL conclusion to a coherent Z3 Python code block for evaluation. 

                            The generated format should follow the example below:
                            <Entity>
                            {
                                'entity': 'The_Sofa_:_A_Moral_Tale',
                                'premises': [
                                    {'property': 'instance_of', 'values': ['written_work']},
                                    {'property': 'author', 'values': ['Claude_Prosper_Jolyot_de_Crébillon']},
                                    {'property': 'genre', 'values': ['fantasy', 'libertine_novel', 'erotic_novel']},
                                    {'property': 'language_of_work_or_name', 'values': ['French']},
                                    {'property': 'country_of_origin', 'values': ['France']},
                                    {'property': 'publication_date', 'values': ['1742']},
                                    {'property': 'example_of', 'values': ['1742_novels']}
                                ]
                            }
                            </Entity>
                            <Query> 
                            {
                                "condition": "OR",
                                "conditions": [
                                    {"property": "publication_year", "operator": "=", "value": 1742},
                                    {
                                        "condition": "AND",
                                        "conditions": [
                                            {"property": "genre", "value": "satirical"},
                                            {"property": "country", "value": "France"}
                                        ]
                                    }
                                ]
                            }
                            </Query>

                            <EVALUATE>
                
                            ```python
                            # Python code using Z3
                            from z3 import *

                            # Declare sorts
                            Work = DeclareSort('Work')
                            Person = DeclareSort('Person')
                            Genre = DeclareSort('Genre')
                            Language = DeclareSort('Language')
                            Country = DeclareSort('Country')

                            # Declare constants for entities
                            the_sofa = Const('The_Sofa_A_Moral_Tale', Work)
                            written_work = Const('Written_Work', Work)
                            crebillon = Const('Claude_Prosper_Jolyot_de_Crebillon', Person)
                            fantasy = Const('Fantasy', Genre)
                            libertine_novel = Const('Libertine_Novel', Genre)
                            erotic_novel = Const('Erotic_Novel', Genre)
                            french = Const('French', Language)
                            france = Const('France', Country)
                            satirical_novel = Const('Satirical_Novel', Genre)
                            # Declare predicates
                            InstanceOf = Function('InstanceOf', Work, Work, BoolSort())
                            AuthorOf = Function('AuthorOf', Work, Person, BoolSort())
                            HasGenre = Function('HasGenre', Work, Genre, BoolSort())
                            LanguageOf = Function('LanguageOf', Work, Language, BoolSort())
                            CountryOfOrigin = Function('CountryOfOrigin', Work, Country, BoolSort())
                            PublicationDate = Function('PublicationDate', Work, IntSort(), BoolSort())

                            # Initialize the solver
                            s = Solver()

                            # Add premises to the solver
                            s.add(InstanceOf(the_sofa, written_work))
                            s.add(AuthorOf(the_sofa, crebillon))
                            s.add(HasGenre(the_sofa, fantasy))
                            s.add(HasGenre(the_sofa, libertine_novel))
                            s.add(HasGenre(the_sofa, erotic_novel))
                            s.add(LanguageOf(the_sofa, french))
                            s.add(CountryOfOrigin(the_sofa, france))
                            s.add(PublicationDate(the_sofa, 1742))

                            # Express the conclusion: "1742 or French satirical novels."
                            # Define the conclusion as a Z3 expression
                            conclusion = Or(
                                PublicationDate(the_sofa, 1742),
                                And(
                                    HasGenre(the_sofa, satirical_novel),
                                    CountryOfOrigin(the_sofa, france)
                                )
                            )
                            # Add the negation of the conclusion to the solver to check for unsatisfiability
                            s.add(Not(conclusion))

                            # Check if the premises and the negation of the conclusion are unsatisfiable and print the result.
                            if s.check() == unsat:
                                print("True")
                            else:
                                print("False")
                            ```
                            </EVALUATE>


                            Now, translate the following premises and query to Z3 Python code:

                            """
                        full_translation_prompt += f"\n<Entity>\n{entity_dict}\n</Entity>\n"
                        full_translation_prompt += f"<Query> {query_translation} </Query>"
                        full_translation_prompt += f"\n<EVALUATE>"

                        print(f"\n\n- Full Translation Prompt: \n{full_translation_prompt}")

                        full_translation_response = rsn.generate("", full_translation_prompt)
                        full_translation_code = rsn.extract_code(full_translation_response)
                        print(f"\n\n- Full Translation Code: \n{full_translation_code}")

                        if full_translation_code != -1 and full_translation_code.strip() != "":
                            try:
                                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                                    exec(full_translation_code)
                                    result = buf.getvalue().strip()
                            except Exception as e:
                                print(f"\n- Error: {e}")
                                result = "CodeError"
                        else:
                            result = "NoCode"
                        
                        if result.lower().strip() in ["true", "false"]:
                            break
                        else:
                            attempt += 1
                            print(f"\n- Attempt {attempt} failed. Retrying...")
                    
                    print(f"\n\n- Final Result: {result}, Gold: {data_test[i]['pred_maps'][doc['doc']]}")
                    print(f"\n\n- Final Code: \n{full_translation_code}")
                    data_test[i]["pred_docs_metadata"][j]["Z3_result"] = result
                    data_test[i]["pred_docs_metadata"][j]["Z3_code"] = full_translation_code

                    attempt_end = time.time()
                    print(f"- Time for Attempt {attempt}: {attempt_end - attempt_start:.2f} seconds")
            
            print(f"\n\n- Writing the results to {output_path}")
            with open(output_path, "w") as f:
                for d in data_test:
                    f.write(json.dumps(d) + "\n")
            
        if experiment_mode == "Logic_LM":
            for i, d in enumerate(tqdm(data_test, desc="Processing Queries", unit="query")):
                for j, doc in enumerate(tqdm(d["pred_docs_metadata"], desc=f"Docs in Query {i}", leave=False, unit="doc")):


                    print(f"\n\n----------Query #{i}: {data_test[i]['query']}, Pred #{j}: {doc['doc']}----------")

                    attempt = 0
                    result = None
                    code = None
                    while attempt < MAX_ATTEMPTS:
                        attempt_start = time.time()  # Start time for the attempt

                        print(f"\n- Attempt {attempt + 1} for doc {j}")
                        if attempt == 0:

                            # Collect premises and categories
                            doc_metadata = doc["wikidata_properties"]
                            filtered_categories = [cat.replace('Category:', '') for cat in doc['wikidata_categories'] if cat.replace('Category:', '') in re.findall(r'<mark>(.*?)</mark>', data_test[i]['original_query'])]

                            doc_name = rsn.process_doc_name(doc["doc"])

                            premises = []

                            for prop, values in doc_metadata.items():

                                if prop not in rsn.useless_properties and "ID" not in prop and "ISBN" not in prop:

                                    prop = prop.replace(" ", "_")

                                    if prop == 'publication_date':
                                        values = values[0]
                                        values = [values.split('-').pop(0)]

                                    values = [v.replace(" ", "_") for v in values]
                                    
                                    if isinstance(values, list) and len(values) > 1:
                                        premises.append({"property": prop, "values": values})
                                    else:
                                        premises.append({"property": prop, "values": values})
            
                            for cat in filtered_categories:
                                print(f"\n\n- Category: {cat}")
                                premises.append({"property": "example_of", "values": cat.replace(" ", "_")})
                            
                            entity_dict = {
                                "entity": doc_name,
                                "premises": premises
                            }
                    
                            print(f"\n\n- Entity with Premises: \n{entity_dict}")
                        
                            # Translate the premises and query to Z3 code simultaneously
                            prompt = """
                                The following is a first-order logic (FOL) problem.
                                The premises are given as a set of first-order logic sentences.
                                The conclusion is a single first-order logic sentence.
                                The task is to translate each of the premises and the conclusion into FOL expressions suitable for evaluation by the Z3 theorem prover.
                                Expressions should be in Python code using the Z3 library and in a python code block.

                                The generated format should follow the example below:

                                <Entity>
                                {
                                    'entity': 'The_Sofa_:_A_Moral_Tale',
                                    'premises': [
                                        {'property': 'instance_of', 'values': ['written_work']},
                                        {'property': 'author', 'values': ['Claude_Prosper_Jolyot_de_Crébillon']},
                                        {'property': 'genre', 'values': ['fantasy', 'libertine_novel', 'erotic_novel']},
                                        {'property': 'language_of_work_or_name', 'values': ['French']},
                                        {'property': 'country_of_origin', 'values': ['France']},
                                        {'property': 'publication_date', 'values': ['1742']},
                                        {'property': 'example_of', 'values': ['1742_novels']}
                                    ]
                                }
                                </Entity>

                                <Query> 1742 or French satirical novels. </Query>

                                <EVALUATE>
                    
                                ```python

                                # Python code using Z3
                                from z3 import *

                                # Declare sorts
                                Work = DeclareSort('Work')
                                Person = DeclareSort('Person')
                                Genre = DeclareSort('Genre')
                                Language = DeclareSort('Language')
                                Country = DeclareSort('Country')

                                # Declare constants for entities
                                the_sofa = Const('The_Sofa_A_Moral_Tale', Work)
                                written_work = Const('Written_Work', Work)
                                crebillon = Const('Claude_Prosper_Jolyot_de_Crebillon', Person)
                                fantasy = Const('Fantasy', Genre)
                                libertine_novel = Const('Libertine_Novel', Genre)
                                erotic_novel = Const('Erotic_Novel', Genre)
                                french = Const('French', Language)
                                france = Const('France', Country)
                                satirical_novel = Const('Satirical_Novel', Genre)
                                # Declare predicates
                                InstanceOf = Function('InstanceOf', Work, Work, BoolSort())
                                AuthorOf = Function('AuthorOf', Work, Person, BoolSort())
                                HasGenre = Function('HasGenre', Work, Genre, BoolSort())
                                LanguageOf = Function('LanguageOf', Work, Language, BoolSort())
                                CountryOfOrigin = Function('CountryOfOrigin', Work, Country, BoolSort())
                                PublicationDate = Function('PublicationDate', Work, IntSort(), BoolSort())

                                # Initialize the solver
                                s = Solver()

                                # Add premises to the solver
                                s.add(InstanceOf(the_sofa, written_work))
                                s.add(AuthorOf(the_sofa, crebillon))
                                s.add(HasGenre(the_sofa, fantasy))
                                s.add(HasGenre(the_sofa, libertine_novel))
                                s.add(HasGenre(the_sofa, erotic_novel))
                                s.add(LanguageOf(the_sofa, french))
                                s.add(CountryOfOrigin(the_sofa, france))
                                s.add(PublicationDate(the_sofa, 1742))

                                # Express the conclusion: "1742 or French satirical novels."
                                # Define the conclusion as a Z3 expression
                                conclusion = Or(
                                    PublicationDate(the_sofa, 1742),
                                    And(
                                        HasGenre(the_sofa, satirical_novel),
                                        CountryOfOrigin(the_sofa, france)
                                    )
                                )
                                # Add the negation of the conclusion to the solver to check for unsatisfiability
                                s.add(Not(conclusion))

                                # Check if the premises and the negation of the conclusion are unsatisfiable and print the result.
                                if s.check() == unsat:
                                    print("True")
                                else:
                                    print("False")
                                ```
                                </EVALUATE>
                            """

                            prompt += f"\n<Entity>\n{entity_dict}\n</Entity>\n"
                            prompt += f"<Query> {data_test[i]['query']} </Query>"
                            prompt += f"\n</EVALUATE>"
                            print(f"\n\n- Prompt: \n{prompt}")

                            output = rsn.generate("", prompt)

                            code = rsn.extract_code(output)
                            print(f"\n\n- Code: \n{code}")
                        # Attempt to execute the code
                        if code and code.strip():
                            try:
                                # Execute the code and capture the result
                                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                                    exec(code)
                                    result = buf.getvalue().strip()
                            except Exception as e:
                                error_received = str(e)
                                print(f"\n- Execution Error: {e}")
                                result = "CodeError"
                        else:
                            result = "NoCode"

                        # Check if the execution result is valid
                        if result.lower().strip() in ["true", "false"]:
                            print("\n- Code executed successfully.")
                            break
                        else:
                            print(f"\n- Attempt {attempt + 1} failed. Revising code with LLM...")

                            # Construct the prompt to revise the code
                            revision_prompt = f"""
                            The following code was generated for translating the premises and query to Z3 Python code:

                            {code}

                            But it failed to execute successfully. Error encountered: {error_received}

                            Please revise the code and ensure it is executable. Provide the corrected Python code in a code block.
                            """
                            revision_response = rsn.generate("", revision_prompt)
                            code = rsn.extract_code(revision_response)
                            print(f"\n\n- Revised Code: \n{code}")

                        # Increment attempt counter and track execution time
                        attempt += 1
                        attempt_end = time.time()
                        print(f"- Time for Attempt {attempt}: {attempt_end - attempt_start:.2f} seconds")

                    # Finalize and store results
                    print(f"\n\n- Final Result: {result}, Gold: {data_test[i]['pred_maps'][doc['doc']]}")
                    print(f"\n\n- Final Code: \n{code}")

                    data_test[i]["pred_docs_metadata"][j]["Z3_result"] = result
                    data_test[i]["pred_docs_metadata"][j]["Z3_code"] = code
                                    
            print(f"\n\n- Writing the results to {output_path}")
            with open(output_path, "w") as f:
                for d in data_test:
                    f.write(json.dumps(d) + "\n")


    with open(output_path, "r") as f:
        final_data = [json.loads(line) for line in f]
    # Initialize the IncrementalEvaluation instance
    # evaluation_path = f"{output_path.replace('.jsonl', '_evaluation.txt')}"
    evaluation_path = f"{output_path.replace('.jsonl', f'_evaluation_{experiment_mode}.json')}"

    evaluation_results = calculate_metrics(final_data, experiment_mode)
    for method in evaluation_results:
        print(f"Metrics for {method}:")
        for metric, value in evaluation_results[method].items():
            print(f"  {metric}: {value:.4f}")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(evaluation_results, indent=4))
    print(f"\n\n- Writing the evaluation results to {evaluation_path}")




# run_experiment("LINC", data_test, rsn)
# run_experiment("SEP_FOL_Z3", data_test, rsn)
run_experiment("Logic_LM", data_test, rsn)











