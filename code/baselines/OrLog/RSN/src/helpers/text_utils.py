import re
import json

def _sanitize_predicate(atom: str) -> str:
    pred = re.sub(r"[^a-z0-9]", "_", atom.replace("{x}", "").strip().lower())
    return re.sub(r"_+", "_", pred).strip("_")

def _sanitize_entity(entity: str) -> str:
    ent = re.sub(r"[^a-z0-9]", "_", entity.lower())
    return re.sub(r"_+", "_", ent).strip("_")

def _sanitize(s: str) -> str:
    """turn arbitrary string into a safe filename token"""
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

def extract_answer_with_regex(output: str) -> str:
    """
    Extract the 'Answer' value ('True' or 'False') from varied LLM output formats.
    Raises ValueError if no valid answer is found.
    """
    # 1. Attempt to extract a JSON block by finding the first '{' and its matching '}'
    start = output.find('{')
    if start != -1:
        depth = 0
        end = -1
        for idx, char in enumerate(output[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end != -1:
            # Extract and sanitize the JSON-like substring
            json_str = output[start:end+1].strip('` \n\r\t')
            # 2. Try standard JSON parsing
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # 3. Attempt to fix single quotes, then parse again
                try:
                    data = json.loads(json_str.replace("'", '"'))
                except json.JSONDecodeError:
                    # 4. Fallback to Python literal eval (handles True/False without quotes)
                    data = ast.literal_eval(json_str)
            # 5. Validate the parsed object
            if isinstance(data, dict) and 'Answer' in data:
                val = data['Answer']
                result = str(val).strip()
                if result in ("True", "False"):
                    return result
                else:
                    raise ValueError(f"Invalid Answer value: {result}")

    # 6. Fallback: search for a plain-text "Answer: True/False"
    match = re.search(r'Answer\s*[:=]\s*[\'"]?(True|False)[\'"]?', output, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    # 7. No valid answer found
    raise ValueError("No 'Answer' found in output")
