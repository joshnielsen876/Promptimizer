import json
import re

def extract_json_string(complex_string):
    # Use regex to extract the JSON part
    match = re.search(r'(\{.*\})', complex_string)
    if match:
        return match.group(1)
    return None

def clean_json_string(json_string):
    # Attempt to fix common issues, such as removing invalid escape characters
    json_string = json_string.replace('\n', '')
    json_string = json_string.replace('\r', '')
    json_string = json_string.replace('\t', '')
    json_string = json_string.replace('\\', '')
    return json_string

def parse_json_string(json_string):
    try:
        # Try to parse the JSON string
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # If an error occurs, clean the JSON string and try again
        print(f"Error: {e}")
        cleaned_json_string = clean_json_string(json_string)
        try:
            return json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            print(f"Second Error: {e}")
            return None

# Example complex string containing JSON
complex_string = 'Some prefix text {"New Prompt": "<generated text>"} some suffix text'

# Extract JSON string
json_string = extract_json_string(complex_string)
if json_string:
    # Parse the extracted JSON string
    parsed_json = parse_json_string(json_string)
    if parsed_json:
        # Extract the generated text portion
        generated_text = parsed_json["New Prompt"]
        print(generated_text)
    else:
        print("Failed to parse JSON after cleaning.")
else:
    print("No JSON part found in the string.")
