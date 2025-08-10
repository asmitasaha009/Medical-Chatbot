from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fuzzywuzzy import process, fuzz
import mysql.connector
import re
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Database Configuration ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "sql@22112003"),
    "database": os.getenv("DB_NAME", "lab_test_db")
}

def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        raise ConnectionError(f"Failed to connect to database: {err}")

# --- Keyword Mapping and Entity Recognition ---
COLUMN_MAPPING = {
    "patient id": "patient_id",
    "patient": "patient_id",
    "condition": "`Condition`",
    "test date": "test_date",
    "date": "test_date",
    "test name": "test_name",
    "test": "test_name",
    "test value": "test_value",
    "value": "test_value",
    "unit": "unit",
    "reference low": "ref_low",
    "ref low": "ref_low",
    "reference high": "ref_high",
    "ref high": "ref_high",
    "interpretation": "interpretation",
    "result": "interpretation",
}

POSSIBLE_FILTER_VALUES = {
    "interpretation": ["high", "normal", "low"],
    "Condition": ["Diabetes", "Hypertension", "Anemia", "High_Cholesterol", "Fit"],
    "test_name": ["Blood_glucose", "Cholesterol_Level"],
    "unit": ["mg/dL", "mmol/L"]
}

# NEW: Define synonyms/aliases for test names
TEST_NAME_ALIASES = {
    "blood sugar": "Blood_glucose",
    "sugar level": "Blood_glucose",
    "cholesterol level": "Cholesterol_Level",
    "cholesterol test": "Cholesterol_Level",
    # Add more as needed: "user_alias": "Database_test_name"
}

DEFAULT_SELECT_COLUMNS = ["patient_id", "`Condition`", "test_date", "test_name", "test_value", "unit", "ref_low", "ref_high", "interpretation"]
# Update ALL_VALID_TERMS to include these aliases for correct_spelling
ALL_VALID_TERMS = list(COLUMN_MAPPING.keys()) + \
                    [item for sublist in POSSIBLE_FILTER_VALUES.values() for item in sublist] + \
                    list(TEST_NAME_ALIASES.keys()) + \
                    ["show", "get", "tell", "what", "is", "of", "for", "with", "has", "are", "records", "results", "info", "about"]

def correct_spelling(text: str) -> str:
    """Corrects spelling of words in text based on ALL_VALID_TERMS."""
    corrected_words = []
    words = text.lower().split()
    for word in words:
        match = process.extractOne(word, ALL_VALID_TERMS, scorer=fuzz.ratio)
        if match and match[1] > 80: # Adjust threshold as needed, 80 is usually good for synonyms
            corrected_words.append(match[0])
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

# In main.py, modify the extract_keywords function as shown below:

def extract_keywords(prompt: str):
    """
    Extracts relevant filters and columns of interest from the prompt.
    Returns a dictionary with 'filters' and 'columns_of_interest'
    """
    corrected_prompt = correct_spelling(prompt)
    lower_prompt = corrected_prompt.lower()

    extracted = {
        "filters": {},
        "columns_of_interest": set()
    }

    # --- 1. Extract Specific Filter Values ---

    # Patient ID: Modified regex to include "about" and "of"
    patient_id_match = re.search(r"(?:patient (?:id\s*)?|id\s+|for\s*|about\s*|of\s*)(\d+)", lower_prompt)
    if patient_id_match:
        extracted["filters"]["patient_id"] = int(patient_id_match.group(1))
        extracted["columns_of_interest"].add("patient_id")

    # --- Prioritize Test Name extraction (direct and aliases) ---
    test_name_identified = False
    
    # Check aliases first for strong signal
    for alias, canonical_name in TEST_NAME_ALIASES.items():
        regex_alias = r"\b(?:" + re.escape(alias) + r"(?:\s+test|\s+report|\s+level)?|test\s+name\s+" + re.escape(alias) + r")\b"
        if re.search(regex_alias, lower_prompt):
            extracted["filters"]["test_name"] = canonical_name
            extracted["columns_of_interest"].add("test_name")
            test_name_identified = True
            break # Found a test name via alias, prioritize it

    # If no alias matched, try direct canonical test names
    if not test_name_identified:
        for val in POSSIBLE_FILTER_VALUES["test_name"]:
            regex_val = r"\b(?:" + re.escape(val.lower().replace("_", " ")) + r"(?:\s+test|\s+report|\s+level)?|test\s+name\s+" + re.escape(val.lower().replace("_", " ")) + r")\b"
            if re.search(regex_val, lower_prompt):
                extracted["filters"]["test_name"] = val
                extracted["columns_of_interest"].add("test_name")
                test_name_identified = True
                break
    
    # Interpretation (moved below test_name for potential interaction)
    interpretation_match = re.search(r"(?:interpretation|result) (?:is\s+)?(high|normal|low)", lower_prompt)
    if interpretation_match:
        extracted["filters"]["interpretation"] = interpretation_match.group(1).upper()
        extracted["columns_of_interest"].add("interpretation")

    # Condition - Re-evaluate regex to be less greedy if test_name is present
    if not test_name_identified: # Only try to extract Condition if test_name wasn't already found
        for val in POSSIBLE_FILTER_VALUES["Condition"]:
            # Make sure this regex doesn't accidentally pick up "cholesterol" as a condition if "cholesterol level" was meant as a test name
            regex_pattern = r"\b(?:" + re.escape(val.lower()) + r"(?:\s+condition|\s+patients?)?|condition\s+is\s+" + re.escape(val.lower()) + r")\b"
            if re.search(regex_pattern, lower_prompt):
                extracted["filters"]["Condition"] = val
                extracted["columns_of_interest"].add("`Condition`")
                break

    # --- 2. Identify Columns of Interest (refined logic) ---
    # Process general column keywords
    for user_term, db_col in COLUMN_MAPPING.items():
        if user_term in lower_prompt:
            extracted["columns_of_interest"].add(db_col)

    # Ensure test_value and unit are always added if test_name is relevant,
    # whether it's a filter or a requested column.
    if test_name_identified or "test_name" in extracted["columns_of_interest"]:
        extracted["columns_of_interest"].add("test_value")
        extracted["columns_of_interest"].add("unit")

    # Default to all important columns if very general query or only patient_id specified
    if not extracted["columns_of_interest"] or (len(extracted["columns_of_interest"]) == 1 and "patient_id" in extracted["columns_of_interest"] and not extracted["filters"]):
        extracted["columns_of_interest"].update([c.replace("`","") for c in DEFAULT_SELECT_COLUMNS])
    else:
        # Always include patient_id if other columns are specified, for context
        if "patient_id" not in extracted["columns_of_interest"]:
            extracted["columns_of_interest"].add("patient_id")

    # Clean up backticks for final set comparison
    extracted["columns_of_interest"] = {col.replace("`", "") for col in extracted["columns_of_interest"]}
    
    return extracted
def build_sql_query(extracted_info: dict, offset: int = 0, limit: int = 50):
    """
    Builds a SQL query based on extracted filters and columns of interest.
    Includes OFFSET and LIMIT for pagination.
    Returns data_query, data_params, count_records_query, count_records_params, count_patients_query, count_patients_params
    """
    filters = extracted_info["filters"]
    columns_of_interest = extracted_info["columns_of_interest"]

    select_cols_sql = []
    ordered_cols_template = ["patient_id", "Condition", "test_date", "test_name", "test_value", "unit", "ref_low", "ref_high", "interpretation"]
    
    for col in ordered_cols_template:
        if col in columns_of_interest:
            select_cols_sql.append(f"`{col}`" if col == "Condition" else col)
    
    if not select_cols_sql:
        select_cols_sql = DEFAULT_SELECT_COLUMNS
        columns_of_interest.update([c.replace("`","") for c in DEFAULT_SELECT_COLUMNS])

    select_cols_sql = list(dict.fromkeys(select_cols_sql))

    base_query_no_limit = f"SELECT {', '.join(select_cols_sql)} FROM lab_test"

    conditions = []
    params = []

    if "patient_id" in filters:
        conditions.append("patient_id = %s")
        params.append(filters["patient_id"])
    
    if "Condition" in filters:
        conditions.append("`Condition` = %s")
        params.append(filters["Condition"])
    
    if "test_name" in filters:
        conditions.append("test_name = %s")
        params.append(filters["test_name"])
    
    if "interpretation" in filters:
        conditions.append("interpretation = %s")
        params.append(filters["interpretation"])
    
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)
    
    # Query to get total count of unique *records/sentences* (current behavior)
    count_records_query = f"SELECT COUNT(DISTINCT patient_id, `Condition`, test_name, test_value, unit, interpretation) FROM lab_test {where_clause}"
    count_records_params = params.copy()

    # NEW: Query to get total count of unique *patients*
    count_patients_query = f"SELECT COUNT(DISTINCT patient_id) FROM lab_test {where_clause}"
    count_patients_params = params.copy() # Same parameters as other queries

    # Query for actual data
    data_query = f"{base_query_no_limit}{where_clause} LIMIT %s OFFSET %s"
    data_params = params + [limit, offset]

    return data_query, tuple(data_params), count_records_query, tuple(count_records_params), count_patients_query, tuple(count_patients_params)


def format_records_for_display(results: list[tuple], cursor_description: list, extracted_info: dict) -> list[str]:
    """
    Formats SQL query results into a list of human-readable sentences,
    with deduplication for the *final generated sentence*.
    """
    if not results:
        return []

    col_names = [col[0] for col in cursor_description]
    
    final_sentences = []
    unique_sentences_set = set() # Changed: Deduplicate based on the full generated sentence
    
    filters = extracted_info.get("filters", {})
    columns_of_interest = extracted_info.get("columns_of_interest", set())

    for row_tuple in results:
        row_data = dict(zip(col_names, row_tuple))

        parts = []
        
        if 'patient_id' in row_data:
            parts.append(f"Patient {row_data['patient_id']}")

        if 'Condition' in row_data and row_data['Condition']:
            if row_data['Condition'].lower() == 'fit' and 'Condition' not in filters and 'Condition' not in columns_of_interest:
                pass
            elif row_data['Condition'] != 'Fit' or 'Condition' in filters or 'Condition' in columns_of_interest:
                parts.append(f"has a '{row_data['Condition'].replace('_', ' ')}' condition")

        test_details_added = False
        if 'test_name' in row_data and row_data['test_name']:
            test_info = f"had a '{row_data['test_name'].replace('_', ' ')}' test"
            
            # Check for test_value and unit existence and not None
            if 'test_value' in row_data and row_data['test_value'] is not None:
                test_info += f" with value {row_data['test_value']}"
                if 'unit' in row_data and row_data['unit']:
                    test_info += f" {row_data['unit']}"
            parts.append(test_info)
            test_details_added = True

        if 'interpretation' in row_data and row_data['interpretation']:
            if test_details_added: 
                parts.append(f"which was interpreted as '{row_data['interpretation']}'")
            elif 'interpretation' in filters or 'interpretation' in columns_of_interest:
                parts.append(f"has an interpretation of '{row_data['interpretation']}'")

        candidate_sentence = "" # Initialize here
        if len(parts) == 1 and 'patient_id' in row_data:
            candidate_sentence = f"I found data for {parts[0]}."
        elif parts:
            sentence_core = parts[0]
            if len(parts) > 1:
                if len(parts) == 2:
                    sentence_core += f" and {parts[1]}"
                else:
                    sentence_core += ", " + ", ".join(parts[1:-1]) + f", and {parts[-1]}"
            candidate_sentence = sentence_core + "."
        
        # Deduplicate based on the full generated sentence
        if candidate_sentence and candidate_sentence not in unique_sentences_set:
            unique_sentences_set.add(candidate_sentence)
            final_sentences.append(candidate_sentence)
        
    return final_sentences

# --- History Saving Function ---
def save_chat_history(user_prompt: str, bot_response: str):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "INSERT INTO chat_history (user_prompt, bot_response) VALUES (%s, %s)"
        cursor.execute(query, (user_prompt, bot_response))
        conn.commit()
        print(f"Chat history saved: User='{user_prompt}', Bot='{bot_response}'")
    except ConnectionError as ce:
        print(f"ERROR: Could not save chat history. Database connection failed: {ce}")
    except mysql.connector.Error as err:
        print(f"ERROR: Failed to save chat history to DB: {err}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while saving chat history: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_class=JSONResponse)
async def chat(request: Request, 
               user_prompt: str = Form(...),
               offset: int = Form(0),
               limit: int = Form(5)): # This limit is the requested page size
    
    connection = None
    cursor = None
    
    response_data = {
        "messages": [],
        "has_more": False,
        "next_offset": 0,
        "total_records": 0, # Renamed to reflect unique records/sentences
        "total_unique_patients": 0 # For total unique patients
    }

    try:
        extracted_info = extract_keywords(user_prompt)
        
        if not extracted_info["filters"] and not any(col in user_prompt.lower() for col in COLUMN_MAPPING.keys()):
            response_data["messages"].append("I couldn't understand your request. Please ask about patient data, conditions, or test results (e.g., 'What is the blood glucose for patient 1?', 'Show me patients with diabetes').")
        else:
            data_query, data_params, count_records_query, count_records_params, count_patients_query, count_patients_params = \
                build_sql_query(extracted_info, offset, limit)
            
            print(f"SQL Data Query: {data_query} with Params: {data_params}")
            print(f"SQL Records Count Query: {count_records_query} with Params: {count_records_params}")
            print(f"SQL Patients Count Query: {count_patients_query} with Params: {count_patients_params}") # Debugging
            
            connection = get_db_connection()
            cursor = connection.cursor()

            # Execute patient count query
            cursor.execute(count_patients_query, count_patients_params)
            total_unique_patients = cursor.fetchone()[0]
            print(f"DEBUG: Total unique patients found: {total_unique_patients}")
            response_data["total_unique_patients"] = total_unique_patients

            # Execute records count query (for pagination purposes)
            cursor.execute(count_records_query, count_records_params)
            total_records_count = cursor.fetchone()[0]
            print(f"DEBUG: Total unique records found: {total_records_count}")
            response_data["total_records"] = total_records_count

            # Execute data query
            cursor.execute(data_query, data_params)
            results = cursor.fetchall()
            
            print(f"DEBUG: SQL Results (raw for current page): {results}")
            if cursor.description:
                print(f"DEBUG: Cursor Description: {[col[0] for col in cursor.description]}")

            formatted_sentences = format_records_for_display(results, cursor.description, extracted_info)
            response_data["messages"].extend(formatted_sentences)

            current_displayed_unique_sentences = len(formatted_sentences)

            if (offset + current_displayed_unique_sentences) < total_records_count:
                response_data["has_more"] = True
                response_data["next_offset"] = offset + current_displayed_unique_sentences
            else:
                response_data["has_more"] = False
                response_data["next_offset"] = offset + current_displayed_unique_sentences

            if not formatted_sentences and offset == 0: 
                 response_data["messages"].append("I couldn't find any information matching your request. Could you please rephrase or provide more details?")

    except ConnectionError as ce:
        print(f"Connection Error: {ce}")
        response_data["messages"].append(f"I'm having trouble connecting to the database. Please ensure the MySQL server is running and the database configuration is correct. Details: {ce}")
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        response_data["messages"].append(f"I encountered a database error: {err}. Please ensure your query is valid. Details: {err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        response_data["messages"].append(f"An unexpected error occurred: {e}. Please check the server logs for more details or try rephrasing your request.")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

    initial_bot_response_for_history = response_data["messages"][0] if response_data["messages"] else "No response generated."
    save_chat_history(user_prompt, initial_bot_response_for_history)

    return JSONResponse(content=response_data)


@app.get("/history", response_class=JSONResponse)
async def get_history():
    connection = None
    cursor = None
    history = []
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = "SELECT user_prompt, bot_response, timestamp FROM chat_history ORDER BY timestamp ASC LIMIT 500"
        cursor.execute(query)
        for row in cursor.fetchall():
            history.append({
                "user": row["user_prompt"],
                "bot": row["bot_response"],
                "timestamp": row["timestamp"].isoformat()
            })
    except ConnectionError as ce:
        print(f"ERROR: Could not retrieve history. Database connection failed: {ce}")
        return JSONResponse(content={"error": "Failed to connect to database for history."}, status_code=500)
    except mysql.connector.Error as err:
        print(f"ERROR: Failed to retrieve chat history from DB: {err}")
        return JSONResponse(content={"error": f"Database error retrieving history: {err}"}, status_code=500)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while retrieving history: {e}")
        return JSONResponse(content={"error": f"An unexpected error occurred: {e}"}, status_code=500)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
    
    return JSONResponse(content=history)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)