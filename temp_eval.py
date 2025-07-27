import requests
import json
import random
from v1 import extract_date_from_response
import pandas as pd
from tqdm import tqdm


def get_ollama_responses(prompt, model, temperature, max_tokens=1000):
    seed = random.randint(0, 2**32 - 1)
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        "stream": False,
        "seed": seed
    }
    response = requests.post(url, json=data)
    return response.json()["response"]

def import_csv_as_df(file_path):
    return pd.read_csv(file_path)

def event_to_llm_date(event, context, model, temperature):
    prompt = f"{context}\n {event}"
    response = get_ollama_responses(prompt, model, temperature)
    return extract_date_from_response(response)

def test_event_n_times(event, context, model, temperature, n=100):
    """
    Test the model's ability to provide correct date for a given event. Output the accuracy percentage from n tests.
    """
    for i in range(n):
        date = event_to_llm_date(event, context, model, temperature)
        if date == event["date"]:
            correct_count += 1
    return correct_count / n
    
def add_temp_n_accuracy_to_df(df, model, temperature, context, start_index=0, end_index=None):
    # Determine the range of indices to process
    if end_index is None:
        end_index = len(df)
    # Ensure indices are within bounds
    start_index = max(0, start_index)
    end_index = min(len(df), end_index)

    for index in range(start_index, end_index):
        row = df.iloc[index]
        event = row["event"]
        accuracy = test_event_n_times(event, context, model, temperature)
        df.at[index, "accuracy"] = accuracy
        df.at[index, "temperature"] = temperature
    return df

# parameters
model = "gemma3n:e4b"
temperatures = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
context = "On what date did the following event occur?"

def main():
    df = import_csv_as_df("historical_events.csv")
    df = add_temp_n_accuracy_to_df(df, "gemma3n:e4b", 0.5)