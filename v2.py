import requests
import json
import re
from typing import Optional, List, Dict, Tuple
from datetime import datetime, date
import random
from dataclasses import dataclass

@dataclass
class HistoricalFact:
    event: str
    date: str
    year: int
    month: int
    day: int
    category: str = "general"

@dataclass
class TestResult:
    fact: HistoricalFact
    question: str
    model_response: str
    extracted_date: Optional[str]
    is_correct: bool
    confidence_score: float = 0.0

def query_ollama(
    prompt: str,
    model: str = "gemma3:1b",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    stream: bool = False,
    base_url: str = "http://localhost:11440"
):
    """Query the Ollama API with the given prompt."""
    url = f"{base_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return None

def get_wikipedia_on_this_day_facts(month: int, day: int) -> List[HistoricalFact]:
    """
    Fetch historical facts from Wikipedia's 'On This Day' API.
    Returns a list of HistoricalFact objects.
    """
    url = f"https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/all/{month:02d}/{day:02d}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        facts = []
        
        # Parse events from the response
        events = data.get("events", [])
        for event in events:
            if "year" in event and "text" in event:
                year = event["year"]
                
                # Filter out very old dates (before 1500) and future dates
                if year < 1500 or year > 2024:
                    continue
                
                try:
                    # Parse the date with proper error handling
                    event_date = datetime(year, month, day)
                    
                    fact = HistoricalFact(
                        event=event["text"],
                        date=f"{year}-{month:02d}-{day:02d}",
                        year=year,
                        month=month,
                        day=day,
                        category=event.get("pages", [{}])[0].get("type", "general") if event.get("pages") else "general"
                    )
                    facts.append(fact)
                    
                except ValueError as e:
                    # Skip invalid dates
                    print(f"Skipping invalid date: {year}-{month:02d}-{day:02d} for event: {event['text'][:50]}...")
                    continue
        
        print(f"Successfully fetched {len(facts)} events from Wikipedia API for {month:02d}/{day:02d}")
        return facts
        
    except Exception as e:
        print(f"Error fetching Wikipedia facts: {e}")
        return []

def get_sample_historical_facts() -> List[HistoricalFact]:
    """
    Return a curated list of historical facts for testing.
    This is a fallback when Wikipedia API is not available.
    """
    return [
        HistoricalFact(
            event="John F. Kennedy was assassinated in Dallas, Texas",
            date="1963-11-22",
            year=1963,
            month=11,
            day=22,
            category="politics"
        ),
        HistoricalFact(
            event="The first moon landing occurred with Apollo 11",
            date="1969-07-20",
            year=1969,
            month=7,
            day=20,
            category="space"
        ),
        HistoricalFact(
            event="World War II ended in Europe with Germany's surrender",
            date="1945-05-08",
            year=1945,
            month=5,
            day=8,
            category="war"
        ),
        HistoricalFact(
            event="The Titanic sank after hitting an iceberg",
            date="1912-04-15",
            year=1912,
            month=4,
            day=15,
            category="disaster"
        ),
        HistoricalFact(
            event="The Declaration of Independence was adopted by the Continental Congress",
            date="1776-07-04",
            year=1776,
            month=7,
            day=4,
            category="politics"
        ),
        HistoricalFact(
            event="The Berlin Wall fell, marking the end of the Cold War era",
            date="1989-11-09",
            year=1989,
            month=11,
            day=9,
            category="politics"
        ),
        HistoricalFact(
            event="The Wright brothers made their first powered flight",
            date="1903-12-17",
            year=1903,
            month=12,
            day=17,
            category="aviation"
        ),
        HistoricalFact(
            event="The atomic bomb was dropped on Hiroshima",
            date="1945-08-06",
            year=1945,
            month=8,
            day=6,
            category="war"
        ),
        HistoricalFact(
            event="The Great Depression began with the stock market crash",
            date="1929-10-29",
            year=1929,
            month=10,
            day=29,
            category="economics"
        ),
        HistoricalFact(
            event="The first iPhone was released by Apple",
            date="2007-06-29",
            year=2007,
            month=6,
            day=29,
            category="technology"
        )
    ]

def generate_question_from_fact(fact: HistoricalFact) -> str:
    """Generate a question from a historical fact."""
    event_lower = fact.event.lower()
    
    # Extract key entities for question generation
    if "assassinated" in event_lower:
        return "When was JFK assassinated?"
    elif "moon landing" in event_lower or "apollo" in event_lower:
        return "When did the first moon landing occur?"
    elif "world war ii" in event_lower or "germany's surrender" in event_lower:
        return "When did World War II end in Europe?"
    elif "titanic" in event_lower:
        return "When did the Titanic sink?"
    elif "declaration of independence" in event_lower:
        return "When was the Declaration of Independence adopted?"
    elif "berlin wall" in event_lower:
        return "When did the Berlin Wall fall?"
    elif "wright brothers" in event_lower or "first flight" in event_lower:
        return "When did the Wright brothers make their first powered flight?"
    elif "hiroshima" in event_lower or "atomic bomb" in event_lower:
        return "When was the atomic bomb dropped on Hiroshima?"
    elif "great depression" in event_lower or "stock market crash" in event_lower:
        return "When did the Great Depression begin?"
    elif "iphone" in event_lower:
        return "When was the first iPhone released?"
    else:
        # Generic question based on the event
        return f"When did this event occur: {fact.event}?"

def extract_date_from_response(response: str) -> Optional[str]:
    """
    Extract a date from the model's response.
    Returns the date in YYYY-MM-DD format if found, None otherwise.
    """
    if not response:
        return None
    
    # Common date patterns
    patterns = [
        # YYYY-MM-DD
        r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',
        # MM/DD/YYYY
        r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
        # Month DD, YYYY
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
        # DD Month YYYY
        r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        # Just year
        r'\b(\d{4})\b'
    ]
    
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            match = matches[0]
            
            if len(match) == 3:
                if match[0].isdigit() and match[1].isdigit() and match[2].isdigit():
                    # YYYY-MM-DD or MM/DD/YYYY format
                    if len(match[0]) == 4:  # YYYY-MM-DD
                        year, month, day = int(match[0]), int(match[1]), int(match[2])
                    else:  # MM/DD/YYYY
                        month, day, year = int(match[0]), int(match[1]), int(match[2])
                else:
                    # Month name format
                    if match[0].lower() in month_names:
                        month, day, year = month_names[match[0].lower()], int(match[1]), int(match[2])
                    else:
                        day, month_name, year = int(match[0]), month_names[match[1].lower()], int(match[2])
                        month = month_name
                
                try:
                    # Validate the date
                    datetime(year, month, day)
                    return f"{year:04d}-{month:02d}-{day:02d}"
                except ValueError:
                    continue
            
            elif len(match) == 1 and match[0].isdigit() and len(match[0]) == 4:
                # Just year
                return f"{match[0]}-01-01"  # Use January 1st as default
    
    return None

def calculate_accuracy_score(extracted_date: str, ground_truth_date: str) -> Tuple[bool, float]:
    """
    Calculate if the extracted date is correct and provide a confidence score.
    Returns (is_correct, confidence_score)
    
    Scoring hierarchy:
    - Year correct: 70% of total score
    - Month correct: 20% of total score  
    - Day correct: 10% of total score
    """
    if not extracted_date or not ground_truth_date:
        return False, 0.0
    
    try:
        extracted = datetime.strptime(extracted_date, "%Y-%m-%d")
        ground_truth = datetime.strptime(ground_truth_date, "%Y-%m-%d")
        
        # Check if years match (most important)
        year_correct = extracted.year == ground_truth.year
        
        # Check if full date matches
        full_date_correct = extracted_date == ground_truth_date
        
        # Calculate confidence score with proper weighting
        if full_date_correct:
            confidence = 1.0
        elif year_correct:
            # Year is correct (70% base score)
            year_score = 0.7
            
            # Month correctness (20% of remaining 30%)
            month_correct = extracted.month == ground_truth.month
            month_score = 0.2 if month_correct else 0.0
            
            # Day correctness (10% of remaining 30%)
            day_correct = extracted.day == ground_truth.day
            day_score = 0.1 if day_correct else 0.0
            
            confidence = year_score + month_score + day_score
        else:
            # Year is wrong - calculate penalty based on how far off
            year_diff = abs(extracted.year - ground_truth.year)
            
            # Heavily penalize wrong years
            if year_diff <= 1:
                confidence = 0.3  # Very close year
            elif year_diff <= 5:
                confidence = 0.1  # Close year
            elif year_diff <= 10:
                confidence = 0.05  # Moderately far year
            else:
                confidence = 0.01  # Very far year
        
        return year_correct, confidence
        
    except ValueError:
        return False, 0.0

def test_model_for_hallucinations(facts: List[HistoricalFact], num_tests: int = 5) -> List[TestResult]:
    """
    Test the model for hallucinations using historical facts.
    """
    results = []
    
    # Randomly sample facts
    selected_facts = random.sample(facts, min(num_tests, len(facts)))
    
    for fact in selected_facts:
        print(f"\n--- Testing Fact: {fact.event} ---")
        
        # Generate question
        question = generate_question_from_fact(fact)
        print(f"Question: {question}")
        
        # Query the model
        system_prompt = "You are a helpful assistant. Answer questions accurately and concisely. When asked about dates, provide the specific date in a clear format."
        response = query_ollama(question, system_prompt=system_prompt, temperature=0.3)
        
        if response is None:
            print("Failed to get response from model")
            continue
            
        print(f"Model Response: {response}")
        
        # Extract date from response
        extracted_date = extract_date_from_response(response)
        print(f"Extracted Date: {extracted_date}")
        print(f"Ground Truth: {fact.date}")
        
        # Check accuracy
        is_correct, confidence = calculate_accuracy_score(extracted_date, fact.date)
        
        result = TestResult(
            fact=fact,
            question=question,
            model_response=response,
            extracted_date=extracted_date,
            is_correct=is_correct,
            confidence_score=confidence
        )
        
        results.append(result)
        
        print(f"Correct: {is_correct}, Confidence: {confidence:.2f}")
    
    return results

def print_summary(results: List[TestResult]):
    """Print a summary of the test results."""
    if not results:
        print("No results to summarize")
        return
    
    total_tests = len(results)
    correct_answers = sum(1 for r in results if r.is_correct)
    avg_confidence = sum(r.confidence_score for r in results) / total_tests
    
    print("\n" + "="*50)
    print("HALLUCINATION TEST SUMMARY")
    print("="*50)
    print(f"Total Tests: {total_tests}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Accuracy: {correct_answers/total_tests*100:.1f}%")
    print(f"Average Confidence: {avg_confidence:.2f}")
    
    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✓" if result.is_correct else "✗"
        print(f"{i}. {status} {result.fact.event}")
        print(f"   Expected: {result.fact.date}, Got: {result.extracted_date or 'None'}")
        print(f"   Confidence: {result.confidence_score:.2f}")

def main():
    """Main function to run the hallucination test."""
    print("Starting Gemma Model Hallucination Test")
    print("="*50)
    
    # Get historical facts
    print("Loading historical facts...")
    
    # Try to get facts from Wikipedia API (you'll need to add your API key)
    facts = get_wikipedia_on_this_day_facts(6, 30)  # Example: November 22
    
    # For now, use sample facts
    #facts = get_sample_historical_facts()
    print(f"Loaded {len(facts)} historical facts")
    
    # Run the test
    print(f"\nRunning hallucination test with {min(5, len(facts))} facts...")
    results = test_model_for_hallucinations(facts, num_tests=5)
    
    # Print summary
    print_summary(results)
    
    # Save results to file
    with open("hallucination_test_results.json", "w") as f:
        json.dump([{
            "fact": {
                "event": r.fact.event,
                "date": r.fact.date,
                "category": r.fact.category
            },
            "question": r.question,
            "model_response": r.model_response,
            "extracted_date": r.extracted_date,
            "is_correct": r.is_correct,
            "confidence_score": r.confidence_score
        } for r in results], f, indent=2)
    
    print(f"\nResults saved to hallucination_test_results.json")

if __name__ == "__main__":
    main() 