import requests
import csv
import time
from datetime import datetime, date
from typing import List, Dict, Optional
import calendar
from dataclasses import dataclass, asdict
from tqdm import tqdm

@dataclass
class HistoricalEvent:
    """Data class to represent a historical event"""
    date: str  # YYYY-MM-DD format
    year: int
    month: int
    day: int
    event_text: str
    category: str = "general"
    source_pages: List[str] = None
    
    def __post_init__(self):
        if self.source_pages is None:
            self.source_pages = []

class WikimediaEventScraper:
    """Scraper for Wikimedia 'On This Day' API"""
    
    def __init__(self, delay_between_requests: float = 0.5):
        self.base_url = "https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/all"
        self.delay = delay_between_requests
        self.session = requests.Session()
        # Set a user agent to be polite to the API
        self.session.headers.update({
            'User-Agent': 'Historical Events Dataset Creator/1.0 (Educational Purpose)'
        })
    
    def get_events_for_date(self, month: int, day: int) -> List[HistoricalEvent]:
        """
        Fetch historical events for a specific date.
        
        Args:
            month: Month (1-12)
            day: Day (1-31)
            
        Returns:
            List of HistoricalEvent objects
        """
        url = f"{self.base_url}/{month:02d}/{day:02d}"
        
        try:
            # tqdm will handle progress, so reduce print verbosity
            # print(f"Fetching events for {month:02d}/{day:02d}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            events = []
            
            # Parse events from the response
            api_events = data.get("events", [])
            for event in api_events:
                if "year" in event and "text" in event:
                    year = event["year"]
                    
                    # Filter out very old dates (before 1000) and future dates
                    current_year = datetime.now().year
                    if year < 1000 or year > current_year:
                        continue
                    
                    try:
                        # Parse the date with proper error handling
                        event_date = datetime(year, month, day)
                        
                        # Extract source pages if available
                        source_pages = []
                        if event.get("pages"):
                            source_pages = [page.get("title", "") for page in event["pages"]]
                        
                        # Get category from first page if available
                        category = "general"
                        if event.get("pages") and len(event["pages"]) > 0:
                            category = event["pages"][0].get("type", "general")
                        
                        historical_event = HistoricalEvent(
                            date=f"{year}-{month:02d}-{day:02d}",
                            year=year,
                            month=month,
                            day=day,
                            event_text=event["text"],
                            category=category,
                            source_pages=source_pages
                        )
                        events.append(historical_event)
                        
                    except ValueError as e:
                        # Skip invalid dates
                        # print(f"Skipping invalid date: {year}-{month:02d}-{day:02d}")
                        continue
            
            # print(f"Successfully fetched {len(events)} events for {month:02d}/{day:02d}")
            return events
            
        except requests.exceptions.RequestException as e:
            # print(f"Error fetching data for {month:02d}/{day:02d}: {e}")
            return []
        except Exception as e:
            # print(f"Unexpected error for {month:02d}/{day:02d}: {e}")
            return []
    
    def get_all_events_for_year(self, target_year: Optional[int] = None) -> List[HistoricalEvent]:
        """
        Fetch historical events for all dates in a year.
        
        Args:
            target_year: If specified, only return events from this year. 
                        If None, return events from all years.
            
        Returns:
            List of all HistoricalEvent objects
        """
        all_events = []
        total_dates = 0
        successful_dates = 0

        # Prepare all (month, day) pairs for the year
        current_year = datetime.now().year
        date_pairs = []
        for month in range(1, 13):
            days_in_month = calendar.monthrange(current_year, month)[1]
            for day in range(1, days_in_month + 1):
                date_pairs.append((month, day))
        total_dates = len(date_pairs)

        # Use tqdm for progress bar
        for idx, (month, day) in enumerate(tqdm(date_pairs, desc="Scraping dates", unit="date")):
            # Add delay to be respectful to the API
            if idx > 0:
                time.sleep(self.delay)
            
            events = self.get_events_for_date(month, day)
            
            if events:
                successful_dates += 1
                
                # Filter by target year if specified
                if target_year:
                    events = [event for event in events if event.year == target_year]
                
                all_events.extend(events)
        
        print(f"\nScraping completed!")
        print(f"Total dates processed: {total_dates}")
        print(f"Successful API calls: {successful_dates}")
        print(f"Total events collected: {len(all_events)}")
        
        return all_events
    
    def save_events_to_csv(self, events: List[HistoricalEvent], filename: str = "historical_events.csv"):
        """
        Save events to a CSV file.
        
        Args:
            events: List of HistoricalEvent objects
            filename: Output CSV filename
        """
        if not events:
            print("No events to save!")
            return
        
        # Prepare data for CSV
        csv_data = []
        for event in events:
            row = {
                'date': event.date,
                'year': event.year,
                'month': event.month,
                'day': event.day,
                'event_text': event.event_text,
                'category': event.category,
                'source_pages': '; '.join(event.source_pages) if event.source_pages else ''
            }
            csv_data.append(row)
        
        # Sort by date
        csv_data.sort(key=lambda x: x['date'])
        
        # Write to CSV
        fieldnames = ['date', 'year', 'month', 'day', 'event_text', 'category', 'source_pages']
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"\nEvents successfully saved to {filename}")
            print(f"Total events saved: {len(csv_data)}")
            
        except Exception as e:
            print(f"Error saving to CSV: {e}")

def main():
    """Main function to run the scraper"""
    print("Wikimedia Historical Events Dataset Creator")
    print("=" * 50)
    
    # Initialize scraper
    scraper = WikimediaEventScraper(delay_between_requests=0.5)
    
    # Option to scrape specific year or all years
    target_year = None
    scrape_specific_year = input("Do you want to scrape events for a specific year? (y/n): ").lower().strip()
    
    if scrape_specific_year == 'y':
        try:
            target_year = int(input("Enter the year (e.g., 2020): "))
            print(f"Scraping events for year {target_year}")
        except ValueError:
            print("Invalid year entered. Scraping all years.")
            target_year = None
    
    # Custom filename option
    custom_filename = input("Enter output filename (press Enter for 'historical_events.csv'): ").strip()
    filename = custom_filename if custom_filename else "historical_events.csv"
    
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    print(f"\nStarting data collection...")
    print(f"Output file: {filename}")
    print(f"Delay between requests: {scraper.delay} seconds")
    print("This may take several minutes to complete...\n")
    
    # Scrape all events
    all_events = scraper.get_all_events_for_year(target_year)
    
    if all_events:
        # Save to CSV
        scraper.save_events_to_csv(all_events, filename)
        
        # Print some statistics
        print(f"\nDataset Statistics:")
        print(f"Total events: {len(all_events)}")
        
        # Count events by category
        categories = {}
        for event in all_events:
            categories[event.category] = categories.get(event.category, 0) + 1
        
        print(f"Events by category:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}")
        
        # Year range
        years = [event.year for event in all_events]
        if years:
            print(f"Year range: {min(years)} - {max(years)}")
    
    else:
        print("No events were collected. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
