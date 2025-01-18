import json
from urllib.parse import urlparse, parse_qs
import logging
from typing import Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def normalize_url(url: str) -> str:
    """Normalize URLs to identify duplicates with different query parameters."""
    parsed = urlparse(url)

    # Remove query parameters but keep the essential parts of path
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    # Remove trailing slashes
    base = base.rstrip('/')

    # Convert to lowercase for case-insensitive comparison
    return base.lower()

def remove_duplicates_from_json(input_file: str = 'learncbse_links.json',
                              output_file: str = 'learncbse_links_deduped.json'):
    """Remove duplicate URLs from the JSON structure."""

    try:
        # Read the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found!")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {input_file}!")
        return

    # Keep track of all unique URLs
    seen_urls: Set[str] = set()
    initial_count = 0
    final_count = 0

    # Process each class
    for class_name, subjects in data['classes'].items():
        # Process each subject
        for subject_name, subject_data in subjects.items():
            # Count initial URLs
            initial_count += len(subject_data['chapter_links'])

            # Create a new list of unique chapter links
            unique_chapter_links = []
            temp_seen_urls = set()  # Temporary set for this subject

            # Process each chapter link
            for url in subject_data['chapter_links']:
                normalized_url = normalize_url(url)
                if normalized_url not in seen_urls and normalized_url not in temp_seen_urls:
                    unique_chapter_links.append(url)
                    temp_seen_urls.add(normalized_url)

            # Update the global seen URLs set
            seen_urls.update(temp_seen_urls)

            # Update the data structure with unique links
            subject_data['chapter_links'] = unique_chapter_links
            final_count += len(unique_chapter_links)

    # Save the deduplicated data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Print statistics
    logging.info(f"Initial number of URLs: {initial_count}")
    logging.info(f"Final number of URLs after deduplication: {final_count}")
    logging.info(f"Removed {initial_count - final_count} duplicate URLs")
    logging.info(f"Deduplicated data saved to {output_file}")

def analyze_duplicates(input_file: str = 'learncbse_links.json'):
    """Analyze and print information about duplicate URLs."""

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found!")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {input_file}!")
        return

    # Dictionary to store URL occurrences
    url_count: Dict[str, List[str]] = {}

    # Collect all URLs and their locations
    for class_name, subjects in data['classes'].items():
        for subject_name, subject_data in subjects.items():
            for url in subject_data['chapter_links']:
                normalized_url = normalize_url(url)
                if normalized_url not in url_count:
                    url_count[normalized_url] = []
                url_count[normalized_url].append(f"{class_name} - {subject_name}")

    # Print duplicate analysis
    duplicates_found = False
    for url, locations in url_count.items():
        if len(locations) > 1:
            duplicates_found = True
            logging.info(f"\nDuplicate URL: {url}")
            logging.info("Found in:")
            for location in locations:
                logging.info(f"  - {location}")

    if not duplicates_found:
        logging.info("No duplicates found!")

if __name__ == "__main__":
    # First analyze the duplicates
    logging.info("Analyzing duplicates...")
    analyze_duplicates()

    # Then remove them
    logging.info("\nRemoving duplicates...")
    remove_duplicates_from_json()
