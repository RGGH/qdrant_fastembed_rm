import requests
from bs4 import BeautifulSoup
import json
import time

# Base URL for the property listings
base_url = "https://www.rightmove.co.uk/property-for-sale/find.html"

# Parameters for the initial request
params = {
    'locationIdentifier': 'REGION%5E270',
    'maxBedrooms': 3,
    'maxPrice': 300000,
    'minPrice': 200000,
    'radius': 20.0,
    'sortType': 6,
    'propertyTypes': 'detached',
    'includeSSTC': 'false',
    'mustHave': '',
    'dontShow': 'sharedOwnership%2CnewHome%2Cretirement',
    'furnishTypes': '',
    'keywords': ''
}

# Headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# List of words to skip
skip_words = ["park home","semi - detached","semi detached","semi-detached", "shared ownership", "new home", "retirement","Auction", "Lodge", "terraced","terrace", "Semi - Detached"]

# File to store results
output_file = 'data.jsonl'

def get_page_url(index):
    """Generate the URL for a specific page index."""
    params['index'] = index
    return base_url + '?' + '&'.join([f'{key}={value}' for key, value in params.items()])

def scrape_page(url):
    """Scrape a single page and return the listings."""
    response = requests.get(url, headers=headers)
    listings = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        listings = soup.find_all('div', class_='l-searchResult is-list')
    else:
        print(f'Failed to retrieve the webpage. Status code: {response.status_code}')

    return listings

def extract_property_data(listing):
    """Extract property data from a BeautifulSoup listing element."""
    # Extract property title
    title_tag = listing.find('h2', class_='propertyCard-title')
    title = title_tag.get_text(strip=True) if title_tag else 'No title'

    # Extract property description using more specific XPath-like structure
    description_tag = listing.select_one('div.propertyCard-description')
    if not description_tag:
        description_tag = listing.select_one('div.propertyCard-content span span')
    description = description_tag.get_text(strip=True) if description_tag else 'No description'

    # Check if the title or description contains any words to skip
    if any(skip_word.lower() in title.lower() for skip_word in skip_words) or any(skip_word.lower() in description.lower() for skip_word in skip_words):
        return None

    # Extract property price
    price_tag = listing.find('div', class_='propertyCard-priceValue')
    price_text = price_tag.get_text(strip=True) if price_tag else 'No price'
    try:
        price = int(price_text.replace('£', '').replace(',', '').strip())
    except ValueError:
        price = 0

    # Extract property address
    address_tag = listing.find('address', class_='propertyCard-address')
    address = address_tag.get_text(strip=True) if address_tag else 'No address'

    # Extract property link
    link_tag = listing.find('a', class_='propertyCard-link')
    link = link_tag['href'] if link_tag else ''
    full_link = f"https://www.rightmove.co.uk{link}" if link else 'No link'

    return {
        'title': title,
        'price': price,
        'price_text': price_text,
        'address': address,
        'description': description,
        'link': full_link
    }

def main():
    # Initial page index
    index = 0
    all_listings = []

    while True:
        url = get_page_url(index)
        print(f'Scraping page {index + 1}: {url}')
        listings = scrape_page(url)
        
        if not listings:
            break

        for listing in listings:
            property_data = extract_property_data(listing)
            if property_data:
                all_listings.append(property_data)

        index += 24  # Move to the next page (24 results per page)
        time.sleep(1)  # Be polite and wait a bit between requests

    # Write all listings to the JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        for property_data in all_listings:
            file.write(json.dumps(property_data, ensure_ascii=False) + '\n')

    print(f"Properties have been written to {output_file}")

if __name__ == '__main__':
    main()

