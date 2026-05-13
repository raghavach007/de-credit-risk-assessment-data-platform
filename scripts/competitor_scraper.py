
import requests
from bs4 import BeautifulSoup
import time

urls = [
    "https://www.hdfcbank.com",
    "https://www.icicibank.com",
    "https://www.axisbank.com"
]

headers = {
    "User-Agent": "Mozilla/5.0"
}

for url in urls:
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.text.strip() if soup.title else "No title"

        print(f"Website: {url}")
        print(f"Title: {title}")
        print("-" * 40)

        time.sleep(2)

    except Exception as e:
        print(f"Failed for {url}: {e}")
