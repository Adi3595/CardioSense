import httpx
import re
import json

url = "https://data.mendeley.com/datasets/dzz48mvjht/1"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
r = httpx.get(url, headers=headers, follow_redirects=True)
print("Final URL:", r.url)
print("Status Code:", r.status_code)

title_match = re.search(r"<title>(.*?)</title>", r.text, re.IGNORECASE | re.DOTALL)
print("Title:", title_match.group(1).strip() if title_match else "No title")

# Look for embedded JSON-LD or initial data
json_lds = re.findall(r'<script type="application/ld\+json">(.*?)</script>', r.text, re.DOTALL)
print(f"Found {len(json_lds)} JSON-LD blocks")
for i, jld in enumerate(json_lds):
    try:
        data = json.loads(jld)
        print(f"JSON-LD {i} type:", data.get("@type"), data.get("name"))
        if "distribution" in data:
            print("Distributions:", json.dumps(data["distribution"], indent=2))
        if "description" in data:
            print("Description snippet:", data["description"][:300])
    except Exception as e:
        print(f"Error parsing JSON-LD {i}: {e}")

# Search for download URLs
download_urls = re.findall(r'https?://[^\s"\'<>]+(?:download|files|csv|xlsx|zip)[^\s"\'<>]*', r.text, re.IGNORECASE)
print(f"Found {len(download_urls)} candidate download URLs:")
for u in download_urls[:10]:
    print(" -", u)
