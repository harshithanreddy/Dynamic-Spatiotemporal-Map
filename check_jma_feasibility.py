"""
Inspect hypo.html and daily/monthly bulletin links for 2024 and 2025.
"""
import sys
import urllib.request
import re

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def fetch_url(url):
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return None, str(e)

# 1. Hypocenter catalog archive page
url_hypo = "https://www.data.jma.go.jp/svd/eqev/data/bulletin/hypo.html"
status, html = fetch_url(url_hypo)
print(f"=== {url_hypo} ({status}) ===")
links = re.findall(r'href=[\"\'](.*?)[\"\']', html)
data_links = [l for l in links if "data/hypo" in l or ".zip" in l or "h20" in l]
print("All catalog archive zip links:")
for l in data_links:
    print("  ", l)

# 2. Daily / Recent earthquake catalog page
url_daily = "https://www.data.jma.go.jp/svd/eqev/data/daily/index.html"
status, html_daily = fetch_url(url_daily)
print(f"\n=== {url_daily} ({status}) ===")
links_daily = re.findall(r'href=[\"\'](.*?)[\"\']', html_daily)
print("Daily page links sample:", links_daily[:15])

# 3. Interactive Shindo Search Database Endpoint
url_eqdb = "https://www.data.jma.go.jp/svd/eqdb/data/shindo/index.php"
status, html_eqdb = fetch_url(url_eqdb)
print(f"\n=== {url_eqdb} ({status}) ===")
forms = re.findall(r'<form.*?>.*?</form>', html_eqdb, re.DOTALL)
print(f"Found {len(forms)} forms on eqdb page.")
