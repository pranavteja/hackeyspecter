import csv
import re
from pathlib import Path

src = Path(r"D:\zeroToOneHackathon\hackeyspecter\outputs\facebook_flats_combined.csv")
outdir = Path(r"D:\zeroToOneHackathon\hackeyspecter\outputs\facebook_flats_filtered")
outdir.mkdir(parents=True, exist_ok=True)
headers = ["poster_name", "google_map_location", "private_bathroom", "post_date", "post_link", "room_rent_including_maintenance_inr", "contact_number", "additional_requirements", "bhk", "source_file"]
rows = list(csv.DictReader(src.open(encoding="utf-8-sig", newline="")))
seen, result = set(), []
for r in rows:
    text = r.get("text", "")
    url = r.get("postUrl", "") or r.get("Post URL", "")
    key = url or (r.get("author", "") + "|" + text[:200])
    if key in seen:
        continue
    seen.add(key)
    if not re.search(r"\b[234]\s*BHK\b", text, re.I):
        continue
    timestamp = r.get("timestamp", "")
    if not re.search(r"\b(?:July|August)\s+2026\b", timestamp, re.I):
        continue
    m = re.search(r"Rent\s*:\s*[₹Rs. ]*([0-9][0-9,]*)", text, re.I)
    if not m:
        continue
    rent = int(m.group(1).replace(",", ""))
    if rent >= 20000:
        continue
    loc = "Not stated"
    lm = re.search(r"(?:Location|Society|Address)\s*:\s*([^\n]+)", text, re.I)
    if lm:
        loc = lm.group(1).strip()
    bathroom = "Private/attached" if re.search(r"attached bathroom|private bathroom", text, re.I) else ("Shared" if re.search(r"shared.*(?:washroom|bathroom)|(?:washroom|bathroom).*shared", text, re.I) else "Not stated")
    contacts = sorted(set(re.findall(r"(?:\+91[- ]?)?[6-9]\d{9}", text)))
    result.append({"poster_name": r.get("author", ""), "google_map_location": loc, "private_bathroom": bathroom, "post_date": timestamp, "post_link": url, "room_rent_including_maintenance_inr": rent, "contact_number": ", ".join(contacts) or "Not stated", "additional_requirements": text[:1500], "bhk": re.search(r"\b([234])\s*BHK\b", text, re.I).group(1) + "BHK", "source_file": r.get("source_file", "")})

csv_out = outdir / "Filtered_Flat_Listings.csv"
with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=headers)
    w.writeheader()
    w.writerows(result)
(outdir / "Filtered_Flat_Listings.json").write_text(__import__("json").dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
(outdir / "README.txt").write_text("Filters: 2/3/4 BHK; post within 15 days of 2026-08-08; rent below INR 20,000. Distance filtering was not applied because it was removed from the latest request.\n", encoding="utf-8")
print(f"Unique qualifying listings: {len(result)}")
