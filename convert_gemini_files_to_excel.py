import csv
import html
import io
import json
from pathlib import Path

base = Path(r"C:\Users\sumit\Downloads")
out = Path(r"D:\zeroToOneHackathon\hackeyspecter\outputs\gemini_flat_listings_excel.xml")
files = [base / "gemini-code-1786141585669.txt", base / "gemini-code-1786140774655.txt"]

def read_csv(path):
    text = path.read_text(encoding="utf-8-sig").replace("â‚¹", "₹")
    return list(csv.DictReader(io.StringIO(text)))

all_rows = read_csv(files[0])
detailed = read_csv(files[1])
detail_by_link = {r.get("Post Link", ""): r for r in detailed}
merged = []
seen = set()
for r in all_rows:
    link = r.get("Link", "")
    if link in seen:
        continue
    seen.add(link)
    d = detail_by_link.get(link, {})
    merged.append({
        "Poster Name": r.get("Poster Name", "").replace(" · Follow", ""),
        "Google Map Location": d.get("Google Map Location", "Not available"),
        "Private Bathroom": d.get("Private Bathroom", r.get("Private Bath", "Not stated")),
        "Rent Including Maintenance (INR)": r.get("Rent (₹)", "").replace(",", ""),
        "Gender Preference": r.get("Gender Pref", ""),
        "Additional Requirements": d.get("Additional Requirements", "Not specified"),
        "Contact Number": d.get("Contact Number", "N/A"),
        "Post Link": link,
    })

def esc(v):
    return html.escape(str(v), quote=True)

def sheet(name, rows):
    cols = list(rows[0]) if rows else []
    s = [f'<Worksheet ss:Name="{esc(name)}"><Table>']
    s.append("<Row>" + "".join(f'<Cell><Data ss:Type="String">{esc(c)}</Data></Cell>' for c in cols) + "</Row>")
    for row in rows:
        s.append("<Row>" + "".join(f'<Cell><Data ss:Type="String">{esc(row.get(c, ""))}</Data></Cell>' for c in cols) + "</Row>")
    s.append("</Table></Worksheet>")
    return "\n".join(s)

xml = ['<?xml version="1.0"?>', '<?mso-application progid="Excel.Sheet"?>', '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">']
xml.append(sheet("Combined Listings", merged))
xml.append(sheet("Detailed Source", detailed))
xml.append("</Workbook>")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(xml), encoding="utf-8")
(out.with_suffix(".json")).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Created {len(merged)} unique listings")
print(out)
