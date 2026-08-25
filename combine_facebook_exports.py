import csv
import json
import hashlib
from pathlib import Path

src = Path(r"C:\Users\sumit\Downloads\facebook_flats")
out = Path(r"D:\zeroToOneHackathon\hackeyspecter\outputs\facebook_flats_combined.csv")
json_out = out.with_suffix(".json")
tsv_out = out.with_suffix(".tsv")
xml_out = out.with_suffix(".xml")

def rows_from_json(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("posts", "data", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return []

records = []
all_keys = set()
for path in sorted(src.iterdir()):
    if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
        continue
    try:
        if path.suffix.lower() == ".json":
            rows = rows_from_json(path)
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean = {str(k): ("" if v is None else str(v)) for k, v in row.items() if k is not None}
            clean["source_file"] = path.name
            all_keys.update(clean)
            records.append(clean)
    except Exception as exc:
        print(f"Skipped {path.name}: {exc}")

def identity(row):
    for key in ("postUrl", "post_url", "postLink", "post_link", "url", "link"):
        if row.get(key, "").strip():
            return key + ":" + row[key].strip()
    for key in ("text", "content", "postText", "description", "message"):
        if row.get(key, "").strip():
            payload = "|".join(row.get(k, "").strip() for k in (key, "timestamp", "date", "author", "groupName"))
            return "fallback:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return "row:" + hashlib.sha1(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()

unique = {}
for row in records:
    unique.setdefault(identity(row), row)

preferred = ["author", "name", "timestamp", "date", "postUrl", "post_url", "postLink", "post_link", "text", "content", "description", "groupName", "group", "source_file"]
columns = [k for k in preferred if k in all_keys]
columns += sorted(all_keys - set(columns))
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(unique.values())

json_out.write_text(json.dumps(list(unique.values()), ensure_ascii=False, indent=2), encoding="utf-8")
with tsv_out.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", delimiter="\t")
    writer.writeheader()
    writer.writerows(unique.values())

def xml_escape(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;"))

rows_xml = []
rows_xml.append("<?xml version=\"1.0\"?>")
rows_xml.append("<?mso-application progid=\"Excel.Sheet\"?>")
rows_xml.append("<Workbook xmlns=\"urn:schemas-microsoft-com:office:spreadsheet\" xmlns:ss=\"urn:schemas-microsoft-com:office:spreadsheet\">")
rows_xml.append("<Worksheet ss:Name=\"Facebook Flats\"><Table>")
rows_xml.append("<Row>" + "".join(f"<Cell><Data ss:Type=\"String\">{xml_escape(c)}</Data></Cell>" for c in columns) + "</Row>")
for row in unique.values():
    rows_xml.append("<Row>" + "".join(f"<Cell><Data ss:Type=\"String\">{xml_escape(row.get(c, ''))}</Data></Cell>" for c in columns) + "</Row>")
rows_xml.append("</Table></Worksheet></Workbook>")
xml_out.write_text("\n".join(rows_xml), encoding="utf-8")
print(f"Input records: {len(records)}")
print(f"Unique records: {len(unique)}")
print(f"Outputs: {out}, {json_out}, {tsv_out}, {xml_out}")
