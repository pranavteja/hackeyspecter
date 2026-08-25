import csv, io, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

base = Path(r"C:\Users\sumit\Downloads")
out = Path(r"D:\zeroToOneHackathon\hackeyspecter\outputs\gemini_flat_listings.xlsx")

def read(path):
    txt = path.read_text(encoding="utf-8-sig").replace("â‚¹", "₹")
    return list(csv.DictReader(io.StringIO(txt)))

all_rows = read(base / "gemini-code-1786141585669.txt")
detail = read(base / "gemini-code-1786140774655.txt")
by_link = {r.get("Post Link", ""): r for r in detail}
rows, seen = [], set()
for r in all_rows:
    link = r.get("Link", "")
    if link in seen: continue
    seen.add(link); d = by_link.get(link, {})
    rows.append([r.get("Poster Name", "").replace(" · Follow", ""), d.get("Google Map Location", "Not available"), d.get("Private Bathroom", r.get("Private Bath", "Not stated")), r.get("Rent (₹)", ""), r.get("Gender Pref", ""), d.get("Additional Requirements", "Not specified"), d.get("Contact Number", "N/A"), link])
headers = ["Poster Name","Google Map Location","Private Bathroom","Rent Including Maintenance (INR)","Gender Preference","Additional Requirements","Contact Number","Post Link"]

def cell(v): return '<c t="inlineStr"><is><t xml:space="preserve">' + escape(str(v)) + '</t></is></c>'
def sheet(data):
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
    for i, row in enumerate(data, 1): xml.append(f'<row r="{i}">' + ''.join(cell(v) for v in row) + '</row>')
    xml.append('</sheetData></worksheet>'); return ''.join(xml)

ct = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>'
rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
wb = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Combined Listings" sheetId="1" r:id="rId1"/><sheet name="Detailed Source" sheetId="2" r:id="rId2"/></sheets></workbook>'
wbr = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>'
detail_data = [list(detail[0].keys())] + [list(r.values()) for r in detail]
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml', ct); z.writestr('_rels/.rels', rels); z.writestr('xl/workbook.xml', wb); z.writestr('xl/_rels/workbook.xml.rels', wbr); z.writestr('xl/worksheets/sheet1.xml', sheet([headers] + rows)); z.writestr('xl/worksheets/sheet2.xml', sheet(detail_data))
print(out)
