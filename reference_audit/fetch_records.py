"""Public, unauthenticated bibliography audit; never reads local credentials."""
import concurrent.futures as cf
import datetime, json, re, subprocess, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from urllib.parse import quote, urlencode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reference_audit'
RAW = OUT / 'api_records'
RAW.mkdir(parents=True, exist_ok=True)
text = (ROOT / 'peerj_submission/main_manuscript.md').read_text(encoding='utf-8')
refs = text.split('## References\n', 1)[1].split('## Additional information', 1)[0]
entries = [x.strip() for x in refs.split('\n\n') if x.strip()]
catalogue = []
for i, entry in enumerate(entries, 1):
    match = re.search(r'DOI: (\S+)', entry)
    doi = match.group(1).rstrip('.') if match else None
    catalogue.append({'index': i, 'entry': entry, 'doi': doi})
snapshot = OUT / 'bibliography.json'
if snapshot.exists():
    catalogue = json.loads(snapshot.read_text(encoding='utf-8'))
else:
    snapshot.write_text(json.dumps(catalogue, ensure_ascii=False, indent=2), encoding='utf-8')

def fetch(url, target):
    if target.exists():
        return json.loads(target.read_text(encoding='utf-8'))
    proc = subprocess.run(['curl.exe', '-sS', '--max-time', '45', '--retry', '1', url], capture_output=True, encoding='utf-8')
    try:
        data = json.loads(proc.stdout)
    except Exception:
        data = {'error': proc.stderr or proc.stdout[:300], 'returncode': proc.returncode}
    record = {'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'url': url, 'response': data}
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    time.sleep(0.4)
    return record

def provider(name):
    for item in catalogue:
        doi = item['doi']
        if not doi: continue
        if name == 'crossref': url = 'https://api.crossref.org/works/' + quote(doi, safe='')
        elif name == 'europepmc': url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urlencode({'query': 'DOI:"' + doi + '"', 'format': 'json', 'resultType': 'core'})
        else: url = 'https://api.openalex.org/works/https://doi.org/' + quote(doi, safe='')
        rec = fetch(url, RAW / f"{item['index']:02d}_{name}.json")
        d = rec['response']
        if name == 'crossref':
            m = d.get('message', {})
            print(item['index'], name, m.get('title'), m.get('page'), m.get('article-number'), flush=True)
        elif name == 'openalex': print(item['index'], name, d.get('id', d.get('error')), flush=True)
    return name

if __name__ == '__main__':
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        print(list(pool.map(provider, ['crossref', 'europepmc', 'openalex'])))
