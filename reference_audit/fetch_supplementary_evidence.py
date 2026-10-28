import json, re, subprocess, time
from pathlib import Path
from urllib.parse import urlencode, quote
from fetch_records import fetch, RAW, OUT

additional = ['10.1152/jappl.1948.1.2.93','10.1002/andp.19123420503','10.1364/AO.37.005271','10.1016/j.jphotobiol.2016.09.031']
for i, doi in enumerate(additional, 42):
    for provider, url in [('crossref', 'https://api.crossref.org/works/'+quote(doi,safe='')),('openalex','https://api.openalex.org/works/https://doi.org/'+quote(doi,safe='')),('europepmc','https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urlencode({'query':'DOI:"'+doi+'"','format':'json','resultType':'core'}))]:
        fetch(url, RAW/f'{i:02d}_{provider}.json')

# Direct NCBI E-utilities API verification, no API key or local secret.
ids=[]
for f in sorted(RAW.glob('*_europepmc.json')):
    for r in json.loads(f.read_text(encoding='utf8'))['response'].get('resultList',{}).get('result',[]):
        if r.get('source')=='MED': ids.append(r['id'])
url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'+urlencode({'db':'pubmed','id':','.join(ids),'retmode':'xml','tool':'manuscript_reference_audit'})
p=subprocess.run(['curl.exe','-sS','--max-time','60',url],capture_output=True,encoding='utf8')
(OUT/'ncbi_pubmed_records.xml').write_text(p.stdout,encoding='utf8')
(OUT/'ncbi_request.json').write_text(json.dumps({'url':url,'record_ids':ids},indent=2),encoding='utf8')
for f in sorted(RAW.glob('*_europepmc.json')):
    records=json.loads(f.read_text(encoding='utf8'))['response'].get('resultList',{}).get('result',[])
    if records and records[0].get('pmcid'):
        pmcid=records[0]['pmcid']
        target=OUT/(pmcid+'.xml')
        if not target.exists():
            url='https://www.ebi.ac.uk/europepmc/webservices/rest/'+pmcid+'/fullTextXML'
            p=subprocess.run(['curl.exe','-sS','--max-time','40',url],capture_output=True,encoding='utf8')
            target.write_text(p.stdout,encoding='utf8')
        time.sleep(.35)
print('Additional sources, NCBI verification and available full texts saved.')
