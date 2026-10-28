from fetch_records import fetch, OUT, RAW
from urllib.parse import urlencode, quote
import json, subprocess
queries={2:'Deep reinforcement learning at the edge of the statistical precipice',8:'Statistical Power Analysis for the Behavioral Sciences Cohen',15:'A simple sequentially rejective multiple test procedure Holm',25:'Improving reproducibility in machine learning research Pineau'}
for i,title in queries.items():
 fetch('https://api.openalex.org/works?'+urlencode({'search':title,'per-page':3}),RAW/f'{i:02d}_openalex_search.json')
 fetch('https://api.crossref.org/works?'+urlencode({'query.bibliographic':title,'rows':3}),RAW/f'{i:02d}_crossref_search.json')
doi='10.1080/00401706.1979.10489755'
fetch('https://api.crossref.org/works/'+quote(doi,safe=''),RAW/'46_crossref.json')
fetch('https://api.openalex.org/works/https://doi.org/'+quote(doi,safe=''),RAW/'46_openalex.json')
urls={
 'kim_bioc.xml':'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC5079817/unicode',
 'peerj_author_instructions.html':'https://peerj.com/journals/computer-science/author-instructions',
 'scipy_citing.html':'https://scipy.org/citing-scipy/',
 'agarwal_proceedings.html':'https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html',
 'pineau_jmlr.html':'https://jmlr.org/papers/v22/20-303.html',
 'henderson_aaai.html':'https://ojs.aaai.org/index.php/AAAI/article/view/11694',
 'bootstrap_publisher.html':'https://www.taylorfrancis.com/books/mono/10.1201/9780429246593/introduction-bootstrap-bradley-efron-tibshirani'}
for name,url in urls.items():
 proc=subprocess.run(['curl.exe','--compressed','-L','-sS','--max-time','45',url],capture_output=True,encoding='utf8',errors='replace')
 (OUT/name).write_text(proc.stdout,encoding='utf8')
print('Remaining reference searches and primary pages saved.')
