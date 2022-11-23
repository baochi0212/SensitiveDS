from googlesearch import search

query = '"Anatomy articles"'
num_results = 3
lang = 'en'

# results = search(term=query, num_results=num_results, lang=lang)
from googlesearch import search
for url in search(term=query, num_results=num_results, lang=lang):
    print(url)