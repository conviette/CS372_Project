from urllib import request
from bs4 import BeautifulSoup as bs
import json


def scrape_list(base_url):
    targets = []
    base_url += '?page='
    i = 0
    while True:
        url = base_url+str(i)
        response = request.urlopen(url).read().decode('utf8')
        parsed = bs(response, 'lxml').findAll('div', class_="field-title")
        if len(parsed) == 0:
            break
        for p in parsed:
            targets.append(p.find('a')['href'])
        i += 1
    return targets

def scrape_body(url):
    response = request.urlopen('https://www.presidency.ucsb.edu{}'.format(url)).read().decode('utf8')
    parsed = bs(response, 'lxml')
    title = parsed.find('div', class_='field-ds-doc-title').text
    date = parsed.find('span', class_='date-display-single').text
    body = parsed.find('div', class_='field-docs-content').text
    return {'title':title, 'date':date, 'body':body}




TO_SCRAPE = {'sotus_a': 'https://www.presidency.ucsb.edu/documents/app-categories/spoken-addresses-and-remarks/presidential/state-the-union-addresses',
            'oral_a': 'https://www.presidency.ucsb.edu/documents/app-categories/spoken-addresses-and-remarks/presidential/oral-address'}

def main():

    for k, v in TO_SCRAPE.items():
        url_list = scrape_list(v)
        articles = list(map(scrape_body, url_list))
        print(len(articles))

        with open('{}.json'.format(k), 'w') as f:
            json.dump(articles, f)





if __name__ == '__main__':
    main()
