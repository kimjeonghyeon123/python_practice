import requests
from bs4 import BeautifulSoup

url = "https://comic.naver.com/index"
res = requests.get(url)
res.raise_for_status()
print(res.text)
# html 문서를 lxml 파서를 통해 뷰티풀숩 객체로 만듦
soup = BeautifulSoup(res.text, "lxml")
#print(soup.title)
#print(soup.title.get_text())
