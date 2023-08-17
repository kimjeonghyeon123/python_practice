import random

words = {'rear': '뒷부분', 'haven': '안식처, 피난처', 'look out over': '~을 내다보다, 바라보다',
        'hang out': '시간을 보내다', 'spacious': '널찍한', 'suburb(서벌브)': '교외', 'urban': '도시의',
        'stuffy': '(환기가 안 되어) 답답한', 'on the plus side': '좋은 점으로는', 'flood': '물이 넘치다, 침수',
        'root': '근원, 뿌리', 'plumber(플러멀)': '배관공 (*b 묵음)', 'thoroughly(쏘롤리)': '완전히, 대단히'
}
k = list(words.keys())
v = list(words.values())
length = len(words)
li = [i for i in range(length)]
random.shuffle(li)

for i in li:
    print(v[i], ' :', end=' ')
    answer = input()
    if answer == k[i].split('(')[0]:
        continue
    else:
        print('틀렸습니다. 답은', k[i], '입니다.')
        break