import re

def createWordBag(articles):
    wordBag = set()
    for article in articles:
        words = re.split(r'\W+', article)
        words = [s for s in words if s != '']
        wordBag.update(words)
    return list(wordBag)

def createDataSet(articles, wordBag):
    dataSet = [[0 for j in range(len(wordBag))] for i in range(len(articles))]
    for i, article in enumerate(articles):
        for word in article.split():
            try:
                dataSet[i][wordBag.index(word)] += 1
            except ValueError:
                pass

    return dataSet
