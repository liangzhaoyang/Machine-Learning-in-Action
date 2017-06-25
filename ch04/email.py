import random
import pickle
import articles
import bayes

def getArticles(filename):
    file = open(filename, 'rb')
    articles = pickle.load(file)
    file.close()
    return articles

ham = getArticles('data/email/ham.pickle')
spam = getArticles('data/email/spam.pickle')

# Generate training set and test set
random.shuffle(ham)
random.shuffle(spam)

hamTestLen = len(ham) // 3
spamTestLen = len(spam) // 3
testData = ham[:hamTestLen] + spam[:spamTestLen]
testLabels = ['ham' for i in range(hamTestLen)] + ['spam' for i in range(spamTestLen)]
trainData = ham[hamTestLen:] + spam[spamTestLen:]
trainLabels = ['ham' for i in range(len(ham) - hamTestLen)] + ['spam' for i in range(len(spam) - spamTestLen)]

# Train model
wordBag = articles.createWordBag(trainData)
trainData = articles.createDataSet(trainData, wordBag)
model = bayes.train(trainData, trainLabels)

# Test model
correct = 0
testData = articles.createDataSet(testData, wordBag)
for i, data in enumerate(testData):
    res = bayes.classify(data, model)
    if res == testLabels[i]:
        correct += 1
print('Correctness: %d/%d' % (correct, len(testData)))
