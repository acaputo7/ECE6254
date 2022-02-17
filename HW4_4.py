import numpy as np
import json
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt

x = open('fedpapers_split.txt').read()
papers = json.loads(x)

papersH = papers[0] # papers by Hamilton
papersM = papers[1] # papers by Madison
papersD = papers[2] # disputed papers

nH, nM, nD = len(papersH), len(papersM), len(papersD)

# This allows you to ignore certain common words in English
# You may want to experiment by choosing the second option or your own
# list of stop words, but be sure to keep 'HAMILTON' and 'MADISON' in
# this list at a minimum, as their names appear in the text of the papers
# and leaving them in could lead to unpredictable results
stop_words = text.ENGLISH_STOP_WORDS.union({'hamilton','madison','united','states','new','york','1787','1788'})
# stop_words = {'hamilton','madison','united','states','new','york','1787','1788'}

## Form bag of words model using words used at least 10 times
vectorizer = text.CountVectorizer(stop_words=stop_words,min_df=10)
X = vectorizer.fit_transform(papersH+papersM+papersD).toarray()

# Uncomment this line to see the full list of words remaining after filtering out
# stop words and words used less than min_df times
# vocab = vectorizer.vocabulary_
# print(vocab)

# Split word counts into separate matrices
XH, XM, XD = X[:nH,:], X[nH:nH+nM,:], X[nH+nM:,:]


# Estimate probability of each word in vocabulary being used by Hamilton

WCH = np.sum(XH,axis=0)
TWCH = np.sum(WCH)
fH = (WCH+1)/(TWCH + X.shape[1]) #prob with Laplace smoothing

# Estimate probability of each word in vocabulary being used by Madison
WCM = np.sum(XM,axis=0)
TWCM = np.sum(WCM)
fM = (WCM+1)/(TWCM + X.shape[1])

# Compute ratio of these probabilities
fratio = fH/fM

# Compute prior probabilities
piH = nH/(nH+nM+nD)
piM = nM/(nH+nM+nD)

for xd in XD: # Iterate over disputed documents
    # Compute likelihood ratio for Naive Bayes model
    LR = np.prod(np.power(fratio, xd))
    if LR > piM/piH:
        print('Hamilton')
    else:
        print('Madison')
