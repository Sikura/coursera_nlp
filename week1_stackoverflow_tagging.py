# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:31:16 2018

@author: vshar222
"""

import os
os.chdir("C:\\Users\\vshar222\\Desktop\\Python\\NLP\\natural-language-processing-master\\natural-language-processing-master\\week1")

from nltk.corpus import stopwords
import re
from nltk import word_tokenize

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

text = "How to free c++ memory vector<int> * arr?"
text = text.lower() # lowercase text
text = re.sub(REPLACE_BY_SPACE_RE, " ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
text = re.sub(BAD_SYMBOLS_RE, "", text)# delete symbols which are in BAD_SYMBOLS_RE from text
text = " ".join([word for word in word_tokenize(text) if word not in STOPWORDS])# delete stopwords from text

print(text)
# free c++ memory vectorint arr