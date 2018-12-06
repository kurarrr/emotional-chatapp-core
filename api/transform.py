from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

from gensim.scripts.glove2word2vec import glove2word2vec
# transform : glove -> tmp
glove2word2vec('./analysis/glove.840B.300d.txt', './analysis/stanford_w2v.txt')