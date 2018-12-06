from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

from gensim.scripts.glove2word2vec import glove2word2vec
# transform : glove -> tmp
glove2word2vec('./analysis/stanford_glove.txt', './analysis/stanford_w2v.txt')