#!/bin/sh
# download word2vec
FILE_ID=0B7XkCwpI5KDYNlNUTTlSS21pQmM
FILE_NAME=google_word2vec.bin.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
