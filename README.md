# emotion-chat

Implementation for a paper,  
`"Automated Font Selection System based on Message Sentiment in English Text-Based Chat"`  
Font data and trained model are excluded in this repository.

## Components
```
.  
├── api         --- api and notebooks for VAD training and prediction  
├── scripts     --- scripts for download, etc  
└── webapp      --- firebase application for chat  
```

## API

- Environment
  - python3.6
  - pytorch(aws deeplearning ami,pytorch_p36)
  - falcon, gunicorn

- Corpus setting
	- `http://nlp.stanford.edu/data/glove.840B.300d.zip`
  - `python ./api/transform.py`
    - put `stanford_glove.txt` in `api/analysis/`
    - convert glove to word2vec

-  Start server `gunicorn api:api -t 1000`
	- You may have to set timeout option `-t` since it would be a long time.
- Send requests periodically so that a server doesn't sleep.
	- register `crontab req.conf`
	- remove   `crontab -r`
	- reference `http://staffblog.amelieff.jp/entry/2018/07/06/150851`


## webapp

This webapp is based on firebase. 
- `cd webapp/webapp`  
- Run in local `firebase serve`  
- Deploy `firebase deploy`

## Reference

- Paper
	- Ryota Yonekura, Saemi Choi, Ryota Yoshihashi, Katsufumi Matsui and Ari Hautasaari, “Automated Font Selection System based on Message Sentiment in English Text-Based Chat”, IEICE Technical Report, vol. 118, no. 502, IMQ2018-45, IE2018-129, MVE2018-76, pp. 131-136, March 2019

- EmoBank
	- https://github.com/JULIELab/EmoBank
	- Sven Buechel and Udo Hahn. 2017. EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis. In EACL 2017 - Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics. Valencia, Spain, April 3-7, 2017. Volume 2, Short Papers, pages 578-585. Available: http://aclweb.org/anthology/E17-2092
	- Sven Buechel and Udo Hahn. 2017. Readers vs. writers vs. texts: Coping with different perspectives of text understanding in emotion annotation. In LAW 2017 - Proceedings of the 11th Linguistic Annotation Workshop @ EACL 2017. Valencia, Spain, April 3, 2017, pages 1-12. Available: https://sigann.github.io/LAW-XI-2017/papers/LAW01.pdf

- FriendlyChatapp
	- https://codelabs.developers.google.com/codelabs/firebase-web/#0
	- https://github.com/firebase/friendlychat-web
