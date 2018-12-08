# emotion-chat

## API

- python3.6,pytorch(aws deeplearning ami,pytorch_p36)
- `pip install falcon` webフレームワーク
- `pip install gunicorn` wsgiサーバー

- corpus
  - `http://nlp.stanford.edu/data/glove.840B.300d.zip`
  - wget -> unzip
  - stanford_glove.txtが`api/analysis`配下にある状態で
  - `python ./api/transform.py`
  - glove形式をword2vecに変換する
  - download.shは使わない

- 起動 : `gunicorn api:api -t 1000`
  - `-t` タイムアウトオプション(読み込みに時間がかかる) 

- 定期的にリクエストを送って起こしてやる
  - `crontab req.conf`
  - 登録を消す `crontab -r`
  - `http://staffblog.amelieff.jp/entry/2018/07/06/150851`


## アプリ

- `cd webapp/webapp`  
- ローカルテスト `firebase serve`  
- デプロイ `firebase deploy`
- 会話データの初期化
  - firebaseのDatabaseコンソールで`webapp/data/all_data.json`をインポート
