# emotion-chat
## API
`cd corpus-api`  
ローカルテスト `chalice local`  
デプロイ `chalice deploy`

APIのテスト  
httpieのインストール `brew install httpie`  
テスト `http POST 127.0.0.1:8000 name=fuga`   
127.0.0.1:8000に{name:'fuga'}をPOSTする

## アプリ
`cd friendlychat/web-start`  
ローカルテスト `firebase serve`  
デプロイ `firebase deploy`