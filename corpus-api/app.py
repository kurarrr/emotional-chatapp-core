import json
from chalice import Chalice,BadRequestError

app = Chalice(app_name='corpus-api')


@app.route('/',methods=['POST'], cors=True)
def index():
    data = app.current_request.json_body
    try:
        msg = data['msg']
        msg_user_attr = data['msg_user_attr']
    except:
        raise BadRequestError("Keys [msg,msg_user_attr] are necessary")

    res = {}

    if msg_user_attr == 'native':
        effect_to = 'non-native'
    elif msg_user_attr == 'non-native':
        effect_to = 'native'

    # ここでメッセージを処理する
    if 'happy' in msg or 'happiness' in msg:
        effect = 'happiness'
    elif 'sad' in msg:
        effect = 'sadness'
    elif 'anger' in msg or 'angry' in msg:
        effect = 'anger'
    elif 'disgust' in msg:
        effect = 'disgust'
    elif 'fear' in msg:
        effect = 'fear'
    elif 'surprise' in msg:
        effect = 'surprise'
    else:
        effect = 'none'
    
    res[effect_to] = 'effect-'+effect

    return { 'effect' : res }


# The view function above will return {"hello": "world"}
# whenever you make an HTTP GET request to '/'.
#
# Here are a few more examples:
#
# @app.route('/hello/{name}')
# def hello_name(name):
#    # '/hello/james' -> {"hello": "james"}
#    return {'hello': name}
#
# @app.route('/users', methods=['POST'])
# def create_user():
#     # This is the JSON body the user sent in their POST request.
#     user_as_json = app.current_request.json_body
#     # We'll echo the json body back to the user in a 'user' key.
#     return {'user': user_as_json}
#
# See the README documentation for more examples.
#
