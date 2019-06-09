import json,random
from chalice import Chalice,BadRequestError

app = Chalice(app_name='corpus-api')


@app.route('/prediction_api',methods=['GET'], cors=True)
def index():
    req = app.current_request
    data = req.query_params

    try:
        msg = data['msg']
    except:
        raise BadRequestError("Keys [msg] are necessary")

    res = {
        'Valence' : random.random(),
        'Arousal' : random.random()
    }

    return res


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
