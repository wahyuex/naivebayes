from flask import Flask, redirect, url_for, request

app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
    return 'velcome %s' % name


@app.route('/login', methods=['POST', 'GET'])

def login():
    print(request.method)
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name =user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name =user))