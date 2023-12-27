from flask import Flask

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return 'index Page'
def hi():
    return 'hi,Guys'
print("hello word")
if __name__ == '__main__':
    app.run()