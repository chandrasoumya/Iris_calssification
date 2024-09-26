from flask import Flask, render_template ,request
import pickle

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def home():
    Species = None

    if request.method == 'POST':
        sl =float(request.form['sepal_length'])
        sw =float(request.form['sepal_width'])
        pl =float(request.form['petal_length'])
        pw =float(request.form['petal_width'])
        sample = [[sl,sw,pl,pw]]

        with open('iris.pkl', 'rb') as f:
            model = pickle.load(f)

        res = model.predict(sample)
        
        if res == 0 :
            Species = 'Iris-setosa'
        elif res == 1:
            Species = 'Iris-versicolor'
        else:
            Species = 'Iris-virginica'

    return render_template('index.html',species=Species)

if __name__ == "__main__":
    app.run(debug=True,port=5000)