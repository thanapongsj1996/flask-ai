from flask import Flask
import numpy as np
import pandas as pd
from joblib import load


app = Flask(__name__)


def get_model():
    clf3 = load('filename.joblib')
    return clf3

@app.route('/')
def hello_geek():
    print("-----------------")
    clf = get_model()
    data = pd.read_csv('trade_data.csv')
    start_test_position = int(len(data) * 80 / 100 )

    x = data.iloc[:, 3:-2]
    y = data.iloc[:, -1:]
    x_test = x.iloc[start_test_position:]
    y_test = y.iloc[start_test_position:]

    my_array = np.array([[0.012689,88.6868,90.6858,4.278100,4.293000], [0.000136,	46.6008,	12.1795,	-1.673270,	-0.863508]])
    df = pd.DataFrame(my_array, columns = ['close1_ema33','rsi14','sto533','pos_bb20','sd_pos'])
    r = clf.predict(df)

    # acc2 = clf.score(x_test, y_test)
    # print(x_test)
    # print(acc2)
    print(r)
    return '<h1>Hello from Flask & Dockerzz -></h2>'


if __name__ == "__main__":
    app.run(debug=True)