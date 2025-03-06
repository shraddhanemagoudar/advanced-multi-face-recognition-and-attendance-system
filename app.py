from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/attendance")
def attendance():
    df = pd.read_csv("attendance.csv")
    return df.to_html()

if __name__ == "__main__":
    app.run(debug=True)
