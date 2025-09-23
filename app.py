from flask import Flask, request, render_template, redirect,jsonify
import clustering_algos.birch_algo as ba
import clustering_algos.k_mean_algo as ka
import clustering_algos.dbscan_algo as db
import clustering_algos.conn as conn
import pandas as pd
import numpy as np
from sklearn.cluster import Birch,KMeans
from kneed import KneeLocator
import threading
import time
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


import clustering_algos.elbow as elbow_clust

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/execute', methods=['GET', 'POST'])
def execute():
    data_size = int(request.form['data_size'])
    batch_size=int(request.form['batch_size'])
    validation_status = request.form['validation_status']
    algorithm = request.form['algorithm']
    if algorithm == 'BIRCH':
        total_clusters = int(request.form['total_cluster'])
        ba.run(data_size,batch_size,validation_status,total_clusters)
        return render_template("dashboard.html")
    if algorithm == 'KMEANS':
        total_clusters = int(request.form['total_cluster'])
        ka.run(data_size,batch_size,validation_status, total_clusters)
        return render_template("dashboard.html")
    if algorithm == 'DBSCAN':
        db.run(data_size,batch_size,validation_status)
        return render_template("dashboard.html")


@app.route('/auth', methods=['GET', 'POST'])
def auth():
    username = request.form['username']
    password = request.form['password']
    if username == "admin@outlieriq.rf.gd" and password == "admin@12345":
        return render_template("dashboard.html")
    else:
        return "error"

@app.route('/elbow', methods=['GET', 'POST'])
def elbow():
    data_size=int(request.form['data_size'])
    batch_size=int(request.form['batch_size'])
    if data_size!=0:
        elbow_clust.plot_elbow_graph(batch_size,data_size)
        return render_template("dashboard.html")
    else:
        return render_template("dashboard.html")



@app.route('/logout')
def logout():
    return render_template("index.html")


@app.route('/index.php')
def php():
    return "i am php request"



#---------------Real Time Birch Analysis--------------

def calculate_optimal_clusters(data):
    if data.empty or data.shape[0] < 2:
        raise ValueError("Insufficient data for clustering.")

    if data[['temp', 'hum']].nunique().min() <= 1:
        raise ValueError("Input data contains constant features. Clustering is not meaningful.")

    sse = []
    X = data[['temp', 'hum']].values

    for k in range(2, 11):
        birch_model = Birch(n_clusters=k)
        birch_model.fit(X)
        subcluster_count = len(birch_model.subcluster_centers_)
        sse.append(subcluster_count)

    if not sse or all(val == 0 for val in sse):
        raise ValueError("Subcluster count is zero for all cluster sizes. Check the input data.")

    kl = KneeLocator(range(2, 11), sse, curve="convex", direction="decreasing")
    optimal_clusters = kl.elbow or 3

    if optimal_clusters < 2:
        raise ValueError("Insufficient variability to form multiple clusters.")

    return optimal_clusters


def create_initial_dataframe():
    data = conn.connection(10000, 100000)
    df = pd.DataFrame(data)
    return df

df = create_initial_dataframe()
n_clusters = calculate_optimal_clusters(df)

def convert_objectid_to_string(data):
    for item in data:
        if '_id' in item:
            item['_id'] = str(item['_id'])
    return data

def update_clusters():
    global n_clusters
    while True:
        n_clusters = calculate_optimal_clusters(df)
        threading.Event().wait(60)


cluster_update_thread = threading.Thread(target=update_clusters)
cluster_update_thread.start()

@app.route('/rta_birch')
def rta_birch():
    return render_template('rta_birch.html')

@app.route('/data')
def get_data():
    data = df.to_dict(orient='records')
    data = convert_objectid_to_string(data)
    return jsonify(data)


@app.route('/update_data')
def update_data():
    global df, n_clusters, final_time, size, no_i

    new_data = conn.connection(10, 100)
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)
    X = df[['temp', 'hum']].values
    print(f"Number of clusters: {n_clusters}")
    birch_model = Birch(n_clusters=n_clusters)
    birch_model.fit(X)
    df['cluster'] = birch_model.predict(X)

    # Convert the last 100000 records to dictionary format for returning
    data = df.tail(100000).to_dict(orient='records')
    data = convert_objectid_to_string(data)
    return jsonify(data)
#---------------Real Time Birch Analysis--------------




if __name__ == '__main__':
    app.run()
