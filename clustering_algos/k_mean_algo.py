import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import time
import clustering_algos.conn as con

def run(data_size,batch_size,validation_status, n_clusters):
    try:
        data = con.connection(batch_size, data_size)
        start = time.time()
        df = pd.DataFrame(data, columns=['temp', 'hum', 'device_id', 'location', 'time_stamp'])
        df = df.iloc[:data_size]

        numeric_cols = ['temp', 'hum']

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        print(df)

        features = ['temp', 'hum']
        data = df[features]

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data_scaled)
        labels = kmeans.labels_

        if validation_status == "1":
            silhouette =" | Silhouette Score: "+str( silhouette_score(data_scaled, labels))
            print(silhouette)
        else:
            silhouette = " "

        df['cluster'] = labels



        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=df['temp'],
            y=df['hum'],
            z=df['cluster'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['cluster'],
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Data Points',
            hovertemplate='<b>Temperature</b>: %{x}<br><b>Humidity</b>: %{y}<br><b>Cluster</b>: %{z}<br>'
                          '<b>Device ID</b>: %{customdata[0]}<br><b>Location</b>: %{customdata[1]}<br>'
                          '<b>Timestamp</b>: %{customdata[2]}<br><extra></extra>',
            customdata=df[['device_id', 'location', 'time_stamp']]
        ))
        # Stop timer
        stop = time.time()
        exc_time = stop - start
        title = 'K-Means Clustering in 3D'

        fig.update_layout(scene=dict(
            xaxis_title='Temperature',
            yaxis_title='Humidity',
            zaxis_title='Cluster'),
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            )
        )
        fig.show()
        stop = time.time()
        exc_time = stop - start
        print(f"Total Execution Time: {exc_time:.2f} seconds")
        return exc_time
    except Exception as e:
        print("Error:", e)


# size=100000
# b_size = 10000
# final_time=0.0
# for i in range(1,101):
#     timet=run(data_size=size, batch_size=b_size, validation_status="0", n_clusters=3)
#     final_time= final_time + timet
#     opo="Iteration No: ",i
#     print(opo)
#     op="Data Size: ",size," | Batch Size: ",b_size," | Final Time: ",final_time
#     print(op)
#     size=int(size+100)
#     b_size=int(size/10)
#     with open('../Results/kmeans_results.txt', 'a') as file:
#         # List of lines to append, ensuring they all start on new lines
#         lines_to_add = [
#             str(opo),
#             '\n'+str(timet),
#             '\n'+str(op),
#             '\n-------------------------'
#             '\n'
#         ]
#         # Append the list of lines to the file
#         file.writelines(lines_to_add)