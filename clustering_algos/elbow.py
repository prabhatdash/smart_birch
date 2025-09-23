import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import clustering_algos.conn as con
from kneed import KneeLocator  # Library to find the elbow point

def plot_elbow_graph(batch_size, data_size):
    try:
        data = con.connection(batch_size, data_size)
        df = pd.DataFrame(data, columns=['temp', 'hum', 'device_id', 'location', 'time_stamp'])
        df = df.iloc[:data_size]
        numeric_cols = ['temp', 'hum']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        data = df[['temp', 'hum']]

        inertia = []
        k_range = range(1, 11)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)
        kn = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
        optimal_clusters = kn.knee

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=inertia, mode='lines+markers'))
        fig.update_layout(
            title=f'Elbow Method For Optimal k (Optimal Clusters: {optimal_clusters})',
            xaxis_title='Number of clusters',
            yaxis_title='Inertia'
        )
        fig.show()

        print(f'The optimal number of clusters is: {optimal_clusters}')

    except Exception as e:
        print(e)

# Example usage
# plot_elbow_graph(batch_size=30000, data_size=300000)
