from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor

def fetch_data(collection, query, batch_size):
    cursor = collection.find(query).sort('_id', -1).limit(batch_size)
    return list(cursor)

def connection(batch_size, total_size):
    connection_url = ""
    client = MongoClient(connection_url)
    db = client['iot_data']
    collection = db['sensor_data']

    results = []
    last_id = None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        while len(results) < total_size:
            query = {} if last_id is None else {'_id': {'$lt': last_id}}
            futures.append(executor.submit(fetch_data, collection, query, batch_size))

            # Process completed futures
            for future in futures:
                chunk = future.result()
                if not chunk:
                    continue
                results.extend(chunk)
                last_id = chunk[-1]['_id']

                print(f"Fetched {len(results)} records so far...")

            if len(results) >= total_size:
                break
    return results

# data=connection(10000,100000)
# import pandas as pd
# df=pd.DataFrame(data)
# df.to_csv("sample_data.csv")
