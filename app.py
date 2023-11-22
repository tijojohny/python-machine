# from flask import Flask, render_template
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)
# # template_dir = os.path.abspath('templates')
# # app.template_folder = template_dir
# @app.route('/')
# def index():
#     # Load data
#     data = pd.read_csv('/CC_GENERAL.csv')
#     X = data[['BALANCE', 'PURCHASES']]

#     # Standardize the data
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     # Determine the optimal number of clusters using the elbow method
#     wcss = []
#     for i in range(1, 11):
#         kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#         kmeans.fit(X)
#         wcss.append(kmeans.inertia_)

#     # Select the optimal number of clusters
#     n_clusters = 3

#     # Apply K-Means clustering
#     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X)

#     # Add the cluster labels to the DataFrame
#     data['Cluster'] = kmeans.labels_

#     # Create a dictionary to pass data to the template
#     chart_data = {
#         'chart_title': 'K-Means Clustering',
#         'x_label': 'BALANCE',
#         'y_label': 'PURCHASES',
#         'labels': kmeans.labels_.tolist(),
#         'centers': scaler.inverse_transform(kmeans.cluster_centers_).tolist(),
#         'counts': data['Cluster'].value_counts().to_dict()
#     }

#     return render_template('index.html', chart_data=chart_data)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Set the template folder explicitly
template_dir = os.path.abspath('templates')
app.template_folder = template_dir
def get_chart_data():
    # Add your logic to generate chart data
    chart_data = {
        'chart_title': 'K-Means Clustering',
        'x_label': 'BALANCE',
        'y_label': 'PURCHASES',
        'labels': [1, 2, 3],  # Example values, replace with actual data
        'centers': [[1, 2], [3, 4], [5, 6]],  # Example values, replace with actual data
        'counts': {1: 10, 2: 15, 3: 20}  # Example values, replace with actual data
    }
    return chart_data
@app.route('/')
def index():
    # return render_template('index.html')
    chart_data = get_chart_data()
    return render_template('index.html', chart_data=chart_data)


@app.route('/submit_data', methods=['POST'])
def submit_data():
    # Retrieve data from the AJAX request
    input_data = request.get_json()

    # Access the balance and purchase values
    balance_value = float(input_data.get('balance', 0))
    purchase_value = float(input_data.get('purchase', 0))

    # Filter data based on balance and purchase values
    filtered_data = data[(data['BALANCE'] == balance_value) & (data['PURCHASES'] == purchase_value)]

    # Convert the filtered data to a list of dictionaries
    result = filtered_data.to_dict(orient='records')
# print(chart_data)
# return render_template('index.html', chart_data=chart_data)
    # Return the filtered data as chart_data
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/')
# def index():
#     # Load data
#     data = pd.read_csv('/CC_GENERAL.csv')
#     X = data[['BALANCE', 'PURCHASES']]

#     # Standardize the data
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     # Determine the optimal number of clusters using the elbow method
#     wcss = []
#     for i in range(1, 11):
#         kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#         kmeans.fit(X)
#         wcss.append(kmeans.inertia_)

#     # Select the optimal number of clusters
#     n_clusters = 3

#     # Apply K-Means clustering
#     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X)

#     # Add the cluster labels to the DataFrame
#     data['Cluster'] = kmeans.labels_

#     # Create a dictionary to pass data to the template
#     chart_data = {
#         'chart_title': 'K-Means Clustering',
#         'x_label': 'BALANCE',
#         'y_label': 'PURCHASES',
#         'labels': kmeans.labels_.tolist(),
#         'centers': scaler.inverse_transform(kmeans.cluster_centers_).tolist(),
#         'counts': data['Cluster'].value_counts().to_dict()
#     }

#     return render_template('index.html', chart_data=chart_data)

# if __name__ == '__main__':
#     app.run(debug=True)
