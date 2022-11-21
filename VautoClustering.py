import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format
from sklearn.preprocessing import StandardScaler
from pycaret.clustering import setup, create_model, evaluate_model, plot_model, assign_model, save_model

def make_model(df:pd.DataFrame, num_clusters:int=4, seed:int=42) -> tuple:
    scaler = StandardScaler()
    scaler.fit(df)

    setup(df, normalize=True, session_id=seed, silent=True)
    kmeans = create_model('kmeans', num_clusters=num_clusters)
    save_model(kmeans, 'best_model')

    result = assign_model(kmeans)

    knn_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    knn_centers = pd.DataFrame(knn_centers, columns=df.columns)    

    if __name__ == '__main__':
        print('cluster assignment:')
        print(result)
        print('cluster centers:')
        print(knn_centers)
        display_format = None
        savefig = True
    else:
        import streamlit as st
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download the model file', f, 'trained_model.pkl')
        st.info('Note: model can be loaded similar to the following,  \n  \
                from pycaret.classification import load_model  \n \
                model = load_model("trained_model.pkl")')
        st.write('cluster assignment:')
        st.dataframe(result)
        st.write('cluster centers:')
        st.dataframe(knn_centers.style.format("{:.2f}"))
        savefig = False

        display_format = 'streamlit'
    plot_model(kmeans, plot='elbow', display_format=display_format, save=savefig)
    plot_model(kmeans, plot='silhouette', display_format=display_format, save=savefig)
    plot_model(kmeans, plot='cluster', display_format=display_format, save=savefig)
    plot_model(kmeans, plot='tsne', display_format=display_format, save=savefig)
    plot_model(kmeans, plot='distance', display_format=display_format, save=savefig)

    print(kmeans)

    return kmeans, scaler

if __name__ == '__main__':
    # create test data
    from sklearn.datasets import make_blobs
    features, _ = make_blobs(n_samples = 2000,
                                    n_features = 10, 
                                    centers = 5,
                                    cluster_std = 0.4,
                                    shuffle = True)
    df = pd.DataFrame(features, columns=["Feature 1", "Feature 2", "Feature 3",
         "Feature 4", "Feature 5", "Feature 6", "Feature 7", "Feature 8",
         "Feature 9", "Feature 10"])
    

    make_model(df, num_clusters=5)