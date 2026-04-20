import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import joblib
import os

def train_model(matrix, matrix_norm, n_components=50, save_path=None):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_features = svd.fit_transform(matrix_norm.to_numpy())
    movie_features = svd.components_
    
    predicted_ratings_array = np.dot(user_features, movie_features)
    user_ratings_mean = matrix.mean(axis=1).values.reshape(-1, 1)
    
    predictions_df = pd.DataFrame(
        predicted_ratings_array + user_ratings_mean, 
        columns=matrix.columns, 
        index=matrix.index
    )

    # Eğer bir yol verildiyse modeli oraya fırlat
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(predictions_df, save_path)
        print(f"Model başarıyla kaydedildi: {save_path}")

    return predictions_df

def load_saved_model(save_path):
    """Eğer önceden eğitilmiş model varsa yükler."""
    if os.path.exists(save_path):
        return joblib.load(save_path)
    return None