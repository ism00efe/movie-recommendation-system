from sklearn.metrics import mean_squared_error
import math

def calculate_rmse(matrix, predictions_df):
    """
    Gerçek puanlar ile tahmin edilen puanlar arasındaki RMSE değerini hesaplar.
    """
    # Gerçek puanları (NaN olmayanlar) çek
    actual_ratings = matrix.stack().reset_index()
    actual_ratings.columns = ['userId', 'title', 'actual_rating']
    
    # Tahmin edilen puanları çek
    predicted_ratings = predictions_df.stack().reset_index()
    predicted_ratings.columns = ['userId', 'title', 'predicted_rating']
    
    # İki tabloyu birleştir (Sadece gerçek oyların olduğu satırlar kalır)
    eval_df = actual_ratings.merge(predicted_ratings, on=['userId', 'title']).dropna()
    
    if eval_df.empty:
        return 0.0
        
    mse = mean_squared_error(eval_df['actual_rating'], eval_df['predicted_rating'])
    rmse = math.sqrt(mse)
    
    return rmse