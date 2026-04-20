import sys
import os
import pandas as pd

# 1. YOL TANIMLAMA (Path Fix)
# api klasörünün içinde olduğun için bir üst dizini (root) Python'a tanıtıyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)

# 2. MODÜLER IMPORTLAR
# src/__init__.py üzerinden tüm fonksiyonları tek seferde çekiyoruz
from src import (
    load_and_preprocess, 
    create_pivot_matrix, 
    train_model, 
    load_saved_model,
    calculate_rmse, 
    get_hybrid_recommendations
)

def main():
    # DOSYA YOLLARI
    # Root dizinine göre data ve models klasörlerinin yerini belirliyoruz
    ratings_path = os.path.join(root_dir, 'data', 'raw', 'ratings.csv')
    movies_path = os.path.join(root_dir, 'data', 'raw', 'movies.csv')
    model_dir = os.path.join(root_dir, 'models')
    model_path = os.path.join(model_dir, 'svd_predictions.pkl')

    print("\n--- 1. ADIM: Veri Yükleme ve Önişleme ---")
    # data_loader.py: Veriyi okur ve filtreler
    df = load_and_preprocess(ratings_path, movies_path)
    
    # data_loader.py: Pivot & Normalizasyon matrislerini hazırlar
    matrix, matrix_norm = create_pivot_matrix(df)
    print(f"Veri hazırlandı. Matris Boyutu: {matrix.shape}")

    print("\n--- 2. ADIM: Model Yükleme veya Eğitme ---")
    # engine.py: Önce models/ klasöründe hazır model var mı diye bakar
    predictions_df = load_saved_model(model_path)

    if predictions_df is not None:
        print(f"Hazır model bulundu: {model_path}. Yükleniyor...")
    else:
        print("Hazır model bulunamadı. Model eğitiliyor ve kaydediliyor...")
        # engine.py: Modeli eğitir ve belirtilen yola kaydeder
        predictions_df = train_model(matrix, matrix_norm, n_components=50, save_path=model_path)
        print(f"Model eğitildi ve şuraya kaydedildi: {model_path}")

    print("\n--- 3. ADIM: Model Değerlendirme (RMSE) ---")
    # evaluator.py: Tahmin başarısını ölçer
    rmse_score = calculate_rmse(matrix, predictions_df)
    print(f"Sistem RMSE Skoru: {rmse_score:.4f}")

    print("\n--- 4. ADIM: Kullanıcı Önerileri ---")
    # Seri takibi için gereken sözlük (Geliştirilebilir)
    movie_sequels = {
        "Star Wars: Episode V - The Empire Strikes Back (1980)": "Star Wars: Episode IV - A New Hope (1977)",
        "Godfather: Part II, The (1974)": "Godfather, The (1972)",
        "Star Wars: Episode VI - Return of the Jedi (1983)": "Star Wars: Episode V - The Empire Strikes Back (1980)",
        "Aliens (1986)": "Alien (1979)",
        "Terminator 2: Judgment Day (1991)": "Terminator, The (1984)"
    }
    
    # recommender.py: 3 Popüler + 2 Niş mantığıyla öneri üretir
    user_id = 42
    recommendations = get_hybrid_recommendations(user_id, predictions_df, matrix, df, movie_sequels)
    
    print(f"\nKullanıcı {user_id} için Dengeli Öneriler:")
    print("-" * 50)
    print(recommendations[['title', 'type', 'score']])
    print("-" * 50)

if __name__ == "__main__":
    main()