import pandas as pd

def load_and_preprocess(ratings_path, movies_path):
    """Verileri diskten okur ve ham birleşik DataFrame'i döner."""
    movies = pd.read_csv(movies_path, encoding='latin-1', sep=';', usecols=[0, 1, 2])
    ratings = pd.read_csv(ratings_path, encoding='latin-1', sep=';', usecols=[0, 1, 2, 3])
    
    # 2. Birleştir
    df = pd.merge(ratings, movies, on='movieId')
    # 3. Aktif Kullanıcı Filtresi (En az 50 oy)
    user_counts = df['userId'].value_counts()
    active_users = user_counts[user_counts >= 50].index
    df = df[df['userId'].isin(active_users)]
    
    # 4. Popüler Film Filtresi (En az 50 oy)
    movie_counts = df['title'].value_counts()
    popular_movies = movie_counts[movie_counts >= 50].index
    df = df[df['title'].isin(popular_movies)]
    
    return df

def create_pivot_matrix(df):
    """
    Temizlenmiş DataFrame'den pivot tablo oluşturur 
    ve Mean Centering (Normalizasyon) uygular.
    """
    # Pivot tabloyu oluştur
    pivot = df.pivot_table(index='userId', columns='title', values='rating')
    
    # Dengeleme (Mean Centering) - Kullanıcı bazlı ortalamayı çıkarıyoruz
    pivot_centered = pivot.apply(lambda x: x - x.mean(), axis=1)
    
    # NaN -> 0 (SVD için boşlukları 0 ile dolduruyoruz)
    return pivot, pivot_centered.fillna(0)