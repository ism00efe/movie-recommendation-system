import pandas as pd

def get_hybrid_recommendations(user_id, predictions_df, matrix, df, movie_sequels):
    """
    Kullanıcıya 3 Popüler (Mainstream) ve 2 Niş (Niche) film önerir.
    Ayrıca seri takibi yaparak izlenmemiş öncül filmler için puan kırar.
    """
    # 1. Kullanıcının tahminlerini yüksekten düşüğe sırala
    user_preds = predictions_df.loc[user_id].sort_values(ascending=False)
    
    # 2. Kullanıcının zaten izlediği filmleri bul
    user_watched = matrix.loc[user_id].dropna().index.tolist()
    
    # 3. Popülerlik eşiği (Örn: 1000'den fazla oyu olanlar popülerdir)
    movie_counts = df['title'].value_counts()
    too_popular = movie_counts[movie_counts > 1000].index
    
    final_recs = []
    
    for movie, score in user_preds.items():
        if movie in user_watched:
            continue
            
        # Seri kontrolü: Eğer filmin bir öncülü varsa ve kullanıcı onu izlemediyse puanı %20 düşür
        adjusted_score = score
        if movie in movie_sequels:
            prequel = movie_sequels[movie]
            if prequel not in user_watched:
                adjusted_score *= 0.8 # Ceza puanı
        
        is_popular = movie in too_popular
        
        # Hibrit Seçim Mantığı: 3 Popüler + 2 Niş
        mainstream_count = len([r for r in final_recs if r['type'] == 'Mainstream'])
        niche_count = len([r for r in final_recs if r['type'] == 'Niche'])
        
        if is_popular and mainstream_count < 3:
            final_recs.append({'title': movie, 'type': 'Mainstream', 'score': adjusted_score})
        elif not is_popular and niche_count < 2:
            final_recs.append({'title': movie, 'type': 'Niche', 'score': adjusted_score})
            
        # 5 filme ulaştığımızda dur
        if len(final_recs) == 5:
            break
            
    return pd.DataFrame(final_recs)