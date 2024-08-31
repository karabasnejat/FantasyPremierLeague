import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st

# Streamlit başlık
st.title('Fantasy Premier League Analizi')

# Veri yükleme
@st.cache_data
def load_data():
    df = pd.read_csv('players.csv')
    return df

data = load_data()

# Kullanıcıdan pozisyon seçimi
position = st.selectbox(
    "Pozisyon Seçin",
    ("GKP", "DEF", "MID", "FWD")
)

# Seçilen pozisyona göre oyuncuları filtreleme
filtered_data = data[data['position'] == position].copy()

# Kullanılacak özelliklerin seçimi
features = [
    'goals_scored',
    'assists',
    'minutes',
    'clean_sheets',
    'expected_goals',
    'expected_assists',
    'influence',
    'creativity',
    'threat',
    'ict_index',
    'bonus'
]

# Filtrelenmiş veri kümesini oluşturma
X = filtered_data[features].copy()

# Eksik verilerin ortalama değerlerle doldurulması
X.fillna(X.mean(), inplace=True)

# Verilerin ölçeklendirilmesi (standartlaştırma)
X_scaled = StandardScaler().fit_transform(X)

# PCA ile boyut indirgeme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# K-means kümeleme
optimal_k = 4  # Küme sayısı
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Küme bilgilerini ve PCA bileşenlerini DataFrame'e ekleme
filtered_data.loc[:, 'cluster'] = clusters
filtered_data.loc[:, 'PCA1'] = X_pca[:, 0]
filtered_data.loc[:, 'PCA2'] = X_pca[:, 1]

# Etkileşimli Scatter Plot
fig = px.scatter(
    filtered_data,
    x='PCA1',
    y='PCA2',
    color='cluster',
    hover_name='name',
    hover_data=features,
    text='name',  # Oyuncu isimlerini doğrudan nokta üzerinde gösterir
    title=f'{position} Pozisyonu için Oyuncu Kümeleri',
    labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'}
)

# Metinleri noktaların yanına kaydırma (ofset) ve görünümü iyileştirme
fig.update_traces(textposition='top center')

# Grafiği gösterme
st.plotly_chart(fig)
