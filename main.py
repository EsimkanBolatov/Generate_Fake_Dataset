from src.config import RAW_DATA, FEATURES_PATH, CLUSTERED_PATH, N_CLUSTERS, CLUSTERED_PATH_CSV, RAW_DATA_CSV, \
    FEATURES_PATH_CSV
from src.utils import load_parquet, save_parquet
from src.utils import load_csv, save_csv
from src.preprocessing import preprocess
from src.features import compute_features
from src.clustering import cluster_customers
from src.interpretation import assign_cluster_names
import generate_transactions

print('🔹 Загрузка данных')
df = load_parquet(RAW_DATA)
df1 = load_csv(RAW_DATA_CSV)

print('🔹 Предобработка')
df_clean = preprocess(df)
df1_clean = preprocess(df1)

print('🔹 Feature Engineering')
features = compute_features(df_clean)
features1 = compute_features(df1_clean)
save_parquet(features, FEATURES_PATH)
save_csv(features1, FEATURES_PATH_CSV)

print('🔹 Кластеризация')
clustered_df, model = cluster_customers(features, N_CLUSTERS)
clustered_df1, model1 = cluster_customers(features1, N_CLUSTERS)

print('🔹 Назначение имён кластерам')
clustered_df = assign_cluster_names(clustered_df)
clustered_df1 = assign_cluster_names(clustered_df1)
save_parquet(clustered_df, CLUSTERED_PATH)
save_csv(clustered_df1, CLUSTERED_PATH_CSV)

print('Готово!')
