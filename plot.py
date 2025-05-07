import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

################################################################
# Plot correlations
################################################################

# Load the data
# target = 'fad8_glauc'  # Replace with your actual target variable
# file_path = f'D:/fadsdiet-foundational/data/fadsdiet_preprocessed_{target}.csv'
# df = pd.read_csv(file_path)

# Visualize the fad8_glauc feature
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=df['fad8_glauc'])
# plt.title('Boxplot of fad8_glauc')
# plt.show()

# Check for correlations with other features
# correlation_matrix = df.corr()
# fad8_glauc_corr = correlation_matrix['fad8_glauc'].sort_values(ascending=False)

# print("Correlations with fad8_glauc:")
# print(fad8_glauc_corr)



################################################################
# Plot errors
################################################################

foundational_data = {
    'Target Variable': ['fad8_chol', 'fad8_ldlc', 'fad8_hdlc', 'fad8_tg', 'fad8_crp', 'fad8_gluc0'],
    'MSE_Foundational': [0.17, 0.09, 0.04, 0.08, 0.05, 0.08],
    'MAE_Foundational': [0.34, 0.26, 0.16, 0.23, 0.08, 0.22]
}

rf_data = {
    'Target Variable': ['fad8_chol', 'fad8_ldlc', 'fad8_hdlc', 'fad8_tg', 'fad8_crp', 'fad8_gluc0'],
    'MSE_RF': [0.23, 0.17, 0.07, 0.06, 0.60, 0.11],
    'MAE_RF': [0.38, 0.34, 0.19, 0.19, 0.61, 0.25]
}

df_found = pd.DataFrame(foundational_data)
df_rf = pd.DataFrame(rf_data)
df = pd.merge(df_found, df_rf, on='Target Variable')

plt.figure(figsize=(14, 6))

# MSE 
plt.subplot(1, 2, 1)
plt.plot(df['Target Variable'], df['MSE_Foundational'], marker='o', label='MSE Foundational', color='deepskyblue')
plt.plot(df['Target Variable'], df['MSE_RF'], marker='o', label='MSE Random Forest', color='tomato')
plt.title('MSE comparison')
plt.xlabel('Target')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
#plt.show()

# MAE 
plt.subplot(1, 2, 2)
plt.plot(df['Target Variable'], df['MAE_Foundational'], marker='s', label='MAE Foundational', color='blue')
plt.plot(df['Target Variable'], df['MAE_RF'], marker='s', label='MAE Random Forest', color='red')
plt.title('MAE comparison')
plt.xlabel('Target')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
# plt.xt