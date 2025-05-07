import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


file_path = 'D:/fadsdiet-foundational/data/fadsdiet_preprocessed.csv'

df = pd.read_csv(file_path)

print(f"Shape of the data: {df.shape}")

targets = ['fad8_chol', 'fad8_ldlc', 'fad8_hdlc', 'fad8_tg', 'fad8_crp', 'fad8_gluc0',  
    'fad8_hba1c', 'fad8_glauc', 'fad8_gluc0']

for target in targets:
    columns_to_drop = df.filter(regex='^fad8').columns.difference([target]).tolist()
    df_dropped = df.drop(columns=columns_to_drop)

    print(f"Shape of the data after dropping columns: {df_dropped.shape}")

    # Random forest feature importance
    X = df_dropped.drop(columns=[target])
    y = df_dropped[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    feature_importances = rf.feature_importances_

    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # feature_importances_df.to_csv('D:/fadsdiet-foundational/data/feature_importances.csv', index=False)

    print("Feature importances:")
    print(feature_importances_df)

    # drop features with importance less than 0.0004 but don't drop MOMID,diet,genotype,group,fad0_weight,fad0_height,fad0_bmi,fad0_waist
    # fad8_chol, fad8_ldlc, fad8_hdlc, fad8_tg, fad8_crp, fad8_gluc0, fad8_gluc30, fad8_gluc120, fad8_hba1c, fad8_ins0, fad8_ins30, fad8_ins120, fad8_glauc

    columns_to_keep = ['MOMID', 'diet', 'genotype', 'group', 'fad0_weight', 'fad0_height', 'fad0_bmi', 'fad0_waist']

    columns_to_drop = feature_importances_df[
        (feature_importances_df['Importance'] < 0.0004) & 
        (~feature_importances_df['Feature'].isin(columns_to_keep))
    ]['Feature'].tolist()

    df_dropped = df_dropped.drop(columns=columns_to_drop)

    print(f"Shape of the data after dropping columns with importance less than 0.0004: {df_dropped.shape}")

    # add fad8_chol back to the dataframe
    df_dropped = pd.concat([df_dropped, df[target]], axis=1)
    df_dropped.to_csv(f'D:/fadsdiet-foundational/data/fadsdiet_preprocessed_{target}.csv', index=False)