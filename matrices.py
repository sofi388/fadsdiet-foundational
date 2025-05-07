import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


################################################################
# Correlation matrices of biomarkers and nutrition markers
################################################################

file_path = f'D:/fadsdiet-foundational/data/fadsdiet_preprocessed.csv'
df = pd.read_csv(file_path)

print(df.shape)

column_names = df.columns.tolist()
column_names_df = pd.DataFrame(column_names, columns=['Column Names'])

column_names_df.to_csv('D:/fadsdiet-foundational/data/column_names_column.csv', index=False)

df_subset_bio = df[['MOMID', 'diet', 'genotype', 'group', 'fad0_weight', 'fad0_height', 
                'fad0_bmi', 'fad0_waist', 'fad0_systbp1', 'fad0_diastbp1', 'fad0_systbp2', 
                'fad0_diastbp2', 'fad0_systbp3', 'fad0_diastbp3', 'fad0_systbp', 'fad0_diastbp',
                'fad0_chol', 'fad0_ldlc', 'fad0_hdlc', 'fad0_tg', 'fad0_alt', 'fad0_creat', 
                'fad0_crp', 'fad0_gluc0', 'fad0_gluc30', 'fad0_gluc120', 'fad0_hba1c', 'fad0_ins0', 
                'fad0_ins30', 'fad0_ins120', 'fad0_glauc', 'fad0_gliauc', 'fad0_insauc', 'fad0_insiauc',
                'fad0_matsuda', 'fad0_di', 'fad0_homair', 'fad0_homais', 'fad0_insgenin', 'fad0_inssec30',
                'fad8_weight', 'fad8_height', 'fad8_bmi', 'fad8_waist', 'fad8_systbp1', 'fad8_diastbp1', 
                'fad8_systbp2', 'fad8_diastbp2', 'fad8_systbp3', 'fad8_diastbp3', 'fad8_systbp', 'fad8_diastbp',
                'fad8_chol', 'fad8_ldlc', 'fad8_hdlc', 'fad8_tg', 'fad8_crp', 'fad8_gluc0', 'fad8_gluc30', 
                'fad8_gluc120', 'fad8_hba1c', 'fad8_ins0', 'fad8_ins30', 'fad8_ins120', 'fad8_glauc', 'fad8_gliauc', 
                'fad8_insauc', 'fad8_insiauc', 'fad8_matsuda', 'fad8_di', 'fad8_homair', 'fad8_homais', 'fad8_insgenin',
                'fad8_inssec30', 'rs174550', 'Lnfad0_crp', 'Lnfad8_crp', 'Lnfad0_di', 'Lnfad8_di', 'FCfad_crp']]


df_subset_nut = df[['MOMID', 
    'fad0_ENERGY_KCAL', 'fad0_ENERGY_J', 'fad0_PROTEIN', 'fad0_PROTEIN_Epros', 'fad0_CARBOHYDRATES', 
    'fad0_CARBOHYDRATES_Epros', 'fad0_ALCOHOL', 'fad0_ALCOHOL_Epros', 'fad0_ASH', 'fad0_FAT', 'fad0_FAT_Epros', 
    'fad0_FATTRI', 'fad0_SFA', 'fad0_SFA_Epros', 'fad0_MUFA', 'fad0_MUFA_Epros', 'fad0_PUFA', 'fad0_PUFA_Epros', 
    'fad0_N3_FA', 'fad0_N6_FA', 'fad0_CHOLESTEROL', 'fad0_F16P0T', 'fad0_FA18', 'fad0_F18P1T', 'fad0_ALFALINOLENICACID', 
    'fad0_LINOLEICACID', 'fad0_EPA', 'fad0_DHA', 'fad0_F20D4N6', 'fad0_FATRN', 'fad0_STERT', 'fad0_LACS', 'fad0_SUCROSE', 
    'fad0_SUGAR', 'fad0_STARCH', 'fad0_FIBER', 'fad0_FRUS', 'fad0_FIBINS', 'fad0_PSACNCS', 'fad0_CHOCDF', 'fad0_A_VITAMIN', 
    'fad0_D_VITAMIN', 'fad0_E_VITAMIN', 'fad0_K_VITAMIN', 'fad0_C_VITAMIN', 'fad0_THIAMIN_B1', 'fad0_RIBOFLAVIN_B2', 
    'fad0_NIASIN_EQUIVALENT', 'fad0_PYRIDOXIN_B6', 'fad0_KOBALAMIN_B12', 'fad0_CAROTENOIDS', 'fad0_BETACAROTENS', 
    'fad0_RETINOLI', 'fad0_FOLATE', 'fad0_NATRITUM', 'fad0_MAGNESIUM', 'fad0_CALSIUM', 'fad0_IRON', 'fad0_POTASSIUM', 
    'fad0_CR', 'fad0_CU', 'fad0_FD', 'fad0_IODINE', 'fad0_MN', 'fad0_MO', 'fad0_NATRIUMCHLORIDE', 'fad0_NT', 'fad0_PHOSPHORIUM', 
    'fad0_SELENIUM', 'fad0_ZINC', 'fad4_DaDate', 'fad4_ENERGY_KCAL', 'fad4_ENERGY_J', 'fad4_PROTEIN', 'fad4_PROTEIN_Epros', 
    'fad4_CARBOHYDRATE', 'fad4_CARBOHYDRATE_Epros', 'fad4_ALCOHOL', 'fad4_ALCOHOL_Epros', 'fad4_ASH', 'fad4_FAT', 
    'fad4_FAT_Epros', 'fad4_FATTRI', 'fad4_SFA', 'fad4_SFA_Epros', 'fad4_MUFA', 'fad4_MUFA_Epros', 'fad4_PUFA', 'fad4_PUFA_Epros', 
    'fad4_N3_FA', 'fad4_N6_FA', 'fad4_CHOLESTEROL', 'fad4_F16P0T', 'fad4_FA18', 'fad4_F18P1T', 'fad4_ALFALINOLENICACID', 
    'fad4_LINOLEICACID', 'fad4_EPA', 'fad4_DHA', 'fad4_F20D4N6', 'fad4_FATRN', 'fad4_STERT', 'fad4_LACS', 'fad4_SUCROSE', 
    'fad4_SUGAR', 'fad4_STARCH', 'fad4_FIBER', 'fad4_FRUS', 'fad4_FIBINS', 'fad4_PSACNCS', 'fad4_CHOCDF', 'fad4_A_VITAMIN', 
    'fad4_D_VITAMIN', 'fad4_E_VITAMIN', 'fad4_K_VITAMIN', 'fad4_C_VITAMIN', 'fad4_THIAMIN_B1', 'fad4_RIBOFLAVIN_B2', 
    'fad4_NIASIN_EQUIVALENT', 'fad4_PYRIDOXIN_B6', 'fad4_KOBALAMIN_B12', 'fad4_CAROTENOIDS', 'fad4_BETACAROTENS', 
    'fad4_RETINOLI', 'fad4_FOLATE', 'fad4_NATRITUM', 'fad4_MAGNESIUM', 'fad4_CALSIUM', 'fad4_IRON', 'fad4_POTASSIUM', 
    'fad4_CR', 'fad4_CU', 'fad4_FD', 'fad4_IODINE', 'fad4_MN', 'fad4_MO', 'fad4_NATRIUMCHLORIDE', 'fad4_NT', 'fad4_PHOSPHORIUM', 
    'fad4_SELENIUM', 'fad4_ZINC', 'fad8_DaDate', 'fad8_ENERGY_KCAL', 'fad8_ENERGY_J', 'fad8_PROTEIN', 'fad8_PROT_Epros', 
    'fad8_CARBOHYDRATES', 'fad8_CARBOHYDRATES_Epros', 'fad8_ALCOHOL', 'fad8_ALCOHOL_Epros', 'fad8_ASH', 'fad8_FAT', 
    'fad8_FAT_Epros', 'fad8_FATTRI', 'fad8_SFA', 'fad8_SFA_Epros', 'fad8_MUFA', 'fad8_MUFA_Epros', 'fad8_PUFA', 'fad8_PUFA_Epros', 
    'fad8_N3_FA', 'fad8_N6_FA', 'fad8_CHOLESTEROL', 'fad8_F16P0T', 'fad8_FA18', 'fad8_F18P1T', 'fad8_ALFALINOLENICACID', 
    'fad8_LINOLEICACID', 'fad8_EPA', 'fad8_DHA', 'fad8_F20D4N6', 'fad8_FATRN', 'fad8_STERT', 'fad8_LACS', 'fad8_SUCROSE', 
    'fad8_SUGAR', 'fad8_STARCH', 'fad8_FIBER', 'fad8_FRUS', 'fad8_FIBINS', 'fad8_PSACNCS', 'fad8_CHOCDF', 'fad8_A_VITAMIN', 
    'fad8_D_VITAMIN', 'fad8_E_VITAMIN', 'fad8_K_VITAMIN', 'fad8_C_VITAMIN', 'fad8_THIAMIN_B1', 'fad8_RIBOFLAVIN_B2', 
    'fad8_NIASIN_EQUIVALENT', 'fad8_PYRIDOXIN_B6', 'fad8_KOBALAMIN_B12', 'fad8_CAROTENOIDS', 'fad8_BETACAROTENS', 
    'fad8_RETINOLI', 'fad8_FOLATE', 'fad8_NATRIUM', 'fad8_MAGNESIUM', 'fad8_CALSIUM', 'fad8_IRON', 'fad8_POTASSIUM', 
    'fad8_CR', 'fad8_CU', 'fad8_FD', 'fad8_IODINE', 'fad8_MN', 'fad8_MO', 'fad8_NATRIUMCHLORIDE', 'fad8_NT', 'fad8_PHOSPHORIUM', 
    'fad8_SELENIUM', 'fad8_ZINC', 'OIL_CONS_AIMED', 'OIL_CONS_AIMED_cat', 'OIL_CONS_REAL', 'fadinterv_ENERGY_KCAL', 
    'fadinterv_PROTEIN_Epros', 'fadinterv_CARBOHYDRATES_Epros', 'fadinterv_FAT_Epros', 'fadinterv_SFA_Epros', 
    'fadinterv_MUFA_Epros', 'fadinterv_PUFA_Epros', 'fadinterv_FIBER', 'fadinterv_SUCROSE', 'fadinterv_ALFALINOLENICACID',
    'fadinterv_LINOLEICACID', 'fadinterv_CHOLESTEROL', 'fad0_SUCROSE_epros', 'fad4_SUCROSE_epros', 'fad8_SUCROSE_epros', 
    'fadinterv_SUCROSE_epros', 'lg_fad0_ENERGY_KCAL', 'lg_fadinterv_ENERGY_KCAL', 'lg_fad0_SFA_Epros', 'lg_fadinterv_SFA_Epros', 
    'lg_fad0_MUFA_Epros', 'lg_fadinterv_MUFA_Epros', 'lg_fad0_PUFA_Epros', 'lg_fadinterv_PUFA_Epros', 'lg_fad0_PROTEIN_Epros', 
    'lg_fadinterv_PROTEIN_Epros', 'lg_fad0_CHOLESTEROL', 'lg_fadinterv_CHOLESTEROL', 'lg_fad0_ALFALINOLENICACID', 
    'lg_fadinterv_ALFALINOLENICACID', 'lg_fad0_LINOLEICACID', 'lg_fadinterv_LINOLEICACID', 'lg_fad0_FIBER', 'lg_fadinterv_FIBER', 
    'FOLD_ENERGY_KCAL', 'FOLD_PROTEIN_Epros', 'FOLD_CARBOHYDRATES_Epros', 'FOLD_FAT_Epros', 'FOLD_SFA_Epros', 'FOLD_MUFA_Epros', 
    'FOLD_PUFA_Epros', 'FOLD_CHOLESTEROL', 'FOLD_ALFALINOLENICACID', 'FOLD_LINOLEICACID', 'FOLD_SUCROSE_Epros', 'FOLD_FIBER']]


correlation_matrix = df_subset_nut.corr()

plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix: Nutritional Biomarkers')
plt.show()