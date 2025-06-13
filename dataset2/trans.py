import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

# 读取文件（假设文件名为：filtered_herbs_with_ingredients.csv）
df = pd.read_csv('filtered_herbs_with_ingredients.csv', header=None, dtype=str)

# 转换函数：使用 PubChem 查找成分名对应的 SMILES
def get_smiles(name):
    try:
        compound = pcp.get_compounds(name, 'name')
        if compound:
            return compound[0].canonical_smiles
    except:
        pass
    return None

# 创建新的 DataFrame 存储结果
new_data = []

# 遍历每一行（每个中药及其成分）
for index, row in tqdm(df.iterrows(), total=len(df), desc="Converting to SMILES"):
    herb_name = row[0]
    ingredients = row[1:].dropna().unique()
    smiles_list = [get_smiles(ing) for ing in ingredients]
    new_data.append([herb_name] + smiles_list)

# 保存为新CSV
output_df = pd.DataFrame(new_data)
output_df.to_csv('herb_ingredients_smiles.csv', index=False, header=False, encoding='utf-8-sig')
