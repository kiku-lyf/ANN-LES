import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("endijtrain_011_1w.xlsx")

# 保留不包含零值的行
df_filtered = df[~(df == 0).any(axis=1)]

# 保存到新的 Excel 文件
df_filtered.to_excel("end1w.xlsx", index=False)

