import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel import PanelOLS, RandomEffects

# 讀取 Excel 檔案
file_path = '銀行資料_丟迴歸分析_20240523.csv'
data = pd.read_csv(file_path)

# 選擇變數並重命名
df = data[['公司代碼', '年', 'Z-score', 'ln資產總額', 'ln淨收益', '當季季底P/B', 
           '負債比率', '公司年齡', '董事總持股數%', '經理人總持股%', 
           '是否在金融危機時期', '是否在COVID時期']]
df.columns = ['id', 'time', 'y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']

# 設置 Panel Data
df = df.set_index(['id', 'time'])



''' 進行 VIF 檢定 '''
# 準備自變數矩陣
X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']]
X['const'] = 1  # 增加常數項

# 計算VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)

''' 
解讀 VIF 檢定結果
VIF 值小於 10：表明變數之間的多重共線性問題不嚴重，可以保留在模型中。
VIF 值大於 10：表明變數之間存在嚴重的多重共線性，可能需要考慮移除該變數或使用其他方法處理共線性問題。
'''



''' 進行 Breusch-Pagan 檢定 '''
# 設置自變數矩陣
X = sm.add_constant(df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']])

# 設置應變數
y = df['y']

# 擬合回歸模型
model = sm.OLS(y, X).fit()

# 進行 Breusch-Pagan 檢定
bp_test = het_breuschpagan(model.resid, model.model.exog)

labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
bp_result = dict(zip(labels, bp_test))

# 打印 Breusch-Pagan 檢定結果
print("\nBreusch-Pagan Test:")
for key, value in bp_result.items():
    print(f"  {key}: {value}")

'''
解讀結果
Lagrange multiplier statistic：這是 Breusch-Pagan 檢定的檢定統計量。它衡量的是模型中異方差性的程度。該值本身並沒有一個直接的解釋，主要是與 p 值結合起來使用。
p-value：這是與檢定統計量相關聯的 p 值。它表示在零假設（即沒有異方差性）的情況下，檢定統計量比觀測到的值更極端的概率。通常使用 0.05 作為顯著性水平。如果 p 值小於 0.05，則可以拒絕零假設，認為存在異方差性。
f-value：這是基於 F 檢定的統計量，衡量的是異方差性的程度。和 Lagrange multiplier statistic 一樣，該值本身主要是與其對應的 p 值結合使用。
f p-value：這是與 f-value 對應的 p 值。它與 Lagrange multiplier statistic 的 p 值解釋類似。
'''



''' 進行 Hausman 檢定 '''
# 固定效應模型
fixed_effects_model = PanelOLS.from_formula('y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + EntityEffects', data=df)
fixed_effects_results = fixed_effects_model.fit()

# 隨機效應模型
random_effects_model = RandomEffects.from_formula('y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9', data=df)
random_effects_results = random_effects_model.fit()

# 進行 Hausman 檢定
def hausman(fe, re):
    b_diff = fe.params - re.params
    b_diff_cov = fe.cov - re.cov
    chi2 = np.dot(np.dot(b_diff.T, np.linalg.inv(b_diff_cov)), b_diff)
    df = b_diff.size
    p_value = stats.chi2.sf(chi2, df)
    return chi2, df, p_value

chi2, df, p_value = hausman(fixed_effects_results, random_effects_results)

hausman_result = {
    'Hausman test statistic': chi2,
    'Degrees of freedom': df,
    'p-value': p_value,
    'Fixed effects coefficients': fixed_effects_results.params,
    'Random effects coefficients': random_effects_results.params
}

# 顯示 Hausman 檢定結果
print(f"Hausman test statistic: {hausman_result['Hausman test statistic']}")
print(f"Degrees of freedom: {hausman_result['Degrees of freedom']}")
print(f"p-value: {hausman_result['p-value']}")

print("\nFixed effects coefficients:")
for key, value in hausman_result['Fixed effects coefficients'].items():
    print(f"  {key}: {value}")

print("\nRandom effects coefficients:")
for key, value in hausman_result['Random effects coefficients'].items():
    print(f"  {key}: {value}")

'''
解讀結果
Hausman test statistic：這是檢定統計量。該值越大，表示固定效應模型和隨機效應模型的估計結果之間差異越大。
Degrees of freedom：這是檢定統計量的自由度，通常等於參數的個數。
p-value：這是與檢定統計量相關聯的 p 值。如果 p 值小於某個顯著性水平（0.05），則表示資料適合使用固定效應模型。
Fixed effects coefficients：這是固定效應模型的估計係數。
Random effects coefficients：這是隨機效應模型的估計係數。
'''



''' 進行隨機效應模型分析 '''
# 讀取上傳的 CSV 檔案
file_path = '銀行資料_丟迴歸分析_20240523.csv'
data = pd.read_csv(file_path)

# 選擇變數並重命名
df = data[['公司代碼', '年', 'Z-score', 'ln資產總額', 'ln淨收益', '當季季底P/B', '負債比率', '公司年齡', '董事總持股數%', '經理人總持股%', '是否在金融危機時期', '是否在COVID時期']]
df.columns = ['id', 'time', 'y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']

# 設置 Panel Data
df = df.set_index(['id', 'time'])

# 建立隨機效應模型
random_effects_model = RandomEffects.from_formula('y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9', data=df)
random_effects_results = random_effects_model.fit()

# 顯示模型摘要
print(random_effects_results.summary)



'''
# 進行固定效應模型分析
# 讀取上傳的 CSV 檔案
file_path = '銀行資料_丟迴歸分析_20240523.csv'
data = pd.read_csv(file_path)

# 選擇變數並重命名
df = data[['公司代碼', '年', 'Z-score', 'ln資產總額', 'ln淨收益', '當季季底P/B', '負債比率', '公司年齡', '董事總持股數%', '經理人總持股%', '是否在金融危機時期', '是否在COVID時期']]
df.columns = ['id', 'time', 'y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']

# 設置 Panel Data
df = df.set_index(['id', 'time'])

# 建置固定效應模型
fixed_effects_model = PanelOLS.from_formula('y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + EntityEffects', data=df)
fixed_effects_results = fixed_effects_model.fit()

# 顯示結果摘要
print(fixed_effects_results.summary)
'''