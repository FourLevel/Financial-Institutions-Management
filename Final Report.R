# 讀取必要的 R 套件
library(readr)
library(dplyr)
library(car)  # 用於 VIF 計算
library(lmtest)  # 用於 Breusch-Pagan 檢定
library(plm)  # 用於 Panel Data 分析
library(Formula)

# 讀取檔案
data <- read.csv("/Users/ning/Asher/Financial Institution/銀行資料_丟迴歸分析_20240523.csv")

# 轉換變數名稱以符合 R 的命名慣例
colnames(data) <- c("id", "company_name", "ym", "time", "establishment_year", "roa", "s",
                    "equity_to_assets_100", "roa_std", "z_score", "total_assets", "ln_total_assets",
                    "net_income", "ln_net_income", "pb_ratio", "debt_ratio", "company_age",
                    "board_ownership", "manager_ownership", "financial_crisis", "covid_period")

# 確保 financial_crisis 和 covid_period 為 factor 型態，以作為 dummy variables
data$financial_crisis <- as.factor(data$financial_crisis)
data$covid_period <- as.factor(data$covid_period)

# 標準化
data$standardized_total_assets <- scale(data$total_assets)
data$standardized_net_income <- scale(data$net_income)



# (1) 進行 VIF 檢定
# 建置線性迴歸模型
model <- lm(z_score ~ standardized_total_assets + ln_net_income + pb_ratio + debt_ratio + company_age +
              board_ownership + manager_ownership + financial_crisis + covid_period, data = data)

# 計算 VIF
vif_values <- vif(model)
print(vif_values)
summary(model)

# 解讀 VIF 檢定結果：
# VIF 值小於 10：表明變數之間的多重共線性問題不嚴重，可以保留在模型中。
# VIF 值大於 10：表明變數之間存在嚴重的多重共線性，可能需要考慮移除該變數或使用其他方法處理共線性問題。



# Model 1-----------------------------------------------------------------------
# model 1 直接選取原始 model 中顯著之變數，進行 Breusch-Pagan 檢定
model_1 <- lm(z_score ~ standardized_total_assets + ln_net_income + company_age +
                manager_ownership + financial_crisis, data = data)
summary(model_1)
bp_test_1 <- bptest(model_1)
print(bp_test_1)

# 解讀結果：
# BP：這是 Breusch-Pagan 檢定的檢定統計量。它衡量的是模型中異方差性的程度。該值本身並沒有一個直接的解釋，主要是與 p 值結合起來使用。
# p-value：這是與檢定統計量相關聯的 p 值。它表示在零假設（即沒有異方差性）的情況下，檢定統計量比觀測到的值更極端的概率。通常使用 0.05 作為顯著性水平。如果 p 值小於 0.05，則可以拒絕零假設，認為存在異方差性。


# model 1 進行 Hausman 檢定
# 將資料轉換為 Panel Data 格式
pdata <- pdata.frame(data, index = c("id", "time"))

# 建置固定效應模型
fe_model_1 <- plm(z_score ~ standardized_total_assets + ln_net_income + company_age +
                    manager_ownership + financial_crisis, data = pdata, model = "within")

# 建置隨機效應模型
re_model_1 <- plm(z_score ~ standardized_total_assets + ln_net_income + company_age +
                    manager_ownership + financial_crisis, data = pdata, model = "random")

# 進行 Hausman 檢定
hausman_test_1 <- phtest(fe_model_1, re_model_1)
print(hausman_test_1)

# 解讀結果：
# Hausman test statistic：這是檢定統計量。該值越大，表示固定效應模型和隨機效應模型的估計結果之間差異越大。
# Degrees of freedom：這是檢定統計量的自由度，通常等於參數的個數。
# p-value：這是與檢定統計量相關聯的 p 值。如果 p 值小於某個顯著性水平（0.05），則表示資料適合使用固定效應模型。


# model 1 進行固定效應模型分析
summary(fe_model_1)
#-------------------------------------------------------------------------------




# Model 2-----------------------------------------------------------------------
# model 2 以 model 1 為基礎，加上 Debt Ratio 作為變數，進行 Breusch-Pagan 檢定
model_2 <- lm(z_score ~ standardized_total_assets + ln_net_income + debt_ratio +
                company_age + manager_ownership + financial_crisis, data = data)
summary(model_2)
bp_test_2 <- bptest(model_2)
print(bp_test_2)


# model 2 進行 Hausman 檢定
# 將資料轉換為 Panel Data 格式
pdata <- pdata.frame(data, index = c("id", "time"))

# 建置固定效應模型
fe_model_2 <- plm(z_score ~ standardized_total_assets + ln_net_income + debt_ratio +
                    company_age + manager_ownership + financial_crisis, data = pdata, model = "within")

# 建置隨機效應模型
re_model_2 <- plm(z_score ~ standardized_total_assets + ln_net_income + debt_ratio +
                    company_age + manager_ownership + financial_crisis, data = pdata, model = "random")

# 進行 Hausman 檢定
hausman_test_2 <- phtest(fe_model_2, re_model_2)
print(hausman_test_2)


# model 2 進行固定效應模型分析
summary(fe_model_2)
#-------------------------------------------------------------------------------




# Model 3-----------------------------------------------------------------------
# model 3 以 model 2 為基礎，刪除不顯著之 ln(Net Income)，換為 P/B Ratio 作為變數，進行 Breusch-Pagan 檢定
model_3 <- lm(z_score ~ standardized_total_assets + pb_ratio + debt_ratio +
                company_age + manager_ownership + financial_crisis, data = data)
summary(model_3)
bp_test_3 <- bptest(model_3)
print(bp_test_3)


# model 3 進行 Hausman 檢定
# 將資料轉換為 Panel Data 格式
pdata <- pdata.frame(data, index = c("id", "time"))

# 建置固定效應模型
fe_model_3 <- plm(z_score ~ standardized_total_assets + pb_ratio + debt_ratio +
                    company_age + manager_ownership + financial_crisis, data = pdata, model = "within")

# 建置隨機效應模型
re_model_3 <- plm(z_score ~ standardized_total_assets + pb_ratio + debt_ratio +
                    company_age + manager_ownership + financial_crisis, data = pdata, model = "random")

# 進行 Hausman 檢定
hausman_test_3 <- phtest(fe_model_3, re_model_3)
print(hausman_test_3)


# model 3 進行固定效應模型分析
summary(fe_model_3)
#-------------------------------------------------------------------------------