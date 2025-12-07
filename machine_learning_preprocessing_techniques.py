import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("ai_usage_in_studlife.csv")
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

x = df.drop(["SatisfactionRating", "SessionID", "SessionDate"], axis=1)
y = df["SatisfactionRating"]

cat_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = x.select_dtypes(include=["int", "float"]).columns.tolist()

simp1 = SimpleImputer(strategy='mean')
x[num_cols] = simp1.fit_transform(x[num_cols])
simp2 = SimpleImputer(strategy='most_frequent')
x[cat_cols] = simp2.fit_transform(x[cat_cols])

onehot = OneHotEncoder(sparse_output=False, max_categories=2)
new = onehot.fit_transform(x[cat_cols])
new_col = onehot.get_feature_names_out(cat_cols)
x_one = pd.DataFrame(new, columns=new_col)
x_two = x.drop(columns=cat_cols)
final = pd.concat([x_one, x_two], axis=1)

x_train, x_test, y_train, y_test = train_test_split(final,y,test_size=0.2, random_state=0)

minmax = MinMaxScaler()
minmax.fit(x_train)
x_train = minmax.transform(x_train)
x_test = minmax.transform(x_test)

print(final.columns)
print(final.select_dtypes(include=["object"]).head())

log = LinearRegression()
model = log.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)

r2 = ("R2_score:", r2_score(y_test,pred))
print(r2)



