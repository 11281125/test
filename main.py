import re
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus']=False

train_data=pd.read_csv(r'C:\Users\DYL\Desktop\titanic\train.csv')

# 数据信息总览
print(train_data.info())
print(train_data.describe())
print("-" * 40)

#存活的比例
plt.pie(x=train_data['Survived'].value_counts(), labels=['Survived_0', 'Survived_1'], autopct='%.1f%%')
plt.show()

# 缺失值的处理
# 1、Embarked这一属性（共有三个上船地点），缺失俩值，可以用众数赋值
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
# 2、对于标称属性，可以赋一个代表缺失的值，比如‘U’。船舱号Cabin这一属性，缺失可能代表并没有船舱。
train_data['Cabin'] = train_data.Cabin.fillna('U')
# 3、Age在该数据集里是一个相当重要的特征，使用回归 随机森林等模型来预测缺失属性的值
from sklearn.ensemble import RandomForestRegressor
age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]

RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])
train_data.loc[train_data['Age'].isnull(), ['Age']]= predictAges
print(train_data.info())
print("-" * 40)

# 分析数据关系
# 1、性别与是否生存的关系
print(train_data.groupby(['Sex','Survived'])['Survived'].count())
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
plt.show()
print("-" * 40)

# 2、船舱等级与是否生存的关系
print(train_data.groupby(['Pclass','Survived'])['Pclass'].count())
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
plt.show()
print("-" * 40)
# 不同等级船舱男女的生存率
print(train_data.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count())
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
plt.show()
print("-" * 40)

#3、年龄与是否生存的关系
# 不同等级船舱的年龄分布和生存的关系
fig, ax = plt.subplots(1, 2, figsize = (18, 8))
sns.violinplot("Pclass", "Age", hue="Survived", data=train_data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
# 不同性别下的年龄分布和生存的关系
sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
plt.show()
# 总体的年龄分布
plt.figure(figsize=(12,5))
plt.subplot(121)
train_data['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
train_data.boxplot(column='Age', showfliers=False)
plt.show()
# 不同年龄下的生存和非生存的分布情况：
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.show()
# 不同年龄下的平均生存率
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Age_int'] = train_data['Age'].astype(int)
average_age = train_data[['Age_int', 'Survived']].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x='Age_int', y='Survived', data=average_age)
plt.show()

print(train_data['Age'].describe())
# 按照年龄，将乘客划分为儿童、少年、成年和老年，分析四个群体的生还情况
bins = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'], bins)
by_age = train_data.groupby('Age_group')['Survived'].mean()
print(by_age)

by_age.plot(kind = 'bar')
plt.show()
# 4、称呼与存活与否的关系
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(pd.crosstab(train_data['Title'], train_data['Sex']))
# 不同称呼与生存率的关系
train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()
plt.show()
# 名字长度和生存率之间的关系
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Name_length'] = train_data['Name'].apply(len)
name_length = train_data[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()
sns.barplot(x='Name_length', y='Survived', data=name_length)
plt.show()
# 5、有无兄弟姐妹和存活与否的关系
sibsp_df = train_data[train_data['SibSp'] != 0]
no_sibsp_df = train_data[train_data['SibSp'] == 0]

plt.figure(figsize=(10,5))
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('sibsp')

plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_sibsp')
plt.show()

train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar()
plt.title('SibSp and Survived')
plt.show()
# 6、有无父母子女和存活与否的关系
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]

plt.figure(figsize=(10,5))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('parch')

plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_parch')
plt.show()

train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar()
plt.title('Parch and Survived')
plt.show()
# 7、亲友的人数和存活与否的关系
train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
train_data[['Family_Size','Survived']].groupby(['Family_Size']).mean().plot.bar()
plt.show()
# 8、票价分布和存活与否的关系
plt.figure(figsize=(10,5))
train_data['Fare'].hist(bins = 70)

train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
plt.show()

print(train_data['Fare'].describe())
# 生存与否与票价均值和方差的关系
fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]
average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
average_fare.plot(yerr=std_fare, kind='bar', legend=False)
plt.show()
# 9、 船舱类型和存活与否的关系
# 是否有船舱类型与生存与否的关系
train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U'
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'U' else 1)
train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()
plt.show()
# 不同类型的船舱与生存与否的关系
train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()
plt.show()
# 10、港口和存活与否的关系
sns.countplot('Embarked', hue='Survived', data=train_data)
plt.title('Embarked and Survived')
plt.show()
sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)
plt.title('Embarked and Survived rate')
plt.show()

# 对数据采用独热编码（one-hot Encoding）并拼接在原来的"train_data"之上
dummies_Cabin = pd.get_dummies(train_data['Has_Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(train_data['Embarked'], prefix= 'Embarked')
dummies_Title = pd.get_dummies(train_data['Title'], prefix= 'Title')
dummies_Sex = pd.get_dummies(train_data['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(train_data['Pclass'], prefix= 'Pclass')
df = pd.concat([train_data, dummies_Cabin, dummies_Embarked,dummies_Title, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age_int','Age_group','Title','Name_length','Family_Size','Has_Cabin','CabinLetter'], axis=1, inplace=True)

# 对Age、Fare进行Scaling
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(np.array(df['Age']).reshape((len(df['Age']), 1)))
df['Age_scaled'] = scaler.fit_transform(np.array(df['Age']).reshape((len(df['Age']), 1)), age_scale_param)
fare_scale_param = scaler.fit(np.array(df['Fare']).reshape((len(df['Fare']), 1)))
df['Fare_scaled'] = scaler.fit_transform(np.array(df['Fare']).reshape((len(df['Fare']), 1)), fare_scale_param)

# #使用逻辑回归进行分类
from sklearn import linear_model
#用正则表达式取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values
train_y = train_np[:, 0]    #Survival结果：第0列
train_X = train_np[:, 1:]   #特征属性：第1列及后序列
#fit到LogisticRegression之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6,solver='liblinear')
clf.fit(train_X, train_y)
print(clf)


3.

#测试数据预测
test_data=pd.read_csv(r'C:\Users\DYL\Desktop\titanic\test.csv')
print(test_data.info())
print(test_data.describe())
#缺失值处理(测试集缺失Age、Cabin、Fare特征)
test_data.loc[ (test_data.Fare.isnull()), 'Fare' ] = 0
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data.loc[test_data.Cabin.isnull(), 'Cabin'] = 'U'
test_data['Has_Cabin'] = test_data['Cabin'].apply(lambda x: 0 if x == 'U' else 1)
# test_data数据预处理 用回归 随机森林等模型来预测缺失属性age的值
from sklearn.ensemble import RandomForestRegressor
age_df2 = test_data[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df2_notnull = age_df2.loc[(test_data['Age'].notnull())]
age_df2_isnull = age_df2.loc[(test_data['Age'].isnull())]
X = age_df2_notnull.values[:,1:]
Y = age_df2_notnull.values[:,0]
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df2_isnull.values[:,1:])
test_data.loc[test_data['Age'].isnull(), ['Age']]= predictAges

# 对数据采用独热编码（one-hot Encoding）并拼接在原来的"test_data"之上
dummies_Cabin = pd.get_dummies(test_data['Has_Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(test_data['Embarked'], prefix= 'Embarked')
dummies_Title = pd.get_dummies(test_data['Title'], prefix= 'Title')
dummies_Sex = pd.get_dummies(test_data['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(test_data['Pclass'], prefix= 'Pclass')
df_test = pd.concat([test_data, dummies_Cabin, dummies_Embarked,dummies_Title, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Has_Cabin','Title'], axis=1, inplace=True)

# # 对Age、Fare进行Scaling
df_test['Age_scaled'] = scaler.fit_transform(np.array(df_test['Age']).reshape((len(df_test['Age']), 1)), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(np.array(df_test['Fare']).reshape((len(df_test['Fare']), 1)), fare_scale_param)

# 预测结果
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':test_data['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv(r'C:\Users\DYL\Desktop\logistic_regression_predictions.csv', index=False)



