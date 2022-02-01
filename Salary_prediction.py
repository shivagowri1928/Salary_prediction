#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[47]:


df = pd.read_csv("survey_results_public.csv")


# In[48]:


df.head(10)


# In[49]:


df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]
df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
df.head()


# In[50]:


df = df[df["Salary"].notnull()]
df.head()


# In[51]:


df.info()


# In[52]:


df = df.dropna()
df.isnull().sum()


# In[53]:


df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()


# In[54]:


df['Country'].value_counts()


# In[55]:


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


# In[56]:


country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()


# In[57]:


df = df[df["Salary"] <= 250000]
df = df[df["Salary"] >= 10000]
df = df[df["Country"] != "Other"]


# In[58]:


fig , ax = plt.subplots(1,1, figsize=(12,7))
df.boxplot('Salary', "Country", ax = ax)
plt.title(" ")
plt.suptitle('Salary(US$) vs Country')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[59]:


df["YearsCodePro"].unique()


# In[60]:


def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)    
df.head(20)


# In[61]:


df["EdLevel"].unique()


# In[62]:


def clean_education(x):
    if "Bachelor’s degree" in x:
        return 'Bachelor’s degree'
    if "Master’s degree" in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral degree' in x:
        return 'Post grad'
    return "Less than a bachelor's"

df["EdLevel"] = df["EdLevel"].apply(clean_education)
df["EdLevel"].unique()


# In[63]:


from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df["EdLevel"] = le_education.fit_transform(df["EdLevel"])
df["EdLevel"].unique()


# In[64]:


le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df["Country"].unique()


# In[65]:


X = df.drop("Salary", axis=1)
Y = df["Salary"]
print(Y.values)


# ## Using Linear Regression algo

# In[66]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, Y.values)


# In[67]:


y_pred = linear_reg.predict(X)


# In[24]:


from sklearn.metrics import mean_squared_error , mean_absolute_error
error = np.sqrt(mean_squared_error(Y,y_pred))


# In[25]:


error


# ## Using DecisionTree

# In[26]:


from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, Y.values)


# In[27]:


y_pred = dec_tree_reg.predict(X)


# In[28]:


error = np.sqrt(mean_squared_error(Y, y_pred))
print("${:,.02f}".format(error))


# ## Using RandomForest

# In[29]:


from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X,Y.values)


# In[30]:


y_pred = random_forest_reg.predict(X)


# In[31]:


error = np.sqrt(mean_squared_error(Y, y_pred))
print("${:,.02f}".format(error))


# In[32]:


from sklearn.model_selection import GridSearchCV

max_depth = [None , 2,4,6,8,10,12]
parameters = {"max_depth" : max_depth}
regressor = DecisionTreeRegressor(random_state = 0)
gs = GridSearchCV(regressor ,parameters , scoring="neg_mean_squared_error")
gs.fit(X,Y.values)


# In[33]:


regressor = gs.best_estimator_

regressor.fit(X,Y.values)
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(Y, y_pred))
print("${:,.02f}".format(error))


# In[34]:


X


# In[35]:


X = np.array([["Italy", "Bachelor’s degree", 16]])
X


# In[36]:


X[:,0] = le_country.transform(X[:,0])
X[:,1] = le_education.transform(X[:,1])
X = X.astype(float)
X


# In[37]:


y_pred = regressor.predict(X)
y_pred


# In[42]:


import pickle
from pickle import dump


# In[43]:


data = {"model" : regressor, "le_country" : le_country, "le_education" : le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)
    


# In[44]:


with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


# In[45]:


y_pred = regressor_loaded.predict(X)
y_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




