

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor,gradient_boosting,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

```

## Problem Description
Your goal is to predict the total_cases label for each (city, year, weekofyear) in the test set. There are two cities, San Juan and Iquitos, with test data for each city spanning 5 and 3 years respectively. You will make one submission that contains predictions for both cities. The data for each city have been concatenated along with a city column indicating the source: sj for San Juan and iq for Iquitos. The test set is a pure future hold-out, meaning the test data are sequential and non-overlapping with any of the training data. Throughout, missing values have been filled as NaNs.

For more Detail : https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/82/#features_list


### load the data Set


```python
train=pd.read_csv("dataset/dengue_features_train.csv")
label=pd.read_csv("dataset/dengue_labels_train.csv")
test=pd.read_csv("dataset/dengue_features_test.csv")
```


```python
train_df = train.copy()
label_df = label.copy()
```


```python
print("train shape",train_df.shape)
print("label shape",label_df.shape)
```

    ('train shape', (1456, 24))
    ('label shape', (1456, 4))
    


```python
df = pd.merge(train_df, label, how='left', on=['city','year', 'weekofyear'])
df.shape
```




    (1456, 25)




```python
df.columns
```




    Index([u'city', u'year', u'weekofyear', u'week_start_date', u'ndvi_ne',
           u'ndvi_nw', u'ndvi_se', u'ndvi_sw', u'precipitation_amt_mm',
           u'reanalysis_air_temp_k', u'reanalysis_avg_temp_k',
           u'reanalysis_dew_point_temp_k', u'reanalysis_max_air_temp_k',
           u'reanalysis_min_air_temp_k', u'reanalysis_precip_amt_kg_per_m2',
           u'reanalysis_relative_humidity_percent',
           u'reanalysis_sat_precip_amt_mm',
           u'reanalysis_specific_humidity_g_per_kg', u'reanalysis_tdtr_k',
           u'station_avg_temp_c', u'station_diur_temp_rng_c',
           u'station_max_temp_c', u'station_min_temp_c', u'station_precip_mm',
           u'total_cases'],
          dtype='object')



### Histogram of All Column


```python
df.hist(figsize=(20,16))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000000001325F710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000000135C3FD0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000000135270F0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000000001353FF98>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000000137525F8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000000137E22E8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000000137E2DA0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000000001398BB38>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000000138BCFD0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000013B33048>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000000013BB3C18>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000015D93898>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000015E63A58>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000015EDDC88>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000015F6E908>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000000015FFF518>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000016068470>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000015EF94A8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000000161D3978>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000000162B5B38>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000000163857B8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000000001645B898>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000000164955F8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000000165672E8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000000016567DA0>]],
          dtype=object)




![png](output_9_1.png)


### Remove Null value with mean


```python
df = df.fillna(df.mean())
```

### Lable Encoding for categorical variable


```python
from sklearn.preprocessing import LabelEncoder
```


```python
lb=LabelEncoder()
df['city']=lb.fit_transform(df['city'])
```


```python
df=df.drop(['week_start_date'],axis=1)
```


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>year</th>
      <th>weekofyear</th>
      <th>ndvi_ne</th>
      <th>ndvi_nw</th>
      <th>ndvi_se</th>
      <th>ndvi_sw</th>
      <th>precipitation_amt_mm</th>
      <th>reanalysis_air_temp_k</th>
      <th>reanalysis_avg_temp_k</th>
      <th>...</th>
      <th>reanalysis_relative_humidity_percent</th>
      <th>reanalysis_sat_precip_amt_mm</th>
      <th>reanalysis_specific_humidity_g_per_kg</th>
      <th>reanalysis_tdtr_k</th>
      <th>station_avg_temp_c</th>
      <th>station_diur_temp_rng_c</th>
      <th>station_max_temp_c</th>
      <th>station_min_temp_c</th>
      <th>station_precip_mm</th>
      <th>total_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1990</td>
      <td>18</td>
      <td>0.1226</td>
      <td>0.103725</td>
      <td>0.198483</td>
      <td>0.177617</td>
      <td>12.42</td>
      <td>297.572857</td>
      <td>297.742857</td>
      <td>...</td>
      <td>73.365714</td>
      <td>12.42</td>
      <td>14.012857</td>
      <td>2.628571</td>
      <td>25.442857</td>
      <td>6.900000</td>
      <td>29.4</td>
      <td>20.0</td>
      <td>16.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1990</td>
      <td>19</td>
      <td>0.1699</td>
      <td>0.142175</td>
      <td>0.162357</td>
      <td>0.155486</td>
      <td>22.82</td>
      <td>298.211429</td>
      <td>298.442857</td>
      <td>...</td>
      <td>77.368571</td>
      <td>22.82</td>
      <td>15.372857</td>
      <td>2.371429</td>
      <td>26.714286</td>
      <td>6.371429</td>
      <td>31.7</td>
      <td>22.2</td>
      <td>8.6</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 24 columns</p>
</div>



### Correlation Matrix for Feature Selection


```python
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr>0.10, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x169239b0>




![png](output_18_1.png)


### Split Train and Test Data 


```python
# df = df.drop(['year',u'ndvi_ne',
#        u'ndvi_nw', u'ndvi_se', u'ndvi_sw', u'precipitation_amt_mm','reanalysis_max_air_temp_k',u'reanalysis_precip_amt_kg_per_m2',
#        u'reanalysis_relative_humidity_percent',
#        u'reanalysis_sat_precip_amt_mm', u'station_diur_temp_rng_c',
#        u'station_max_temp_c', u'station_precip_mm',],axis=1)
label = u'total_cases'
a = list(df.columns)
a.remove(label)
labels = df[label]
data_only = df[list(a)]
col_name = data_only.columns
data_only = preprocessing.scale(data_only)
X_train, X_test, y_train, y_test = train_test_split(data_only, labels, test_size=0.1,random_state = 42)

```

### Best Regression Model for this Dataset


```python
 Rregression = [
        GradientBoostingRegressor(),
      ExtraTreesRegressor(max_features= 7),
     BaggingRegressor(),
     RandomForestRegressor(n_estimators=100,max_features=7,max_depth=5)
    ]
```

### Model load with mean and r2 accuracy


```python

for classifier in Rregression:
    try:
       
        fit = classifier.fit(X_train,y_train)
        pred = fit.predict(X_test)
    except Exception:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        fit = classifier.fit(X_train,y_train)
        pred = fit.predict(X_test)
        
    print "mean_error",classifier.__class__ ,mean_absolute_error(y_test, pred)
    print "r squre",classifier.__class__ , r2_score(y_test, pred)
   
```

    mean_error <class 'sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'> 15.25854508514302
    r squre <class 'sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'> 0.5891577555271912
    mean_error <class 'sklearn.ensemble.forest.ExtraTreesRegressor'> 17.947945205479453
    r squre <class 'sklearn.ensemble.forest.ExtraTreesRegressor'> 0.4029736284600949
    mean_error <class 'sklearn.ensemble.bagging.BaggingRegressor'> 15.584246575342464
    r squre <class 'sklearn.ensemble.bagging.BaggingRegressor'> 0.5968568738929435
    mean_error <class 'sklearn.ensemble.forest.RandomForestRegressor'> 16.907251551152942
    r squre <class 'sklearn.ensemble.forest.RandomForestRegressor'> 0.44683680223721345
    

### Best Model load for predict value


```python
clf = GradientBoostingRegressor()
fit = clf.fit(data_only,labels)

```

### Feature Important Insight of Model


```python
imp = fit.feature_importances_
d = {'name': col_name,'value':imp}
d = pd.DataFrame(data =d)
d = d.set_index('name')

```


```python
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

data = [go.Bar(
            x=imp,
            y=col_name,
            orientation = 'h'
)]
layout = go.Layout(
    hovermode= 'closest',
    autosize = True,)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<div id="29a774a0-91f2-4544-a144-116a8177becc" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("29a774a0-91f2-4544-a144-116a8177becc", [{"y": ["city", "year", "weekofyear", "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw", "precipitation_amt_mm", "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent", "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k", "station_avg_temp_c", "station_diur_temp_rng_c", "station_max_temp_c", "station_min_temp_c", "station_precip_mm"], "x": [0.0, 0.28294427946508716, 0.17056530277400203, 0.03555402434952732, 0.0157245828201161, 0.01724542591112792, 0.05079043170685516, 0.01621815274942272, 0.09447994623414747, 0.014296103997740315, 0.04422687497012974, 0.022067739924260525, 0.025625480462682287, 0.03091368089478887, 0.010718289473492253, 0.016087587508507063, 0.024160721723706365, 0.022772442506357838, 0.025138845383617082, 0.010382011004532723, 0.023329064172764688, 0.016910053852985427, 0.02984895811414888], "type": "bar", "orientation": "h"}], {"autosize": true, "hovermode": "closest"}, {"linkText": "Export to plot.ly", "showLink": true})});</script>


### Test Data prediction


```python
test['city']=lb.fit_transform(test['city'])
test=test.drop(['week_start_date'],axis=1)
test.shape

```




    (416, 23)




```python
test =test.fillna(test.mean())
```


```python
test1 = preprocessing.scale(test)
```


```python
predictions = fit.predict(test1)
p=predictions.astype('int')
```

### Prediction Value


```python
p
```




    array([ 29,  24,  24,  23,  27,  29,  29,  43,  25,  22,  16,  24,  29,
            34,  49,  79,  62,  78,  90,  81,  86,  76, 149, 105, 108, 100,
            90, 101, 126, 104,  97,  83,  81,  87,  70,  62,  56,  44,  54,
            52,  43,  45,  45,  43,  69,  34,  36,  38,  35,  39,  36,  38,
            38,  36,  37,  38,  51,  43,  37,  51,  77,  47,  64,  48,  72,
            66,  80,  76,  81, 138, 189, 197, 212, 245, 235, 245, 242, 222,
           249, 220, 222, 229, 238, 227, 202, 200, 142,  77,  65,  64,  57,
            65,  61,  60,  57,  52,  52,  47,  49,  52,  50,  49,  42,  45,
            46,  65,  49,  52,  50, 105,  86,  70,  45,  66,  80, 130,  70,
            57,  79,  93, 109, 121, 100, 130,  98,  94,  80,  86,  92, 107,
           106,  75,  61,  59,  68,  54,  73,  66,  52, -29,  13,  12,  13,
            13,  16,  12,  10,  10,   9,   8,   5,   4,   8,   1,   3,   4,
             4,   5,   4,   3,   6,   6,   7,  19,  30,   4,   9,  23,  20,
            41,  17,  34,  22,  24,  20,  19,  33,  15,  23,  27,  23,  23,
            21,  25,  22,  20,  22,  20,  17,  18,   5,   5,  19,  14,  14,
            15,  13,  13,  10,  12,   9,   9,   7,   6,   4,   4,   2,   5,
             5,   5,   4,   4,   4,   7,  22,   8,  44,  38,  12,  12,   8,
            15,  10,  18,  21,  20,  16,  26,  76,  24,  22,  32,  22,  22,
            25,  44,  38,  24,  21,  22,  17,  10,   7,  14,  18,  14,  16,
            13,  13,  12,  12,  11,   6,   7,   6,   4,   4,   5,   4,   3,
            55,  70,  54,  71,  73,  64,  99,  81,  78, 101,  62, 101,  88,
           101, 123,  91, 107, 106, 143,  90,  86,  87,  87,  89, 109, 103,
            10,  13,  12,  12,  13,  14,  11,   9,  10,   9,   6,   5,   6,
             7,   4,   5,   6,   5,   6,   4,   4,   4,   4,   4,   8,   4,
             5,   4,   3,   3,   3,   3,   2,   3,   2,   3,   3,   7,   5,
            14,  10,  10,  12,  11,  10,  10,  14,  12,  12,  13,  12,  10,
            10,  13,  14,  14,  12,  13,  12,  10,   9,   8,   8,   8,   4,
             9,   6,   3,   4,   4,   5,   3,   4,   5,   3,   4,   4,   4,
             5,   3,   1,   4,   2,   3,   3,   4,   4,   5,   3,   4,   4,
             6,   8,  14,  13,  12,  10,   9,  13,  12,  11,  12,  14,   9,
            15,  15,  13,  15,  16,  13,  12,  10,   9,   9,   8,   9,   8,
             5,   6,   5,   5,   3,   6,   6,   6,   5,   5,   4,   3,   3])




```python
# test['total_cases']=p
# test['city']=lb.inverse_transform(test['city'])
```


```python
# test[['city','year','weekofyear','total_cases']].to_csv("dataset/submission1.csv" ,index=False)
```
