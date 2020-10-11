```
import pandas as pd
import numpy as np
import seaborn as sns

```


```
data=pd.read_csv("C:/Users/DELL/Documents/House/data.csv")

```


```
data.head(
)
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
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>street</th>
      <th>city</th>
      <th>statezip</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-05-02 00:00:00</td>
      <td>313000.0</td>
      <td>3.0</td>
      <td>1.50</td>
      <td>1340</td>
      <td>7912</td>
      <td>1.5</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1340</td>
      <td>0</td>
      <td>1955</td>
      <td>2005</td>
      <td>18810 Densmore Ave N</td>
      <td>Shoreline</td>
      <td>WA 98133</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-05-02 00:00:00</td>
      <td>2384000.0</td>
      <td>5.0</td>
      <td>2.50</td>
      <td>3650</td>
      <td>9050</td>
      <td>2.0</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>3370</td>
      <td>280</td>
      <td>1921</td>
      <td>0</td>
      <td>709 W Blaine St</td>
      <td>Seattle</td>
      <td>WA 98119</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-05-02 00:00:00</td>
      <td>342000.0</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>1930</td>
      <td>11947</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1930</td>
      <td>0</td>
      <td>1966</td>
      <td>0</td>
      <td>26206-26214 143rd Ave SE</td>
      <td>Kent</td>
      <td>WA 98042</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-05-02 00:00:00</td>
      <td>420000.0</td>
      <td>3.0</td>
      <td>2.25</td>
      <td>2000</td>
      <td>8030</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1000</td>
      <td>1000</td>
      <td>1963</td>
      <td>0</td>
      <td>857 170th Pl NE</td>
      <td>Bellevue</td>
      <td>WA 98008</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-05-02 00:00:00</td>
      <td>550000.0</td>
      <td>4.0</td>
      <td>2.50</td>
      <td>1940</td>
      <td>10500</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1140</td>
      <td>800</td>
      <td>1976</td>
      <td>1992</td>
      <td>9105 170th Ave NE</td>
      <td>Redmond</td>
      <td>WA 98052</td>
      <td>USA</td>
    </tr>
  </tbody>
</table>
</div>




```
data.columns
```




    Index(['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
           'floors', 'waterfront', 'view', 'condition', 'sqft_above',
           'sqft_basement', 'yr_built', 'yr_renovated', 'street', 'city',
           'statezip', 'country'],
          dtype='object')




```
data.isnull().sum()
```




    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    street           0
    city             0
    statezip         0
    country          0
    dtype: int64




```
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

```


```
train=data.drop(['price','date', 'yr_renovated', 'street', 'city', 'statezip', 'country','waterfront', 'view','condition', 
                 'sqft_above', 'sqft_basement' , 'yr_built'], axis=1)
test=data['price']
```


```
x_train, x_test, y_train, y_test=train_test_split(train, test, test_size=0.3, random_state=2)
```


```
regr = LinearRegression()
```


```
regr.fit(x_train, y_train)
```




    LinearRegression()




```
pred=regr.predict(x_test)


```


```
pred
```




    array([ 361100.67365501,  317614.11382178,  520277.63213974, ...,
           1145356.81292978,  788832.46358354,  242438.23261294])




```
regr.score(x_test, y_test)
```




    0.08427856716814586




```
y_test
```




    4111     232000.0
    1996     299950.0
    2307    1085000.0
    3607     229800.0
    1519     499950.0
              ...    
    3750     590000.0
    1599     425000.0
    2672    1200000.0
    3353    1140000.0
    2793     400000.0
    Name: price, Length: 1380, dtype: float64




```

```
