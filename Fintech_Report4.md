# <center>Fintech_Report4

<center>毛阿妍May


<center>2019/8/23

[TOC]

## 1:实验⽬的 

本实验旨在对金融文本数据进行自然语言处理并对结果进行评估分析。

1. 对TUSHARE 上市公司基本信息中的股票经营范围文本数据进行自然语言处理，
2. 并根据相应的行业分类标签进行文本分类,
3. 并评估分类结果。

## 2:实验步骤 

### 2.1:获取经营范围文本数据 

根据经营范围文本数据接口说明与示例( https://tushare.pro/document/2?doc_id=112 ) 获得经营范围文本数据，其中 fields 参数选择股票代码“ts_code”和经营范围 “business_scope”。 

```python
df1['ts_code'] = df1['ts_code'].str.strip('.SZ')
df1 = df1.rename(columns={'ts_code': 'code'})
```

### 2.2:获取行业分类标签数据

根据行业分类标签数据接口说明与示例(http://tushare.org/classifying.html)获得行业分类标签数据。根据股票代码“code”，可将上一步中获得的股票的经营范围与行业名称相对应。

### 2.3:将文本数据数值化

0. 根据官网说明进行“结巴”中文分词的安装。 “结巴”中文分词官方参考文档:(https://github.com/fxsjy/jieba)。 

1. 利用“结巴”中文分词技术对经营范围文本数据进行分词。 

2. 利用“结巴”中文分词技术对经营范围文本数据进行关键词提取。 

3. 分词结果和关键词串联作为预处理后的文本数据。 

4. 对预处理后的文本数据进行词频向量化，并进行 TF-IDF 处理得到文本数据数值化向量。 

   (我是调包实现的）

jieba的分词功能:

```python
merged_df['wordsegment'] = words  #文本数据分词
merged_df['tags'] = tags #关键词获取功能获得关键词
```

### 2.4:基于数值化文本向量进行分类器学习

1. 进行训练集和测试集的划分。参考工具:sklearn KFold
2. 构建朴素贝叶斯多项式分类器。由于行业标签数量众多，可筛选出单类数据量大于 80 的类进行学习。
    分类器参考工具:sklearn MultinomialNB。
3. 对分类器的效果进行评估，评价指标为 precision，recall，F1-score。 分类评价参考工具:sklearn classification_report





## 3:实验结果

训练的结果不错，其中直接用文本数据分词而不是关键词的效果更好。

![屏幕快照 2019-08-26 下午11.45.34](/Users/may/Desktop/assets/屏幕快照 2019-08-26 下午11.45.34.png)



## 4:附录（主要源代码）

```python
def get_data(token, N):
	# get_data
	pro = ts.pro_api(token)
	pd.set_option('max_colwidth',120)
	df0 = pro.stock_company(exchange='SZSE', fields='ts_code, business_scope')
	df1 = df0.dropna(axis=0, how='any')
	df2 = ts.get_industry_classified()

	# merge
	# Your code here
    # Answer begin
	df1['ts_code'] = df1['ts_code'].str.strip('.SZ')
	df1 = df1.rename(columns={'ts_code': 'code'})
	# Answer end
	df = pd.merge(df1,df2, how='right')

	# filter by number of records
	nonan_df = df.dropna(axis=0, how='any')
	vc  = nonan_df['c_name'].value_counts()
	pat = r'|'.join(vc[vc>N].index)          
	merged_df  = nonan_df[nonan_df['c_name'].str.contains(pat)]
	
	return merged_df
```



```python
def text_preprocess(merged_df):
	# word segmentation + extract keywords (using jieba)
	# Your code here
    # Answer begin
	business_scope = merged_df['business_scope']
	words = []
	tags = []
	for i in business_scope.index:
		words.append(" ".join(jieba.cut(business_scope[i])))
		tags.append(" ".join(jieba.analyse.extract_tags(business_scope[i])))
	merged_df['wordsegment'] = words
	merged_df['tags'] = tags

	return merged_df
	# Answer end
```



```python
def classification(processed_df):
	# split into train and test sets
	kf = KFold(n_splits = 10, shuffle = True, random_state = 2)
	split_result = next(kf.split(processed_df), None)
	train = processed_df.iloc[split_result[0]]
	test = processed_df.iloc[split_result[1]]
	
	# TF-IDF
	X_train_tf, X_test_tf = TF_IDF(train, test)

	# classification
	# Your code here
    # Answer begin


	vc = processed_df['c_name'].value_counts()
	vc = pd.Series([i for i in range(len(vc))], index=vc.index)
	y_train = np.array([vc[train['c_name'][i]] for i in train.index])
	y_test = np.array([vc[test['c_name'][i]] for i in test.index])


	clf = MultinomialNB()
	clf.fit(X_train_tf, y_train)

	y_predict = clf.predict(X_test_tf)

	print("test['c_name'] type is", type(test['c_name']))
	print(" y_predict type is",  y_predict)
	results = metrics.classification_report(test['c_name'], y_predict, target_names=vc.index)
	# ??,target_names = vc.index

	# Answer end
	return results
```



```python
def TF_IDF(train, test):
	# Your code here
    # Answer begin

	vectorizer = CountVectorizer()
	transformer = TfidfTransformer()
	X_train_tf = transformer.fit_transform(vectorizer.fit_transform(train['business_scope'].values)).toarray()
	X_test_tf = transformer.transform(vectorizer.transform(test['business_scope'].values)).toarray()

	# Answer end
	return X_train_tf, X_test_tf
```

