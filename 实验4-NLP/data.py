import tushare as ts
import pandas as pd
import jieba
import jieba.analyse

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