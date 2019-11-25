#encoding=utf-8
from data import get_data
from data import text_preprocess
from classification import classification
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
	token = 'a6dae538a760f0b9e39432c1bff5e50a1c462a1a087e994dae18fa04'
	N = 80 # number for filtering classes with more than N records
	merged_df = get_data(token, N)
	processed_df = text_preprocess(merged_df)
	results = classification(processed_df)
	print(results)
