import pandas as pd

def df_missing_value_injector(df):

	null_values = ['','?','-999']

	for null_i in null_values:
		df = df.replace(null_i,np.nan)

	d_magic_string = 'NaN_789hghgg'
	df.replace(null_values,d_magic_string, inplace=True)

	micro_features = []
	for i, col in enumerate(df.columns):
		df_i = pd.get_dummies(df[col])
		X_i = np.array(df_i.as_matrix(),dtype=float)

		if d_magic_string in df_i.columns:
			missing_data_index = list(df_i.columns).index(d_magic_string)
			missing_data_filter = np.array(df_i[df_i.columns[missing_data_index]].values,dtype=int)==1
			X_i[missing_data_filter,:] = float('NaN')
			X_i[:,missing_data_index] = missing_data_filter
		else:
			pass

		if i==0:
			X = X_i
		else:
			X = np.hstack((X,X_i))
			#print X.shape
		micro_features.append(df_i.columns)

	return X, micro_features
