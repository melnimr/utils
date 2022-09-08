## concatenate

df_rezin = pd.DataFrame(mat_file['ReZin'].reshape(4600 * 21),columns=['ReZin'])
df_imzin = pd.DataFrame(mat_file['ImZin'].reshape(4600 * 21),columns=['ImZin'])
df_freq = pd.DataFrame(mat_file['Freq'].reshape(4600 * 21),columns=['Freq'])
df_input = pd.DataFrame(mat_file['Input'])


replicated_df_input = pd.concat([df_input] *21, ignore_index= True)
replicated_df_input.sort_values(by=[0],inplace=True)
replicated_df_input.reset_index(drop=True,inplace = True)

fulldataframe_4600 = pd.concat([replicated_df_input,df_freq, df_rezin,df_imzin], axis =1 )
fulldataframe_4600.to_csv('fulldataframe_4600.csv',index=False)
