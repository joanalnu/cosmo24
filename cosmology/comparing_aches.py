# Comparison plots
# with original last year
# Ho = [72, 70.0, 67.4, 48.5, (1/slope2), (1/slope3)]
# year = [2001, 2017, 2020, 2023, 2024, 2024]
# error_low = [8, 8.0, 0.5, 5.4, 0.0, 0.0]
# error_high = [0, 12.0, 0.5, 5.4, 0.0, 0.0]
# labels = ['Standard Candles (HST)', 'LIGO','CMB (Planck)', 'Cepheids 23', 'SNIa', 'Cepheids24+SNIa']

# My_Ho = [48.5, (1/slope2), (1/slope3)]
# my_year = [2023, 2024, 2024]
# my_errors = [5.4, 0.0, 0.0]
# my_labels = ['Cepheids 23', 'SNIa', 'Cepheids24+SNIa']


# with last year recalc
# Ho = [72, 70.0, 67.4, (1/slope4), (1/slope2), (1/slope3)]
# year = [2001, 2017, 2020, 2023, 2024, 2024]
# error_low = [8, 8.0, 0.5, 0.0, 0.0, 0.0]
# error_high = [0, 12.0, 0.5, 0.0, 0.0, 0.0]
# labels = ['Standard Candles (HST)', 'LIGO','CMB (Planck)', 'Cepheids 23', 'SNIa', 'Cepheids24+SNIa']

# My_Ho = [(1/slope4), (1/slope2), (1/slope3)]
# my_year = [2023, 2024, 2024]
# my_errors = [0.0, 0.0, 0.0]
# my_labels = ['Cepheids 23(4)', 'SNIa', 'Cepheids24+SNIa']



##################################################################################################################################################



# # with dataset
# df_dataset = pd.read_csv(f'{dirpath}/dataset.csv')
# year = df_dataset['Year']
# aux_id = [i for i in range(len(year))]

# Ho = df_dataset['Value']
# error_low = df_dataset['Lower']
# error_high = df_dataset['Upper']

# authors = df_dataset['First Author']
# colabs = df_dataset['et al']
# # old_authors = list(df_dataset['First Author'])
# # colabs = list(df_dataset['et al'])
# # authors = list()
# # for i in range(len(df_dataset)):
# #     if colabs[i]=='Y':
# #         authors.append(f'{old_authors[i]} et al')

# # for i in range(len(df_dataset)):
# #     if df_dataset.loc[i, 'et al']=='Y':
# #         df_dataset.loc[i, 'First Author'] = f"{df_dataset.loc[i, 'First Author']} et al"

# authors = df_dataset['First Author']
# colabs = df_dataset['et al']
# for i in range(len(colabs)):
#     if colabs[i]=='Y':
#         authors[i] = f'{authors[i]} et al'



# labels = df_dataset['Type']
# methods = df_dataset['Direct/Indirect']
# markers = list(methods.replace({'Direct':'o','Indirect':'^'}))

# types = df_dataset['Type']
# colors = list(types.replace({'Cepheids':'red', 'Cepheids-SNIa':'orange', 'CMB with Planck':'black', 'CMB without Planck':'gold',
# 'GW related':'green', 'HII galaxies':'pink', 'Lensing related; mass model-dependent':'pink',
# 'Masers':'pink', 'Miras-SNIa':'pink', 'No CMB; with BBN':'pink', 'Optimistic average':'pink',
# 'Pl(k) + CMB lensing':'pink','SNII':'pink', 'Surface Brightness Fluctuations':'pink', 'TRGB-SNIa':'firebrick',
# 'Tully-Fisher Relation':'pink', 'Ultra-conservative; no cepheids; no lensing':'pink',
# 'BAO':'coral', 'SNIa-BAO':'sandybrown', 'other':'pink', 'SNIa':'orange','TRGB':'firebrick'}))

# # My_Ho = [(1/slope4), (1/slope2), (1/slope3)]
# # my_year = aux_id[-3:]
# # my_errors = [0.065415337, 0.0959346, 0.054367662]
# # my_labels = ['Cepheids 23(4)', 'SNIa', 'Cepheids24+SNIa']
# My_Ho = [(1/slope2), (1/slope3)]
# my_year = aux_id[-2:]
# my_errors = [0.0959346, 0.054367662]
# my_labels = ['SNIa', 'Cepheids24+SNIa']

# figc, ax = plt.subplots(1,1)
# for i in range(len(aux_id)):
#     ax.scatter(aux_id[i], Ho[i], color=colors[i], marker=markers[i], alpha=0.8, s=10) #s=10
# # ax.scatter(aux_id, Ho, color=colors, alpha=0.8, s=10) #s=10
# ax.errorbar(aux_id, Ho,  yerr=[error_low, error_high], color='gray', fmt='none', alpha=0.5)
# ax.scatter(my_year, My_Ho, color='red', s=20, marker='2')
# ax.errorbar(my_year, My_Ho, yerr=[my_errors], color='gray', alpha=0.5, fmt='none')
# ax.set_xlabel('Year of publication')
# ax.set_ylabel('Ho in [Km/s/Mpc]')
# # for i in range(len(year)):
# #     plt.annotate(labels[i], (year[i], Ho[i]), textcoords='offset points', xytext=(5,5), fontsize=7) # ha='center'

# ax.set_xticks(range(len(authors)), authors, rotation=45, fontsize=2, ha='right')
# figc.savefig(f'{dirpath}/comparissonplot.png', dpi=600)



# figc2, ax = plt.subplots(1, 1, figsize=(12, 7))
# colors = list(methods.replace({'Direct':'lightblue', 'Indirect':'pink'}))
# ax.scatter(aux_id, Ho, color=colors)
# ax.errorbar(aux_id, Ho, yerr=[error_low, error_high], fmt='none', color='gray', capsize=2, alpha=0.5)
# ax.set_xlabel('Year of publication')
# ax.set_ylabel('Ho in [Km/s/Mpc]')

# # ax.scatter(my_year, My_Ho, color='red', s=20, marker='2')
# ax.set_xticks(range(len(aux_id)), year, rotation=55, ha='center', fontsize=5)

# dir_id = list()
# ind_id = list()
# error_low_dir = list()
# error_high_dir = list()
# error_low_ind = list()
# error_high_ind = list()
# for i in range(len(aux_id)):
#     if methods[i]=='Direct':
#         dir_id.append(aux_id[i])
#         error_low_dir.append((Ho[i]-error_low[i]))
#         error_high_dir.append((Ho[i]+error_high[i]))
#     elif methods[i]=='Indirect':
#         ind_id.append(aux_id[i])
#         error_low_ind.append((Ho[i]-error_low[i]))
#         error_high_ind.append((Ho[i]+error_high[i]))

# dir_id = list(dir_id)
# ind_id = list(ind_id)
# error_low_dir = list(error_low_dir)
# error_high_dir = list(error_high_dir)
# error_low_ind = list(error_low_ind)
# error_high_ind = list(error_high_ind)

# cep_df.loc[cep_df['name']==line[0].replace('_',' '), 'V_dist_err'] = float(line[1])/1000000-float(line[2])/1000000
# directs = df_dataset.loc[df_dataset['Direct/Indirect']=='Direct', 'Value']
# indirects = df_dataset.loc[df_dataset['Direct/Indirect']=='Indirect', 'Value']

# mean_direct = np.average(directs)
# mean_indirect = np.average(indirects)
# mean_direct_list = [mean_direct for i in range(len(aux_id))]
# mean_indirect_list = [mean_indirect for i in range(len(aux_id))]
# ax.plot(aux_id, mean_direct_list, color='blue', label=f'Slope: {mean_direct}')
# ax.plot(aux_id, mean_indirect_list, color='red', label=f'Slope: {mean_indirect}')
# ax.legend()


# ax.fill_between(dir_id, error_low_dir, error_high_dir, color='lightblue', alpha=0.5)
# ax.fill_between(ind_id, error_low_ind, error_high_ind, color='pink', alpha=0.5)

# ax.legend()
# figc2.savefig(f'{dirpath}/year_comparissonplot.png', dpi=300)