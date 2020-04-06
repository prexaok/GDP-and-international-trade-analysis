import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def make_growth_data(df, drops):
	growth_data = pd.DataFrame()

	''' CREO EL DATAFRAME CON LAS SERIES HISTORICAS DE CRECIMINETO PARA CADA PAÌS '''

	growth_data['year'] = df.iloc[:,4:].columns.values

	for country in df['Country Name'].unique():
		growth_data[str(country)] = df[df['Country Name'] == str(country)].iloc[:,4:].stack(dropna = False).reset_index(drop = True)

	growth_data.drop(columns = drops['drop_1'], inplace= True)

	return growth_data



def make_means(growth, tol):
	corr = growth.corr()
	links = corr.stack().reset_index()

	links.columns = ['country1', 'country2','correlation']

	''' ACA UN NUEVO DATAFRAME DONDE SOLO ME QUEDO CON LOS PAISES CON MAS DE "TOLERANCE" DE CORRELACION (EN MODULO)'''

	links_filtered=links.loc[ ((links['correlation'] > tol)|(links['correlation']<-tol)) & (links['country1'] != links['country2']) ]

	mean = links_filtered[links_filtered.correlation<1].correlation.mean()

	return mean


def world_mean(growth_data, tol, n):
	mom = 0
	for i in range(n):
		random_countries = []

		for i in range(7):
			j = int(209*random.random())
			random_countries.append(growth_data.columns[j])
		growth_random = growth_data[random_countries]
		corr_random = growth_random.corr()
		links_random = corr_random.stack().reset_index()

		links_random.columns = ['country1', 'country2','correlation']

		links_filtered_random=links_random.loc[ ((links_random['correlation'] > tol)|(links_random['correlation']<-tol)) & (links_random['country1'] != links_random['country2']) ]
		random_mean = links_filtered_random[links_filtered_random.correlation<1].correlation.mean()
		mom += random_mean
	mom = mom/n

	return mom



international_trades = {}
def mean_trade (df_country):
	df_country.groupby('Partner Economy', group_keys =False).apply(lambda x: inner_mean(x, df_country))





def inner_mean (df_partner, df_country):
	import_mean =  df_partner[df_partner.Indicator == 'Commercial services imports by sector and partner – annual (2005-onwards)'].Value.mean()
	export_mean =  df_partner[df_partner.Indicator == 'Commercial services exports by sector and partner – annual (2005-onwards)'].Value.mean()
	partner = df_partner['Partner Economy'].iloc[0]
	country = df_country['Reporting Economy'].iloc[0]
	international_trades[(country,partner)] = [import_mean, export_mean]


def make_dict(df):
	df.groupby('Reporting Economy', group_keys = False).apply(mean_trade)
	np.save('international_trades.npy', international_trades)




def make_trades_map(color_map, places_dict, base_map, trades_dict, country_list):
  for i, source in enumerate(country_list):
    color = color_map(i*5)
    lat1, lon1 = float(places_dict[source][0]),float(places_dict[source][1])
    base_map.plot(lon1,lat1,"o", c=color, alpha = 0.8, markersize = 4, label = source+' services exports')
    plt.legend(loc = 'lower left')
    for i, target in enumerate(places_dict):
      try:
        lat2, lon2 = float(places_dict[target][0]),float(places_dict[target][1])
        lwidth = trades_dict[(source, target)][1]/1000
        max_lwidth = 2
        line, = base_map.drawgreatcircle(lon1, lat1, lon2, lat2, lw=min(lwidth,max_lwidth), color = color, alpha = 0.8)
        p = line.get_path()
        # find the index which crosses the dateline (the delta is large)
        cut_point = np.where(np.abs(np.diff(p.vertices[:, 0])) > 200)[0]
        p_length = p.vertices.shape[0]
        x,y =  p.vertices[int(-p_length/2),:] 
        x2, y2 = p.vertices[int(-p_length/2)+1,:] 
        base_map.plot(lon2,lat2,"o", c=color, markersize = 4, alpha = 0.8)
        #plt.arrow(x,y,x2-x,y2-y, linewidth = 1 , head_width=1, head_length=0.4, color = color, alpha = 0.7)
        if cut_point:
          cut_point = cut_point[0]

          # create new vertices with a nan inbetween and set those as the path's vertices
          new_verts = np.concatenate(
                                     [p.vertices[:cut_point, :], 
                                      [[np.nan, np.nan]], 
                                      p.vertices[cut_point+1:, :]]
                                     )
          p.codes = None
          p.vertices = new_verts
      except KeyError:
        pass