import pandas as pd
import os
import numpy as np
import seaborn as sns
import scipy as scp
import scipy.stats as stats
from matplotlib import pyplot as plt

import time
from datetime import datetime

from scipy.interpolate import make_interp_spline

import matplotlib.colors as mcol
import matplotlib.cm as cm

import matplotlib.ticker as ticker
import matplotlib

import math

from dateutil import parser

"""
Функции конвертирующие даты данные в виде строки в объекты datetime
Parameters
    ----------
    dt_str : str
        строка с датой

    Returns
    -------
    datetime 
"""

def convert_date_time_to_datetime(dt_str):
    return parser.parse(dt_str)

def convert_date_time_to_datetime_Volksgarten(dt_str):
    dt = parser.parse(dt_str)
    minutes = dt.time().minute
    if (minutes<=29):
        return dt.replace(minute=15,second=0)
    else:
        return dt.replace(minute=45,second=0)
	

"""
Функции конвертирующие datetime в даты и время
Parameters
    ----------
    dt : datetime
        строка с датой

    Returns
    -------
    datetime 
"""

def convert_datetime_to_date(dt):
    return dt.date()

def convert_datetime_to_time(dt):
    return dt.time()

"""
Функция строящая график зависимости температуры от времени для одной локации в течении одного дня
Parameters
    ----------
    df : pd.DataFrame
        данные
	location : str
        название локации
	date : str
		дата
	savefig : bool
        сохранять полученную картинку или нет
	legend_loc : str
        положение легенды на графике
	period : str
		Период времени за который строиться график. 
		Нужен только для пометок в названиях файлов, предполагается два вида периодов 12-17 (день) и 00-24 (весь день)
	path : str
		строка, содержащая путь куда сохранить картинку

    Returns
    -------
    None
"""

def plot_one_day_T_one_location(df,location,date='2021-12-13',savefig=False,legend_loc='upper left',period='',path=''):
	date_req = (df['date']==parser.parse(date).date())
	T = df[date_req][location]
	d = []
	for i,j in zip(T.index.hour,T.index.minute):
			if str(j)=='0':
				d.append(str(i)+':00')
			else:
				d.append(str(i)+':'+str(j))
	max_T = T.max()
	min_T = T.min()
	sns.set(style="whitegrid", font_scale=1.4)
	fig, ax = plt.subplots(figsize=(12, 8))
	plt.plot(d, T,'-o',color='black',label=location, alpha=1)
	plt.plot([], [], ' ', label=T.index[0].date())
	plt.ylabel('T,'+u'\N{DEGREE SIGN}C')
	plt.xticks(rotation=30)
	n = 2
	[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
	plt.ylim(min_T-2,max_T+3)
	plt.legend(fontsize=16,loc=legend_loc)
	plt.tight_layout()
	if savefig==True:
		plt.savefig(path+'/tmp/one_day_T_'+location+'_'+date+'_'+period+'.png')
	plt.show()

"""
Функция строящая график зависимости температуры от времени для нескольких локации в течении одного дня
Parameters
    ----------
    df : pd.DataFrame
        данные
	locations : list
        названия локаций
	date : str
		дата
	savefig : bool
        сохранять полученную картинку или нет
	legend_loc : str
        положение легенды на графике
	period : str
		Период времени за который строиться график. 
		Нужен только для пометок в названиях файлов, предполагается два вида периодов 12-17 (день) и 00-24 (весь день)
	path : str
		строка, содержащая путь куда сохранить картинку

    Returns
    -------
    None
"""

def plot_one_day_T_multiple_location(df,locations=[],date='2021-12-13',savefig=False,legend_loc='upper left',period='',path=''):
	location_colors = {'Opel Bauer':'red',
                   'Bahnhof Mülheim':'pink',
                   'Mülheimer Park':'orange',
                   'Mülheimer Park 2':'gold',
                   'Volksgarten - 1':'green',
                   'Rheininsel':'mediumorchid',
                   'Volksgarten - 2':'blue'}
	date_req = (df['date']==parser.parse(date).date())
	df = df[date_req]
	max_T = -1000
	min_T = 1000
	sns.set(style="whitegrid", font_scale=1.4)
	fig, ax = plt.subplots(figsize=(12, 8))
	for location in locations:
		T = df[location]
		d = []
		if T.max() > max_T: max_T = T.max()
		if T.min() < min_T: min_T = T.min() 
		for i,j in zip(T.index.hour,T.index.minute):
			if str(j)=='0':
				d.append(str(i)+':00')
			else:
				d.append(str(i)+':'+str(j))
		plt.plot(d, T,'-o',color=location_colors[location],label=location, alpha=1)
	plt.plot([], [], ' ', label=T.index[0].date())
	plt.ylabel('T,'+u'\N{DEGREE SIGN}C')
	plt.xticks(rotation=30)
	n = 2
	[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
	plt.ylim(min_T-2,max_T+8)
	plt.legend(fontsize=16,loc=legend_loc,ncol=2)
	plt.tight_layout()
	if savefig==True:
		plt.savefig(path+'/tmp/one_day_T_mult_'+date+'_'+period+'.png')
	plt.show()


"""
Функция строящая график зависимости средней дневной температуры от даты для одной локации
Также выделяет дату с максимальной и минимальной температурой и сами эти температуры за данный период
Parameters
    ----------
    df : pd.DataFrame
        данные
	location : str
        название локации
	date_from : str
	date_to : str
		с какой по какую даты строить график
	savefig : bool
        сохранять полученную картинку или нет
	legend_loc : str
        положение легенды на графике
	path : str
		строка, содержащая путь куда сохранить картинку

    Returns
    -------
    None
"""

def plot_mean_T_one_location(df,location,date_from='1980-01-01',date_to='2030-12-31',savefig=False,legend_loc='upper left',path=''):
	dt_from = parser.parse(date_from).date()
	dt_to   = parser.parse(date_to+' 23:59:59').date()
	T = df[location][dt_from:dt_to]
	d = T.index
	max_T = T.max()
	min_T = T.min()
	max_T_arg = list(d)[T.argmax()]
	max_T_arg_p1 = list(d)[T.argmax()+1]
	max_T_arg_m1 = list(d)[T.argmax()-1]
	min_T_arg = list(d)[T.argmin()]
	min_T_arg_p1 = list(d)[T.argmin()+1]
	min_T_arg_m1 = list(d)[T.argmin()-1]
	sns.set(style="whitegrid", font_scale=1.4)
	fig, ax = plt.subplots(figsize=(12, 8))
	plt.plot(d, T,'-o',color='black',label=location)
	plt.ylabel('T,'+u'\N{DEGREE SIGN}C')
	plt.xticks(rotation=30)
	plt.ylim(min_T-2,max_T+3)
	ax.axvspan(max_T_arg_m1, max_T_arg_p1, alpha=0.3, color='red', label='max T = '+str(np.around(max_T,1))+u'\N{DEGREE SIGN}C')
	ax.axvspan(min_T_arg_m1, min_T_arg_p1, alpha=0.3, color='blue',label='min T = '+str(np.around(min_T,1))+u'\N{DEGREE SIGN}C')
	plt.legend(fontsize=15,loc=legend_loc)
	plt.tight_layout()
	if savefig==True:
		plt.savefig(path+'/tmp/mean_T_'+location+'_'+date_from+'_'+date_to+'.png')
	plt.show()

"""
Функция строящая график зависимости средней дневной температуры от даты для нескольких локации
Также выделяет дату с максимальной и минимальной температурой и сами эти температуры за данный период
Parameters
    ----------
    df : pd.DataFrame
        данные
	locations : list
        названия локаций
	date_from : str
	date_to : str
		с какой по какую даты строить график
	savefig : bool
        сохранять полученную картинку или нет
	legend_loc : str
        положение легенды на графике
	path : str
		строка, содержащая путь куда сохранить картинку

    Returns
    -------
    None
"""

def plot_mean_T_multiple_locations(df,locations=[],date_from='1980-01-01',date_to='2030-12-31',savefig=False,legend_loc='upper left',path=''):
	location_colors = {'Opel Bauer':'red',
                   'Bahnhof Mülheim':'pink',
                   'Mülheimer Park':'orange',
                   'Mülheimer Park 2':'gold',
                   'Volksgarten - 1':'green',
                   'Rheininsel':'mediumorchid',
                   'Volksgarten - 2':'blue'}

	dt_from = parser.parse(date_from).date()
	dt_to   = parser.parse(date_to+' 23:59:59').date()

	df = df[dt_from:dt_to]

	max_T = -1000
	min_T = 1000
	max_T_arg = 0
	max_T_arg_p1 = 0
	max_T_arg_m1 = 0
	min_T_arg = 0
	min_T_arg_p1 = 0
	min_T_arg_m1 = 0

	for location in locations:
		if (df[location].max() > max_T):
			max_T = df[location].max()
			max_T_arg = list(df.index)[df[location].argmax()]
			max_T_arg_p1 = list(df.index)[df[location].argmax()+1]
			max_T_arg_m1 = list(df.index)[df[location].argmax()-1]
		if (df[location].min() < min_T):
			min_T = df[location].min()
			min_T_arg = list(df.index)[df[location].argmin()]
			min_T_arg_p1 = list(df.index)[df[location].argmin()+1]
			min_T_arg_m1 = list(df.index)[df[location].argmin()-1]

	sns.set(style="whitegrid", font_scale=1.4)
	fig, ax = plt.subplots(figsize=(12, 8))

	for location in locations:
		T = df[location]
		d = T.index
		plt.plot(d, T,'-o',color=location_colors[location],label=location, alpha=0.9)

	plt.ylabel('T,'+u'\N{DEGREE SIGN}C')
	plt.xticks(rotation=30)
	plt.ylim(min_T-2,max_T+8)
	ax.axvspan(max_T_arg_m1, max_T_arg_p1, alpha=0.3, color='red', label='max T = '+str(np.around(max_T,1))+u'\N{DEGREE SIGN}C')
	ax.axvspan(min_T_arg_m1, min_T_arg_p1, alpha=0.3, color='blue',label='min T = '+str(np.around(min_T,1))+u'\N{DEGREE SIGN}C')
	plt.legend(fontsize=15,loc=legend_loc,ncol=2)
	plt.tight_layout()
	if savefig==True:
		plt.savefig(path+'/tmp/mean_T_mult_'+date_from+'_'+date_to+'.png')
	plt.show()

"""
Функция строящая график зависимости разности средней дневной температуры в Opel Bauer и остальныз локациях от даты
Parameters
    ----------
    df : pd.DataFrame
        данные
	locations : list
        названия локаций
	date_from : str
	date_to : str
		с какой по какую даты строить график
	savefig : bool
        сохранять полученную картинку или нет
	legend_loc : str
        положение легенды на графике
	path : str
		строка, содержащая путь куда сохранить картинку

    Returns
    -------
    None
"""

def plot_mean_T_multiple_locations_diff(df,locations=[],date_from='1980-01-01',date_to='2030-12-31',savefig=False,legend_loc='upper left',path=''):
	location_colors = {'Opel Bauer':'red',
                   'Bahnhof Mülheim':'pink',
                   'Mülheimer Park':'orange',
                   'Mülheimer Park 2':'gold',
                   'Volksgarten - 1':'green',
                   'Rheininsel':'mediumorchid',
                   'Volksgarten - 2':'blue'}

	dt_from = parser.parse(date_from).date()
	dt_to   = parser.parse(date_to+' 23:59:59').date()

	df = df[dt_from:dt_to]

	max_T = -1000
	min_T = 1000
	for location in locations:
		if (df[location].max() > max_T):
			max_T = df[location].max()
		if (df[location].min() < min_T):
			min_T = df[location].min()

	sns.set(style="whitegrid", font_scale=1.4)
	fig, ax = plt.subplots(figsize=(12, 8))

	for location in locations:
		T = df[location]
		d = T.index
		plt.plot(d, T,'-o',color=location_colors[location],label=location, alpha=0.9)

	plt.ylabel(r'$\Delta$ T,'+u'\N{DEGREE SIGN}C')
	plt.xticks(rotation=30)
	plt.ylim(min_T-3,max_T+3)
	plt.legend(fontsize=15,loc=legend_loc,ncol=2)
	plt.tight_layout()
	if savefig==True:
		plt.savefig(path+'/tmp/mean_T_diff_OB_'+date_from+'_'+date_to+'.png')
	plt.show()

"""
Функция гистограму и боксплот для средней дневной температуры в отдной из локаций
Parameters
    ----------
    df : pd.DataFrame
        данные
	locations : str
        название локации
	date_from : str
	date_to : str
		с какой по какую даты строить график
	savefig : bool
        сохранять полученную картинку или нет
	legend_loc : str
        положение легенды на графике
	type : str
        dT если строиться разность температур
	path : str
		строка, содержащая путь куда сохранить картинку

    Returns
    -------
    None
"""

def plot_hist_boxplot(df,location,date_from='1980-01-01',date_to='2030-12-31',savefig=False,legend_loc='upper left',type='T',path=''):
	dt_from = parser.parse(date_from).date()
	dt_to   = parser.parse(date_to+' 23:59:59').date()
	x = df[location][dt_from:dt_to]
	matplotlib.rcParams.update({'font.size': 15})
	f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
	sns.boxplot(x=x, ax=ax_box)
	sns.histplot(x=x, bins=12, kde=False, stat='probability', ax=ax_hist, label=location)
	ax_box.set(yticks=[])
	ax_box.axes.set_xlabel('')
	if type == 'dT':
		ax_hist.axes.set_xlabel(r'$\Delta$ T,'+u'\N{DEGREE SIGN}C')
	else:
		ax_hist.axes.set_xlabel('T,'+u'\N{DEGREE SIGN}C')
	y_max = ax_hist.get_ylim()[1]
	ax_hist.axes.set_ylim(0,y_max+0.1)
	ax_hist.plot([], [], ' ', label=date_from[5:7]+'/'+date_from[0:4]+' - '+date_to[5:7]+'/'+date_from[0:4])
	# ax_hist.legend(loc=legend_loc)
	plt.legend(loc=legend_loc)
	sns.despine(ax=ax_hist)
	sns.despine(ax=ax_box, left=True)
	plt.tight_layout()
	if savefig==True:
		plt.savefig(path+'/tmp/hist_boxplot_'+location+'_'+date_from+'_'+date_to+'.png')
	plt.show()

"""
Функция находящая самый жаркий день среди нескольких локаций за определенный период
Parameters
    ----------
    df : pd.DataFrame
        данные
	locations : str
        названия локаций
	date_from : str
	date_to : str
		с какой по какую даты искать

    Returns
    -------
    datetime
"""

def get_hotest_date(df,locations=[],date_from='1980-01-01',date_to='2030-12-31'):
	dt_from = parser.parse(date_from).date()
	dt_to   = parser.parse(date_to+' 23:59:59').date()
	df = df[dt_from:dt_to]
	max_T = -1000
	for location in locations:
		if (df[location].max() > max_T):
			max_T = df[location].max()
			max_T_arg = list(df.index)[df[location].argmax()]
	return max_T_arg.strftime("%Y-%m-%d")

"""
Функция находящая самый холодный день среди нескольких локаций за определенный период
Parameters
    ----------
    df : pd.DataFrame
        данные
	locations : str
        названия локаций
	date_from : str
	date_to : str
		с какой по какую даты искать

    Returns
    -------
    datetime
"""

def get_coldest_date(df,locations=[],date_from='1980-01-01',date_to='2030-12-31'):
	dt_from = parser.parse(date_from).date()
	dt_to   = parser.parse(date_to+' 23:59:59').date()
	df = df[dt_from:dt_to]
	min_T = 1000
	for location in locations:
		if (df[location].min() < min_T):
			min_T = df[location].min()
			min_T_arg = list(df.index)[df[location].argmin()]
	return min_T_arg.strftime("%Y-%m-%d")


"""
Класс принимающий на вход путь к сырым данным с замерами температур в разных локациях в разное время
Parameters
    ----------
    path_to_files : str
		путь к данным
"""


class DataLoader:

	def __init__(self, path_to_files):
		self.path_to_files = path_to_files
		self.data_dict = {}
		self.data_dict_year = {}
		self.data_dict_year_gb_date_mean = {}
		self.data_dict_year_day = {}
		self.df_gb_datetime = pd.DataFrame({})
		self.df_gb_date_mean = pd.DataFrame({})
		self.df_gb_date_mean_diff = pd.DataFrame({})
		self.df_gb_date_max = pd.DataFrame({})
		self.df_gb_datetime_day = pd.DataFrame({})
		self.df_gb_date_mean_day = pd.DataFrame({})
		self.df_gb_date_mean_day_diff = pd.DataFrame({})
		self.df_gb_date_max_day = pd.DataFrame({})

	"""
	Функция конветирует .csv файлы в pd.DataFrame и собирает их в dict
	"""

	def prepare_dict(self):
		data_dir = self.path_to_files
		dir_content = os.listdir(data_dir)
		csv_files_list = [x for x in dir_content if ('.csv' in x)]

		for file_name in csv_files_list:
			self.data_dict[file_name[:-4]] = pd.read_csv(data_dir+'/'+file_name)


	"""
	Function converts initial date and time to Timestamp format and delete the previous columns
	drop columns of locations that are not included into analysis
	also drop the rows with NaN date or time
	"""

	def process_data(self, columns_to_save=[]):
		for i in self.data_dict:
		    if (('Datum_Uhrzeit' in self.data_dict[i].keys()) | ('Datum Uhrzeit' in self.data_dict[i].keys())):
		        if ('Datum Uhrzeit' in self.data_dict[i].keys()):
		            self.data_dict[i].rename(columns={'Datum Uhrzeit':'Datum_Uhrzeit'}, inplace=True)
		    else:
		        self.data_dict[i]['Datum_Uhrzeit'] = self.data_dict[i]['Datum'] + ' ' + self.data_dict[i]['Uhrzeit']
		    if (i == 'Gesamttabelle_LEAP_2022_T'):
		    	self.data_dict[i] = self.data_dict[i].iloc[0:14184]
		
		for i in self.data_dict:
			if ('Volksgarten' in i):
				self.data_dict[i]['datetime'] = self.data_dict[i]['Datum_Uhrzeit'].apply(convert_date_time_to_datetime_Volksgarten)
			else:
				self.data_dict[i]['datetime'] = self.data_dict[i]['Datum_Uhrzeit'].apply(convert_date_time_to_datetime)

		all_columns = list(set(sum([list(self.data_dict[i].keys()) for i in self.data_dict],[])))
		columns_to_drop = list(set(all_columns) - set(columns_to_save))

		for i in self.data_dict:
		    for j in self.data_dict[i].keys():
		        if (j in columns_to_drop):
		            self.data_dict[i] = self.data_dict[i].drop(columns=[j])

		for i in self.data_dict:
			if ('Volksgarten' in i):
				self.data_dict[i] = self.data_dict[i].groupby('datetime', as_index=False).mean()

	"""
	Возвращает dict с обработанными данными
	"""

	def get_dict(self):
		return self.data_dict
	
	"""
	Еще одна обработка данных и собирание их в dict по годам (2020, 2021, 2022)
	"""

	def prepare_dict_year(self):
		V1_2021_req = ((self.data_dict['T_Volksgarten-1']['datetime']>=parser.parse('01.01.2021')) & (self.data_dict['T_Volksgarten-1']['datetime']<=parser.parse('31.12.2021 23:59:59')))
		V1_2022_req = ((self.data_dict['T_Volksgarten-1']['datetime']>=parser.parse('01.01.2022')) & (self.data_dict['T_Volksgarten-1']['datetime']<=parser.parse('31.12.2022 23:59:59')))
		V2_2021_req = ((self.data_dict['T_Volksgarten-2']['datetime']>=parser.parse('01.01.2021')) & (self.data_dict['T_Volksgarten-2']['datetime']<=parser.parse('31.12.2021 23:59:59')))
		V2_2022_req = ((self.data_dict['T_Volksgarten-2']['datetime']>=parser.parse('01.01.2022')) & (self.data_dict['T_Volksgarten-2']['datetime']<=parser.parse('31.12.2022 23:59:59')))
		V2_2021_data = self.data_dict['T_Volksgarten-2'][V2_2021_req].set_index(['datetime'])
		V2_2022_data = self.data_dict['T_Volksgarten-2'][V2_2022_req].set_index(['datetime'])
		V1_2021_data = self.data_dict['T_Volksgarten-1'][V1_2021_req].set_index(['datetime'])
		V1_2022_data = self.data_dict['T_Volksgarten-1'][V1_2022_req].set_index(['datetime'])
		data_dict_tmp = self.data_dict.copy()
		for i in data_dict_tmp:
			if ('Volksgarten' not in i):
				data_dict_tmp[i] = data_dict_tmp[i].set_index(['datetime'])
		data_2020 = data_dict_tmp['Gesamttabelle_LEAP_2020_T']
		data_2021 = pd.concat([data_dict_tmp['Gesamttabelle_LEAP_2021_T'],V1_2021_data,V2_2021_data],axis=1)
		data_2022 = pd.concat([data_dict_tmp['Gesamttabelle_LEAP_2022_T'],V1_2022_data,V2_2022_data],axis=1)
		self.data_dict_year = {'2020':data_2020,'2021':data_2021,'2022':data_2022}
		for i in self.data_dict_year:
			self.data_dict_year[i]['datetime'] = self.data_dict_year[i].index
			self.data_dict_year[i]['date'] = self.data_dict_year[i]['datetime'].apply(convert_datetime_to_date)
			self.data_dict_year[i]['time'] = self.data_dict_year[i]['datetime'].apply(convert_datetime_to_time)
			for j in self.data_dict_year[i].keys():
				if (('date' not in j) & ('time' not in j) & (self.data_dict_year[i][j].dtype != 'float64')):
					self.data_dict_year[i][j] = self.data_dict_year[i][j].astype(float)

	"""
	гетер для dict по годам (2020, 2021, 2022)
	"""

	def get_dict_year(self):
		return self.data_dict_year
	
	"""
	соединяет данный в один pd.DataFrame и ставит datetime в качетсве индекса
	"""

	def prepare_df_gb_datetime(self):
		self.df_gb_datetime = pd.concat(self.data_dict_year)
		self.df_gb_datetime.reset_index(level=0, inplace=True)
		self.df_gb_datetime.drop(columns=['level_0'], inplace=True)
	
	"""
	гетер для этого pd.DataFrame
	"""

	def get_df_gb_datetime(self):
		return self.df_gb_datetime
	
	"""
	группирует данные по дате (беря среднее), и ставит дату в качетсве индекса
	"""

	def prepare_df_gb_date_mean(self):
		dict_year_gb_date_mean = {}
		for i in self.data_dict_year:
			dict_year_gb_date_mean[i] = self.data_dict_year[i].groupby('date').mean()
		self.df_gb_date_mean = pd.concat(dict_year_gb_date_mean)
		self.df_gb_date_mean.reset_index(level=0, inplace=True)
		self.df_gb_date_mean = self.df_gb_date_mean.drop(columns=['level_0'])
		
	"""
	гетер для этого pd.DataFrame
	"""

	def get_df_gb_date_mean(self):
		return self.df_gb_date_mean
	
	"""
	Возвращает pd.DataFrame с разностью среднедневных температур для Opel Bauer и локаций данных в list locations
	"""
	
	def get_df_gb_date_mean_diff(self, locations=[]):
		for loc in list(set(locations) - set(['Opel Bauer'])):
			self.df_gb_date_mean_diff[loc] = self.df_gb_date_mean.apply(lambda x: x['Opel Bauer'] - x[loc], axis=1)
		return self.df_gb_date_mean_diff
	
	"""
	группирует данные по дате (беря максимальную температуру), и ставит дату в качетсве индекса
	"""
	
	def prepare_df_gb_date_max(self):
		dict_year_gb_date_max = {}
		for i in self.data_dict_year:
			dict_year_gb_date_max[i] = self.data_dict_year[i].groupby('date').max()
		self.df_gb_date_max = pd.concat(dict_year_gb_date_max)
		self.df_gb_date_max.reset_index(level=0, inplace=True)
		self.df_gb_date_max = self.df_gb_date_max.drop(columns=['level_0'])

	"""
	гетер для этого pd.DataFrame
	"""

	def get_df_gb_date_max(self):
		return self.df_gb_date_max
	
	# --------------------------------------------------- DAY (12:00 to 17:00) ---------------------------------------------------

	"""
	Все то же самое только для данных не с 00 до 24, а для данных с 12 до 17
	"""

	def prepare_dict_year_day(self):
		for i in self.data_dict_year:
			day_req = ((self.data_dict_year[i]['time']>=parser.parse('12:00:00').time()) & (self.data_dict_year[i]['time']<=parser.parse('17:00:00').time()))
			self.data_dict_year_day[i] = self.data_dict_year[i][day_req]

	def get_dict_year_day(self):
		return self.data_dict_year_day
	
	def prepare_df_gb_datetime_day(self):
		self.df_gb_datetime_day = pd.concat(self.data_dict_year_day)
		self.df_gb_datetime_day.reset_index(level=0, inplace=True)
		self.df_gb_datetime_day.drop(columns=['level_0'], inplace=True)
	
	def get_df_gb_datetime_day(self):
		return self.df_gb_datetime_day
	
	def prepare_df_gb_date_mean_day(self):
		dict_year_gb_date_mean = {}
		for i in self.data_dict_year_day:
			dict_year_gb_date_mean[i] = self.data_dict_year_day[i].groupby('date').mean()
		self.df_gb_date_mean_day = pd.concat(dict_year_gb_date_mean)
		self.df_gb_date_mean_day.reset_index(level=0, inplace=True)
		self.df_gb_date_mean_day = self.df_gb_date_mean_day.drop(columns=['level_0'])

	def get_df_gb_date_mean_day(self):
		return self.df_gb_date_mean_day
	
	def get_df_gb_date_mean_day_diff(self, locations=[]):
		for loc in list(set(locations) - set(['Opel Bauer'])):
			self.df_gb_date_mean_day_diff[loc] = self.df_gb_date_mean_day.apply(lambda x: x['Opel Bauer'] - x[loc], axis=1)
		return self.df_gb_date_mean_day_diff
	
	def prepare_df_gb_date_max_day(self):
		dict_year_gb_date_max = {}
		for i in self.data_dict_year_day:
			dict_year_gb_date_max[i] = self.data_dict_year_day[i].groupby('date').max()
		self.df_gb_date_max_day = pd.concat(dict_year_gb_date_max)
		self.df_gb_date_max_day.reset_index(level=0, inplace=True)
		self.df_gb_date_max_day = self.df_gb_date_max_day.drop(columns=['level_0'])

	def get_df_gb_date_max_day(self):
		return self.df_gb_date_max_day



def calc_SST(df,mxg):
    df_flatten = df.to_numpy().flatten()
    SST = np.sum((df_flatten - mxg)**2)
    return SST

def calc_SSW(df,pars_df):
    locations = pars_df['location']
    SSW = 0
    for loc in locations:
        mean = pars_df[pars_df['location']==loc]['mean'].values[0]
        SSW += np.sum((df[loc].to_numpy() - mean)**2)
    return SSW

def calc_SSB(df,pars_df,mxg):
    locations = pars_df['location']
    SSB = 0
    for loc in locations:
        mean = pars_df[pars_df['location']==loc]['mean'].values[0]
        SSB += df.shape[0]*(mean - mxg)**2
    return SSB

def calc_F(SSB,SSW,dof_groups,dof_general):
    F = (SSB/dof_groups)/(SSW/dof_general)
    return F

class DataAnalyzer:

	def __init__(self, df):
		self.df = df
		self.data = pd.DataFrame({})
		self.data_dropped = pd.DataFrame({})
		self.data_params = pd.DataFrame({})
		self.data_dropped_params = pd.DataFrame({})
		self.locs_p_value_dict = {}
		self.locs_p_value_sign_dict = {}
		self.comb_p_value_list = []

	def get_data(self,date_from,date_to):
		self.data = self.df[parser.parse(date_from).date():parser.parse(date_to).date()]
		return self.data
	
	def get_data_dropped(self):
		self.data_dropped = self.data
		for j in range(0,len(self.data.isna().sum())):
			if self.data.isna().sum().values[j] != 0:
				col_to_drop = list(self.data.isna().sum().keys())[j]
				self.data_dropped = self.data_dropped.drop(columns=[col_to_drop])
		return self.data_dropped

	def get_data_params(self):
		data_params_dict = {'location': list(self.data.keys()),
							'mean': list(self.data.mean()),
                    		'std': list(self.data.std()),
							'N': np.ones(self.data.shape[1])*[len(self.data[i].dropna()) for i in self.data],
							'se':list(self.data.std()/np.sqrt([len(self.data[i].dropna()) for i in self.data]))}
		self.data_params = pd.DataFrame(data_params_dict)
		return self.data_params
	
	def get_data_dropped_params(self):
		data_dropped_params_dict = {'location': list(self.data_dropped.keys()), 
                           			'mean': list(self.data_dropped.mean()),
                           			'std': list(self.data_dropped.std()),
                           			'N': np.ones(self.data_dropped.shape[1])*len(self.data_dropped)}
		
		self.data_dropped_params = pd.DataFrame(data_dropped_params_dict)
		return self.data_dropped_params
	
	def get_ANOVA_F_p_value(self):
		mean_x_general = self.data_dropped.sum().sum()/(len(self.data_dropped)*self.data_dropped.shape[1])
		SST = calc_SST(self.data_dropped,mean_x_general)
		SSW = calc_SSW(self.data_dropped,self.data_dropped_params)
		SSB = calc_SSB(self.data_dropped,self.data_dropped_params,mean_x_general)
		dof_groups = self.data_dropped_params.shape[0] - 1
		dof_general = self.data_dropped.shape[0]*self.data_dropped.shape[1] - self.data_dropped_params.shape[0]
		F = calc_F(SSB,SSW,dof_groups,dof_general)
		p_value = 1 - stats.f.cdf(F,dof_groups,dof_general)
		return F, p_value
	
	def get_TUKEY_results(self):
		tukey_results = stats.tukey_hsd(*[self.data_dropped[i].values for i in self.data_dropped])
		print(tukey_results)
		return tukey_results
	
	def get_TUKEY_matrix(self):
		tukey_results = stats.tukey_hsd(*[self.data_dropped[i].values for i in self.data_dropped])
		tukey_results_df = pd.DataFrame(tukey_results.pvalue,index=self.data_dropped.keys(),columns=self.data_dropped.keys())
		return tukey_results_df.style.background_gradient(axis=0,cmap='RdYlBu')
	
	def plot_mean_std(self):
		fig, ax = plt.subplots(figsize=(12, 8))
		sns.set(style="white", font_scale=1.4)
		ax.boxplot(self.data_dropped)
		ax.set_xticklabels(list(self.data_dropped.keys())) 
		plt.xticks(rotation=30)
		plt.ylabel('mean T,'+u'\N{DEGREE SIGN}C')
		plt.tight_layout()
		plt.show()

	def describe_dropped_data(self):
		return self.data_dropped.describe().T
	
	def make_paired_t_test(self,threshold=0.05):
		comb_list = []
		for i in self.data_params.index:
			for j in self.data_params.index:
				if ((i==j) | ([i,j] in comb_list) | ([j,i] in comb_list)): continue
				comb_list.append([i,j])

		for (i,j) in comb_list:
			t_value, p_value = stats.ttest_ind(self.data[self.data.keys()[i]].dropna(),self.data[self.data.keys()[j]].dropna())
			self.locs_p_value_dict[self.data_params.iloc[i]['location']+' vs '+self.data_params.iloc[j]['location']] = np.around(p_value,4)
			self.comb_p_value_list.append([i,j,p_value])
			if p_value < threshold:
				self.locs_p_value_sign_dict[self.data_params.iloc[i]['location']+' vs '+self.data_params.iloc[j]['location']] = np.around(p_value,4)