#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# 
# There are given two datasets. 
# 
# One of them is "clicked items investigation" and the other one is "session investigation".
# 
# First of all, i should write my methodology and solution approach.
# 
# I follow five instruction to solve data science problems:
# 
#     - Define the problem
# 
#     - Prepare Data / Data Preprocessing
#         Get Data
#         Data Cleaning/Wrangling
#         Statistical Analysis
#         Data Visualization
#         Feature Selection/Scaling
#         Data Transformation
# 
#     - Check Algorithms
#         Train & Test Data
#         Apply ML Algorithm
#         Test
#         Perform Measure
#         Evaulate accuarcy of different algorithm
# 
#     - Improve Results
#         Algorithm Tuning
# 
#     - Present Results
#         Conclusion
#         Presentation

# ## DEFINITION THE PROBLEM
# 
# We have some information about the items(hotels) and we will try to solve questions using data_analysis_case_study_part1.csv and data_analysis_case_study_part2.csv datasets.
# 
# The aim is to analyze customer click-out behaivour. On the other hand, one of the most important point is click through rates. These informations helps us to evaulate the hotel performance, provide insight for ranking, in addition to that we can predict other hotels might be interesting for the end users. 

# ### 1-CLICKED ITEM INVESTIGATION
# 
# For this section we will try to find answer following questions the below.
# 
#     1-Calculate the CTR of each item. What is the overall avg CTR?
# 
#     2-What is the distribution of clicks among the top 25 positions? What is the share of the first positions? On how many positions are approx. Half of the click-outs made?
# 
#     3-Describe the relationship between the average displayed position and the clicked displayed position. What are your thoughts about the variance between the two?
# 
#     4-In the dataset, we provided you with the average displayed position. What can be wrong with using averages?
# 
# So let's start data analysis...

# In[1]:


# Import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
import math as m
import scipy.stats as sct
#from scipy import stats as st

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
sns.set()


# In[2]:


# Log
import logging
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='data_analyst_search.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


# ###### 1-Calculate the CTR of each item. What is the overall avg CTR?

# In[3]:


# Get data
file = 'data_analysis_case_study_part1.csv'
data = pd.read_csv(file)

data.head(5)
data.shape

logging.debug('data received...')


# In[4]:


# Data types and size
data.info()


# In[5]:


# check negative values; if it is exist, update or ignore these values

#data['impressions']  = np.where((data['impressions']  < 0), 0, data['impressions'])

neg_imp = set(data.loc[data.impressions < 0, 'impressions'])
neg_cli = set(data.loc[data.clicks < 0, 'clicks'])
neg_avg = set(data.loc[data.avg_impressed_position < 0, 'avg_impressed_position'])
neg_user = set(data.loc[data.num_users < 0, 'num_users'])
neg_sess = set(data.loc[data.num_sessions < 0, 'num_sessions'])

print("impression              : ", neg_imp)
print("clicks                  : ", neg_cli)
print("avg_impressed_position  : ", neg_avg)
print("num_users               : ", neg_user)
print("num_sessions            : ", neg_sess)

logging.debug('check negative values')


# In[6]:


# check null values
check_null_columns = data.isnull().sum()
print('**************************')
print('Check Null Values Column by Column')
print(check_null_columns)
print('count of null values          ', data.isnull().values.sum())
print('**************************')

logging.debug('check null values')


# In[7]:


# check unique hotel id
data[data.duplicated(subset=['item_id'],keep=False)]

logging.debug('check unique hotel')


# In[8]:


# some statistical information
data.describe()


# ###### CTR of each item and overall avg CTR?

# In[9]:


# CTR
# overall CTR

data['CTR'] = (data.clicks)/(data.impressions)
data['CTR_overall'] = sum(data.clicks)/sum(data.impressions)

data.head(5)

logging.debug('CTR and overall CTR')


# ###### 2-What is the distribution of clicks among the top 25 positions? What is the share of the first positions? On how many positions are approx. Half of the click-outs made?

# In[10]:


# convert from semicolon to list
clicked_distribution = pd.Series(data.clicked_displayed_positions.str.cat(sep=';').split(';')).value_counts()
clicked_distribution

logging.debug('parsing semicolon data')


# In[11]:


# we expect that the positions should be between 0 and 24. 
# so we need to remove -11 position, because this data is not valid for our analysis.
clicked_distribution = clicked_distribution.drop(labels=['-11'])
clicked_distribution

logging.debug('cleaning the unexpected positions')


# ###### What is the share of the first positions?

# In[12]:


# distribution of clicks
print(" ")
print(" ")
print ('please find distribution of clicks from the pie chart below!')

labels = clicked_distribution.index
sizes = clicked_distribution.values
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

fig = plt.figure(figsize=(10, 10))
_=plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=180)
_=plt.axis('equal')
plt.show();

logging.debug('distribution of positions')


# ###### On how many positions are approx. Half of the click-outs made?

# In[13]:


# share of the first position
# half of the click-outs made

print ('share of the first position     : ', '% 31.1')
print ('half of the click-outs made     : ', 'First three positions= 0,1,2')


# ###### 3-Describe the relationship between the average displayed position and the clicked displayed position. What are your thoughts about the variance between the two?
# 

# In[14]:


# thanks to some range setting assumption, we can find total clicks according to average displayed position.
    
    # 0.0-0.5 = 0 position
    # 0.5-1.5 = 1 position
    # 1.5-2.5 = 2 position 
    # ...
    # 22.5-23.5 = 23 position
    # 23.5-24.0 = 24 position

impressed_distribution = pd.Series(data['clicks'][data['avg_impressed_position'] <= 0.5].sum())

for i in range(1,24,1):
    impressed_distribution = impressed_distribution.append(pd.Series(data['clicks'][((data['avg_impressed_position'] >= (i-0.5)) & 
                                                                            (data['avg_impressed_position'] <= (i+0.5)))].sum()), 
                                                  ignore_index=True)
    
impressed_distribution = impressed_distribution.append(pd.Series(data['clicks'][data['avg_impressed_position'] >= 23.5].sum()), 
                                               ignore_index=True)
impressed_distribution


# In[15]:


# number of clicks vs clicked displayed position
fig = plt.figure(figsize=(12,5))
_=plt.bar(clicked_distribution.index, clicked_distribution.values, color='red')
_=plt.xlabel('clicked_displayed_position')
_=plt.ylabel('number_of_clicks')
_=plt.title('clicks' + ' & displayed_position')
plt.show();

logging.debug('clicks & avg impressed position')


# avg impressions vs clicks
fig = plt.figure(figsize=(12,5))
_=plt.bar(impressed_distribution.index, impressed_distribution.values, color='green')
_=plt.xlabel('avg_impressed_position')
_=plt.ylabel('number_of_clicks')
_=plt.title('clicks' + ' & avg_impressed_position')
plt.show();

logging.debug('clicks & displayed position')


# step one- Describe the relationship between the avg_impressed_position and the clicked displayed position.
fig = plt.figure(figsize=(12,5))
p1, = plt.plot(impressed_distribution.index, impressed_distribution.values, color='green')
p2, = plt.plot(clicked_distribution.index, clicked_distribution.values, color='red')
_=plt.legend([p1, p2], ['avg_imp','click_pos'], loc='best')
_=plt.xlabel('positions')
_=plt.ylabel('clicked_position / avg_impressed')
_=plt.title('clicked_position & avg_impressed')
plt.show();

logging.debug('clicked position & avg impressed position')


# In[16]:


# step two- Describe the relationship between the avg_impressed_position and the clicked displayed position.

impressed_distribution.index = impressed_distribution.index.astype(int)
impressed_distribution.sort_index(inplace=True)

clicked_distribution.index = clicked_distribution.index.astype(int)
clicked_distribution.sort_index(inplace=True)

corr_map = {'avg_impressed' : impressed_distribution, 'clicked_displayed' : clicked_distribution}
corr_df = pd.DataFrame(corr_map)

pd.set_option('precision', 5)
correlations = corr_df.corr(method='pearson')

print('')
print('')
print('')
print('*********************')
print('correlation table')
corr_df.head(5)

print('')
print('')
print('')
print('*********************')
print('correlation coefficient')
correlations

print('')
print('')
print('')
colormap = sns.diverging_palette(210, 20, as_cmap=True)
_=sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
_=plt.xticks(range(len(correlations.columns)), correlations.columns)
_=plt.yticks(range(len(correlations.columns)), correlations.columns)
plt.show();

logging.debug('correlation')


# ###### What are your thoughts about the variance between the two?

# This analysis shows us clearly that there is not any direct relationship between "impressions" and "clicks".
# 
# Moreover, we can say that if you would like to increase the number of clicks you should stay in the first 3 positions on the list.
# 
# On the other hand, one of the important value is the impression. 
# But impression says that your position should be between 6 and 14 for many clicks.
# 
# As a result, we should think together, both clicked position and avg impressed position.
# 
# Your position can be at a different location on the list, but in general, it is in the first 3 positions when the hotel is clicked by users.
# 
# Your position depends on some filters or searches criteria.
# 
# In this time, your click chance is decreasing because of your avg impressed position calculating.
# 
# The last one, you should focus your "currently position" not "average impressed position". 
# 
# Because you can slip down on the list and your avg impressed position can be changed, but it is normal and not vital.

# ###### 4-In the dataset, we provided you with the average displayed position. What can be wrong with using averages?

#  One of the most important questions that are how and why to calculate the avg result.
# 
#  We analyzed avg output the above and this information is not helpful for the next step.
# 
#  Because each customer wants to choose different features and filters.
# 
#  Then sometimes your position drops behind because of ranking or filters, but this not mean you will not be selected by the customers.
#  
#  Your "avg impressed position" can be increased because of all these factors. But when you are placed near the top, your click possibility can increase and not affects from avg impressed position.

# ### 2-SESSION INVESTIGATION
# 
# For this section we will try to find answer following questions the below.
# 
#     1-Describe the data set that you have received. Calculate the 5 most frequent values per column (with frequency). Can you find any suspicious results? If so, what are they? And how would you fix these for the analysis?
# 
#     2-Which search type has the lowest average displayed position? What is the best sorting order for this search type? Which search type should be excluded for a statistical reason?
# 
#     3-What are the top 10 “best” and “worst” performing items? Explain what metric you have chosen to evaluate the performance and why.
# 
#     4-Describe and visualise the relationship between the average displayed position and the CTR among the top 1000 most clicked items.
#     
# So let's start data analysis...

# ###### 1-Describe the data set that you have received. Calculate the 5 most frequent values per column (with frequency). Can you find any suspicious results? If so, what are they? And how would you fix these for the analysis?

# In[17]:


# Get data
file2 = 'data_analysis_case_study_part2.csv'
big_data = pd.read_csv(file2, low_memory=False)

big_data.head(5)
big_data.shape

logging.debug('big data received...')


# In[18]:


# Data types and size
big_data.info()


# In[19]:


# some statistical informations
big_data.describe()


# In[20]:


# check null values
check_null_columns_2 = big_data.isnull().sum()
print('**************************')
print('Check Null Values Column by Column')
print(check_null_columns_2)
print('**************************')
print('count of null values          ', big_data.isnull().values.sum())
print('**************************')

logging.debug('check null values')


# In[21]:


# delete null values
print('**************************')
print('before :')
big_data[big_data['session_id'].isnull()]
big_data = big_data.drop([1903795])
print("")
print('**************************')
print('after :')
big_data[big_data['session_id'].isnull()]

logging.debug('remove null values')


# In[22]:


# check negative values; if it is exist, update or ignore these values

#data['sess_id']  = np.where((data['sess_id']  < 0), 0, data['sess_id'])

neg_sess_id    = set(big_data.loc[big_data.session_id < 0, 'session_id'])
neg_clic_it_id = set(big_data.loc[big_data.clicked_item_id < 0, 'clicked_item_id'])
neg_disp_pos   = set(big_data.loc[big_data.displayed_position < 0, 'displayed_position'])
neg_pg_num     = set(big_data.loc[big_data.page_num < 0, 'page_num'])
neg_srt_ord    = set(big_data.loc[big_data.sort_order < 0, 'sort_order'])
neg_src_type   = set(big_data.loc[big_data.search_type < 0, 'search_type'])
neg_pth_id     = set(big_data.loc[big_data.path_id < 0, 'path_id'])
neg_arr_days   = set(big_data.loc[big_data.arrival_days < 0, 'arrival_days'])
neg_dpt_days   = set(big_data.loc[big_data.departure_days < 0, 'departure_days'])
neg_trf_typ    = set(big_data.loc[big_data.traffic_type < 0, 'traffic_type'])

print("session_id              : ", neg_sess_id)
print("clicked_item_id         : ", neg_clic_it_id)
print("displayed_position      : ", neg_disp_pos)
print("page_num                : ", neg_pg_num)
print("sort_order              : ", neg_srt_ord)
print("search_type             : ", neg_src_type)
print("path_id                 : ", neg_pth_id)
print("arrival_days            : ", neg_arr_days)
print("departure_days          : ", neg_dpt_days)
print("traffic_type            : ", neg_trf_typ)
print("************************")
print("")
big_data[((big_data['displayed_position'] < 0) | (big_data['displayed_position'] > 24)) & (big_data['displayed_position'] != -11)]

logging.debug('check negative values')


# ###### Calculate the 5 most frequent values per column (with frequency)

# In[23]:


# Calculate the 5 most frequent values per column (with frequency)

big_data.user_id = big_data.user_id.astype(float)

impressed_item_ids_distr = pd.Series(big_data["impressed_item_ids "].str.cat(sep=';').split(';')).value_counts()
impressed_item_ids_distr.index = impressed_item_ids_distr.index.astype(int)

for c in big_data.columns:
    if c == "impressed_item_ids ":
        continue    
    print('*********************')
    print(c)
    big_data[c].value_counts().nlargest(5)
    
print('*********************')
print('impressed_item_ids ')
impressed_item_ids_distr.head(5)

logging.debug('frequency analysis')


# ###### Can you find any suspicious results? If so, what are they? And how would you fix these for the analysis?

# In[24]:


# displayed_position should be in between 0 and 24, so -11 is incorrect, 
# market share should be updated.
# we have enough data for analysis and -11 position data is not big, so no big affect to our analysis.

# On the other way if you want to use -11 position data, you need to find a pattern for each position.
# after that, you should change -11 position data with the most matched pattern values.

_=plt.hist(big_data.displayed_position)
_=plt.title("displayed_position")
_=plt.xlabel("position")
_=plt.ylabel("frequency")
plt.show();


# In[25]:


# min arrival_days should be 0, the negative values are meaningless. 
# -1 value frequency is very small into the arrival_days column.
# Then it does not affect our analysis and I can ignore and also I don't need to use it for analysis.

_=plt.hist(big_data.arrival_days)
_=plt.title("arrival days")
_=plt.xlabel("days")
_=plt.ylabel("frequency")
plt.show();


# In[26]:


# min departure_days should be 0, the negative values are meaningless. 
# -1000000 value frequency is too small.
# Then it does not affect our analysis and I can ignore and also I don't need to use it for analysis.

# but if you want to use this data, you can do analysis with other columns and try to find a pattern by this means 
# you can update the correct value. For example, departure_days = -1000000 we can use 0 instead of -1000000.

# in addition to that, you can create insight based on user behavior. 
# thanks to behavior analysis, you can change with an ideal value

_=plt.hist(big_data.departure_days)
_=plt.title("departure days")
_=plt.xlabel("days")
_=plt.ylabel("frequency")
plt.show();


# ###### 2-Which search type has the lowest average displayed position? What is the best sorting order for this search type? Which search type should be excluded for a statistical reason?

# In[27]:


# average displayed position according to search type
search_type = big_data[(big_data.displayed_position != -11)].groupby('search_type').agg({"displayed_position": "sum", 
                                                                                               "search_type": "count"})
search_type.columns = ['sum_displayed_position', 'count_search_type']
search_type["average_displayed_position"] = search_type.sum_displayed_position/search_type.count_search_type
search_type.sort_values(by=["average_displayed_position"])

logging.debug('calculate average displayed position according to search type')


# ###### What is the best sorting order for this search type?

# In[28]:


# best sorting order according to this search type
sorting_order = big_data[(big_data.displayed_position != -11) & (big_data.search_type == 2116)].groupby('sort_order').agg({"sort_order": "count"})
sorting_order.columns = ['count_sort_order']
sorting_order.sort_values(by=["count_sort_order"], ascending=False)

logging.debug('calculate best sorting order according to this search type')


# ###### Which search type should be excluded for a statistical reason?

# In[29]:


# Normally, we can decide to exclude search_id=2100. 
# Because the search is done for two times using this type by users. It is too small.
# But we have to demonstrate as statistically.
# I want to use a confidence interval for deciding the importance level.

# check current dataset distribution
_=sns.distplot(search_type.count_search_type, 
               hist=True, kde=True, color='darkblue', 
               hist_kws={'edgecolor':'black'}, 
               kde_kws={'linewidth': 3})


# In[30]:


# I apply the central limit theorem to convert it to Gauss distribution 
# because now our dataset doesn't have a normal distribution.

bs_search  = np.array([])
for i in range(100000):
    bs_search  = np.append(bs_search, np.mean(np.random.choice(search_type.count_search_type, 
                                                                 replace=True, size=len(search_type))))

print("")
print("")
conf_int_search = np.percentile(bs_search, [2.5, 97.5])
print("confidence interval for p=0.05 : ",conf_int_search)
print("")
print("search_type_2100_frequency     : ",2)
print("")
print("search_type_2113_frequency     : ",854294)
print("")
print("")
print("search_type_2100 and search_type_2113 are out of confidence interval, because of their frequency")
print("")
print("bigger frequency is the desired result, the opposite way smaller frequency is the undesired result")
print("")
print("big frequency means that this search type is used by the user, otherwise small frequency is not")
print("")
print("")
print("so we should exclude search_type_2100")
print("")
print("")

# plot normal distributed dataset
_=sns.distplot(bs_search, 
               hist=True, kde=True, color='darkblue', 
               hist_kws={'edgecolor':'black'}, 
               kde_kws={'linewidth': 3})
_=plt.axvline(conf_int_search[0], color='red', linestyle='dashed', linewidth=2)
_=plt.axvline(conf_int_search[1], color='red', linestyle='dashed', linewidth=2)
plt.show()


# ###### 3-What are the top 10 “best” and “worst” performing items? Explain what metric you have chosen to evaluate the performance and why.

#     -Best performed hotel is the most chosen hotel. This means that is many people have chosen it.
#     -So, I think that the most important criteria are the number of clicks.
#     -Contrary to best performance criteria, the worst performance hotel is the least chosen hotel. 
# 
#     -But we have many hotels one-clicked. Therefore, we should do extra analysis together with other columns.
#     -For example, all columns values seem perfect but it still is one-clicked, so in this case, shows us this type of hotels is worst than the other one-clicked hotels. For this analysis, we should use "displayed_position" and "page number". In addition to that, we can use "search type" and "sort_order" values.
# 
#     -I calculated the coefficient for each column in order to measure hotel performance.

# In[31]:


# the top 10 best and worst hotels

# calculate displayed_position coefficient
disp = pd.DataFrame(big_data[(big_data.displayed_position != -11)].groupby('displayed_position').agg({"displayed_position": "count"}))
disp.columns=["displayed_position_count"]
disp_coeff = disp["displayed_position_count"].sum()
disp["disp_coefficient"] = disp_coeff / disp["displayed_position_count"]
disp.index = disp.index + 1

# calculate page_num coefficient
page = pd.DataFrame(big_data[(big_data.displayed_position != -11)].groupby('page_num').agg({"page_num": "count"}))
page.columns=["page_num_count"]
page_coeff = page["page_num_count"].sum()
page["page_coefficient"] = page_coeff / page["page_num_count"]
page.index = page.index + 1

# calculate sort_order coefficient
sort = pd.DataFrame(big_data[(big_data.displayed_position != -11)].groupby('sort_order').agg({"sort_order": "count"}))
sort.columns=["sort_order_count"]
sort_coeff = sort["sort_order_count"].sum()
sort["sort_coefficient"] = sort["sort_order_count"] / sort_coeff
sort.index = sort.index + 1

# calculate search_type coefficient
search = pd.DataFrame(big_data[(big_data.displayed_position != -11)].groupby('search_type').agg({"search_type": "count"}))
search.columns=["search_type_count"]
search_coeff = search["search_type_count"].sum()
search["search_coefficient"] = search["search_type_count"] / search_coeff

clicked_hotel = big_data
clicked_hotel.drop(clicked_hotel[clicked_hotel['displayed_position'] == -11].index , inplace=True)

clicked_hotel[["displayed_position", "page_num", "sort_order"]] = pd.DataFrame(clicked_hotel[["displayed_position", 
                                                                                              "page_num","sort_order"]].add(1))

clicked_hotel = pd.merge(clicked_hotel, disp[["disp_coefficient"]], on="displayed_position", how="left")
clicked_hotel = pd.merge(clicked_hotel, page[["page_coefficient"]], on="page_num", how="left")
clicked_hotel = pd.merge(clicked_hotel, search[["search_coefficient"]], on="search_type", how="left")
clicked_hotel = pd.merge(clicked_hotel, sort[["sort_coefficient"]], on="sort_order", how="left")

clicked_hotel["performance_value"] = (clicked_hotel[("disp_coefficient")] * 
                                      clicked_hotel[("page_coefficient")] * 
                                      clicked_hotel[("search_coefficient")] * 
                                      clicked_hotel[("sort_coefficient")])

clicked_hotel = clicked_hotel.groupby('clicked_item_id').agg({"displayed_position": "sum",
                                                              "page_num": "sum",
                                                              "clicked_item_id": "count",
                                                              "performance_value": "sum",
                                                              "search_type": "sum",
                                                              "sort_order": "sum"})

total_performance = clicked_hotel.performance_value.sum()

clicked_hotel.columns = ["sum_displayed_position", "sum_page_num", "count_item", 
                         "performance", "sum_search_type", "sum_sort_order"]
clicked_hotel["average_page_number"] = clicked_hotel.sum_page_num/clicked_hotel.count_item
clicked_hotel["average_displayed_position"] = clicked_hotel.sum_displayed_position/clicked_hotel.count_item
clicked_hotel["average_search_type"] = clicked_hotel.sum_search_type/clicked_hotel.count_item
clicked_hotel["average_sort_order"] = clicked_hotel.sum_sort_order/clicked_hotel.count_item
clicked_hotel["performance"] = clicked_hotel.performance / total_performance

# the top 10 best hotels
print("")
print("")
print("**********************")
print("")
print("")
print("the top 10 best hotels")
clicked_hotel.sort_values(by=["performance"], ascending=False).head(10)

# the top 10 worst hotels
print("")
print("")
print("**********************")
print("")
print("")
print("the top 10 worst hotels")
clicked_hotel.sort_values(by=["performance"], ascending=True).head(10)

logging.debug('find the top 10 best and worst hotels')


# ###### 4-Describe and visualise the relationship between the average displayed position and the CTR among the top 1000 most clicked items.

# In[32]:


# step one-prepare dataframe for description and visualization relationship between avg_disp_pos and CTR

# most_clicked_hotels dataframe
most_clicked_hotels = big_data[(big_data.displayed_position != -11)].groupby('clicked_item_id').agg({"clicked_item_id":"count",
                                                                                                     "displayed_position": "sum"})
most_clicked_hotels.columns = ["count_clicked_items", "sum_displayed_position"]

most_clicked_hotels = (most_clicked_hotels.sort_values(by=["count_clicked_items"], ascending=False).
                       nlargest(columns=["count_clicked_items"], n=1000))

most_clicked_hotels["avg_displayed_position"] = (most_clicked_hotels.sum_displayed_position / 
                                                 most_clicked_hotels.count_clicked_items)

most_clicked_hotels.head(5)

# click through rates dataframe
ctr = impressed_item_ids_distr.to_frame()
ctr.columns = ["count_imp_item"]
ctr.index.names = ['imp_item_id']
ctr.head(5)

# join the dataframes
most_clicked_hotels = most_clicked_hotels.join(ctr, lsuffix='_x', rsuffix='_y')
most_clicked_hotels.head(5)

# calculate CTR
most_clicked_hotels["CTR"] = most_clicked_hotels.count_clicked_items / most_clicked_hotels.count_imp_item
most_clicked_hotels.head(5)


# In[33]:


# step two-describe and visualise relationship between avg_disp_pos and CTR

# correlation coefficient
corr_ctr_disp = {'avg_displayed_position' : most_clicked_hotels.avg_displayed_position, 'ctr' : most_clicked_hotels.CTR}
corr_ctr_disp = pd.DataFrame(corr_ctr_disp)

pd.set_option('precision', 5)
corr_coeff = corr_ctr_disp.corr(method='pearson')

print('')
print('')
print('')
print('*********************')
print('correlation table')
corr_ctr_disp.head(5)

print('')
print('')
print('')
print('*********************')
print('correlation coefficient')
corr_coeff

print('')
print('')
print('')
fig = plt.figure(figsize=(12,5))
_=plt.scatter(corr_ctr_disp.avg_displayed_position, corr_ctr_disp.ctr)
_=plt.xlabel('avg_displayed_position')
_=plt.ylabel('CTR')
_=plt.title('avg_disp_pos & CTR')
plt.show();

print('')
print('')
print('')
colormap = sns.diverging_palette(210, 20, as_cmap=True)
_=sns.heatmap(corr_coeff, cmap=colormap, annot=True, fmt=".2f")
_=plt.xticks(range(len(corr_coeff.columns)), corr_coeff.columns)
_=plt.yticks(range(len(corr_coeff.columns)), corr_coeff.columns)
plt.show();

logging.debug('correlation')

