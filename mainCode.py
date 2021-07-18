"""
@author: jacob
@script: Demography Project Main Code
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#df = pd.read_excel(r"C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\dataset.xlsx", sheet_name = 'data')
df = pd.read_excel(r"C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\recidivism_dataset.xlsx", sheet_name = 'results')

from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines import CoxPHFitter
kmf = KaplanMeierFitter()
naf = NelsonAalenFitter()
cph = CoxPHFitter()


#Main survival function. Before splitting into groups and age_intervals:
#kmf.fit(df['week'], event_observed=df['arrest'])
#kmf.survival_function_.plot()
#plt.title('Survival function of criminal recidivism');
#kmf.plot()

unique_ages = np.sort(df['age'].unique())
interval_youth = unique_ages[0:3]
interval_1 = unique_ages[3:8]
interval_2 = unique_ages[8:14]
interval_3 = unique_ages[14:]

# Making survival tabel based on all the data:
tabelCols = ['Weeks','At Risk','Arrested','Cu. Hazard', 'Survival','Cu. Hazard (Cox)', 'Survival (Cox)']#, 'Cu.Hazard']
jumps = 4
survTab = pd.DataFrame(index=np.arange(0, 53, jumps), columns=tabelCols)
survTab['Weeks'] = np.arange(0, 53, jumps)
kmf.fit(df['week'], event_observed=df['arrest'], timeline=np.arange(0, 53, jumps))
#survTab['Kaplan-Meier'] = kmf.survival_function_
#survTab['Cu. Hazard'] = (kmf.survival_function_ - 1) *(-1) #recidivism_rates
# Most recent one^ - before going with cumulative summation of hazard rates

#survTab['Cu.Hazard'] = -np.log(kmf.survival_function_)
naf.fit(df['week'], event_observed=df['arrest'], timeline=np.arange(0, 53, jumps))

# Nelson Aalen + Kaplan-Meier doesn't exactly sum to 1.0..
#nafTEST = naf.cumulative_hazard_
#kmfTEST = kmf.survival_function_
#np.array(nafTEST) + np.array(kmfTEST)
#survTab['Cu.Hazard'] = naf.cumulative_hazard_
#survTab['Survival'][0] = 1 # 100 % at start
#survTab['Cu.Hazard'][0] = 0
#survTab['Cu.Hazard'][0] = 0


idx = (df['arrest'] == 1)
n_convicts = df.count(axis=0)[0]
arrested = []
at_risk = [n_convicts]
for i, week in enumerate(np.arange(0, 53, jumps)):
    week_idx = df['week'].between(week-jumps,week)
    numb_of_arrests = len(df.loc[idx].loc[week_idx])
    arrested.append(numb_of_arrests)
    if i == 0:
        at_risk_cur = n_convicts
        at_risk.append(at_risk_cur)
    else:
        at_risk_cur = at_risk_cur - numb_of_arrests
        at_risk.append(at_risk_cur)

a_d_list = list(zip(arrested,at_risk))   
hazard_rates = np.array([a/d for a,d in a_d_list])

# Computing cumulative hazard and survival functions 'by hand':
cum_hazard = hazard_rates.cumsum(axis=0)
#one_minus_hazard_rates = np.ones(hazard_rates.shape) - hazard_rates
#survival_func = one_minus_hazard_rates.cumprod(axis=0)
#survival_from_hazard = np.exp(-cum_hazard)

#cph.fit(df['week'], event_observed=df['arrest'])
#cph.fit(dfC, duration_col='week', event_col='arrest')

#survTab['Hazard'] = hazard_rates
#survTab['Cu. Hazard'] = cum_hazard
survTab['Cu. Hazard']= naf.cumulative_hazard_
#survTab['Cu. Hazard'] = -np.log(kmf.survival_function_) # 'NelsonAalen' CU

survTab['Survival'] = kmf.survival_function_
survTab['Arrested'] = arrested
survTab['At Risk'] = at_risk[:-1]

#Cox PH model to survival table
# Creating "dfC", a replicate dataset of df with age_interval changed to age_bin
dfC = df.drop(['age_interval'], axis = 1)
dfC['age_bin'] = np.zeros((len(df)))
for age in range(len(dfC['age'])):
    if dfC['age'][age] in interval_youth:
        dfC['age_bin'][age] = 1
    elif dfC['age'][age] in interval_1:
        dfC['age_bin'][age] = 2
    elif dfC['age'][age] in interval_2:
        dfC['age_bin'][age] = 3
    else:
        dfC['age_bin'][age] = 4
print("Done")

# fitting Cox Prop. model
cph.fit(dfC, duration_col='week', event_col='arrest')

survTab['Survival (Cox)'] = cph.baseline_survival_
survTab['Survival (Cox)'][0] = 1 
survTab['Cu. Hazard (Cox)'] = cph.baseline_cumulative_hazard_
survTab['Cu. Hazard (Cox)'][0] = 0
survTab['Cu. Hazard'][0] = 0
survTab = survTab.round({'Cu. Hazard':4, 'Survival':4,'Cu. Hazard (Cox)':4, 'Survival (Cox)':4})
print(survTab)

#cph.predict_hazard()
#np.array(cph.baseline_survival_) + np.array(cph.baseline_cumulative_hazard_)
#np.array(kmf.survival_function_) + np.array(-np.log(kmf.survival_function_))
#np.array(kmf.survival_function_) + np.array(naf.cumulative_hazard_)

#survTab.to_latex(r"C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\survTab.tex",index=False)




# Splitting into age_intervals
unique_ages = np.sort(df['age'].unique())
interval_youth = unique_ages[0:3]
interval_1 = unique_ages[3:8]
interval_2 = unique_ages[8:14]
interval_3 = unique_ages[14:]
'''
df['age_interval'] = np.zeros((len(df)))
for age in range(len(df['age'])):
    if df['age'][age] in interval_youth:
        df['age_interval'][age] = "17-19"
    elif df['age'][age] in interval_1:
        df['age_interval'][age] = "20-24"
    elif df['age'][age] in interval_2:
        df['age_interval'][age] = "25-30"
    else:
        df['age_interval'][age] = "31-44"
print("Done")

df = df.drop(['paro', 'prio'], axis = 1)
df.to_excel(r'C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\recidivism_dataset.xlsx', sheet_name = 'data')
'''


#df = pd.read_excel(r"C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\recidivism_dataset.xlsx", sheet_name = 'data')

# This section prints a plot of the survival function 
# of each of the 4 age intervals

kmf = KaplanMeierFitter()
colors = ["red", "gold", "cyan", "limegreen"]
intervals = np.sort(df['age_interval'].unique())
for i, age_interval in enumerate(intervals):
    idx = df['age_interval'] == age_interval
    kmf.fit(df['week'].loc[idx], df['arrest'].loc[idx], label = age_interval)
    if age_interval == intervals[0]:
        ax = kmf.survival_function_.plot(color = colors[i])
    else:
        kmf.survival_function_.plot(ax = ax, color = colors[i])
        
    plt.ylabel('Survival probability')
    plt.xlabel('Weeks')
    plt.legend(title = "Ages")
    plt.xlim(0, 52)
    plt.title('Kaplan-Meier')

#plt.figure(dpi=300)
  
plt.savefig(r'C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\allAgesKaplanMeier.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()
#plt.savefig('allAgesKaplanMeier.pdf', bbox_inches = 'tight', pad_inches = 0)


# LOG-RANK TEST: Age_interval i vs. all other
ageLogRankTest = np.zeros((4,4), dtype=str)
ageLogRankTestDf = pd.DataFrame(ageLogRankTest)
ageLogRankTestDf.columns = ['17-19', '20-24', '25-30', '31-44']
ageLogRankTestDf.index = ['17-19', '20-24', '25-30', '31-44']

from lifelines.statistics import logrank_test
for i, age_interval in enumerate(intervals):
    
    # Log-Rank Testing:
    idx = df['age_interval'] == age_interval
    durCol, eventCol = df['week'].loc[idx], df['arrest'].loc[idx]
    otherAges = [age_int for age_int in intervals if age_int != age_interval]
    for j in otherAges:
        otherIdx = df['age_interval'] == j
        otherDurCol, otherEventCol = df['week'].loc[otherIdx], df['arrest'].loc[otherIdx]
        rankT = logrank_test(durCol,otherDurCol,eventCol,otherEventCol)
        pval = float(round(rankT.p_value, 3))
        statval = float(round(rankT.test_statistic, 2))
        if pval == 0.0 and len(str(statval)) != 5:
            pval = "0.000"
        elif len(str(statval)) == 5:
            pval = "0.00"
        elif len(str(pval)) != 5:
            pval = str(pval) + "0"
        ageLogRankTestDf[age_interval][j] = str(statval) + " (" + str(pval) + ")"

#ageLogRankTestDf.to_latex(r"C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\ageLogRankTestDf.tex",index=True)



'''
This section prints 4 subplots of survival curves for each
of the 4 age intervals
'''
legends = dict(enumerate(intervals))  
for i, age_interval in enumerate(intervals):
    ax = plt.subplot(2, 2, i + 1)
    idx = df['age_interval'] == age_interval
    kmf.fit(df['week'].loc[idx], df['arrest'].loc[idx], label = age_interval)
    age_label = "Ages " + legends[i]
    kmf.plot(ax = ax, legend=True, figsize=(8,8), label = age_label, color = colors[i])
    plt.xlim(0, 52)
    plt.xlabel('Weeks')
    #plt.figure(dpi=100)
    if i==0 or i==2:
        plt.ylabel('Survival probability')
        
plt.savefig('allAgesSub.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

df['age_interval'].value_counts()

#Plotting groups:

#financial aid
fin = (df["fin"] == 1)
no_fin = (df["fin"] == 0)

#marital status
mar = (df['mar'] == 1)
no_mar = (df['mar'] == 0)

#full-time work experience before incarceration
wor = (df['wexp'] == 1)
no_wor = (df['wexp'] == 0)

#race: 1=black, 0=white/other
race_b = (df['race'] == 1)
race_o = (df['race'] == 0)

#Groups list contains the indexes for splitting the dataset into respective groups
groups = [(fin,no_fin), (mar,no_mar), (wor,no_wor), (race_b, race_o)]

# Group 1 and Group 2:
g1 = wor
g2 = no_wor


dcolors = ["darkred", "goldenrod", "darkturquoise", "green"]
intervals = np.sort(df['age_interval'].unique())
for i, age_interval in enumerate(intervals):
    idx = df['age_interval'] == age_interval
    kmf.fit(df['week'].loc[idx].loc[g1], df['arrest'].loc[idx].loc[g1], label = age_interval)
    if age_interval == intervals[0]:
        ax = kmf.survival_function_.plot(color = colors[i])
    else:
        kmf.survival_function_.plot(ax = ax, color = colors[i])
    
    # No financial aid
    kmf.fit(df['week'].loc[idx].loc[g2], df['arrest'].loc[idx].loc[g2], label = age_interval)
    kmf.survival_function_.plot(ax = ax, color = dcolors[i])
    
    #kmf.plot(ax=ax, ci_show=True)
    plt.ylabel('Survival probability')
    plt.xlabel('Weeks')
    plt.legend(title = "Ages")
    plt.xlim(0, 52)

# Count how many in Group 1, for age interval:
df['arrest'].loc[idx].loc[g1].value_counts()
# Count how many in Group 2, for age interval:
df['arrest'].loc[idx].loc[g2].value_counts()

# Four subplots of specified grouping (ie. fin vs. no_fin) in all age_intervals
intervals = np.sort(df['age_interval'].unique())
for i, age_interval in enumerate(intervals):
    #ax = plt.subplot(2, 2, i + 1, figsize=(15,15))
    ax = plt.subplot(2, 2, i + 1)
    idx = df['age_interval'] == age_interval
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    T1 = df['week'].loc[idx].loc[g1]
    E1 = df['arrest'].loc[idx].loc[g1]
    T2 = df['week'].loc[idx].loc[g2]
    E2 = df['arrest'].loc[idx].loc[g2]
    kmf1.fit(T1, E1, label = age_interval)
    kmf2.fit(T2, E2, label = age_interval)
    
    rankT = logrank_test(T1,T2,E1,E2)
    yes_label = "1" + " ("+legends[i]+")"
    no_label = "0" + " ("+legends[i]+")"
    kmf1.plot(ax = ax, legend=True, figsize=(8,8), label = yes_label, color = colors[i], ci_show=False)
    kmf2.plot(ax = ax, legend=True, figsize=(8,8), label = no_label, color = dcolors[i], ci_show=False)
    plt.xlim(0, 52)
    if i == 2 or i == 3:
        plt.xlabel('Weeks')
    else:
        plt.xlabel('')
    #plt.figure(dpi=100)
    if i==0 or i==2:
        plt.ylabel('Survival probability')
    
    pval = float(round(rankT.p_value, 3))
    statval = float(round(rankT.test_statistic, 2))
    if pval == 0.0 and len(str(statval)) != 5:
        pval = "0.000"
    elif len(str(statval)) == 5:
        pval = "0.00"
    elif len(str(pval)) != 5:
        pval = str(pval) + "0"
    LOG_RANK_TITLE = str(statval) + " (" + str(pval) + ")"
    plt.title(LOG_RANK_TITLE)
if g1.name == 'wexp':
    plt.suptitle('Work experience (1) vs. No work experience (0)', size=14)
    plt.subplots_adjust(top=0.91)      
plt.savefig(r'C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\allAgesKM_wexp.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

'''
#### SAME THING, BUT FOR COX PROP HAZARDS MODEL:
    # COX MODEL WON'T FIT INDIVIDUAL AGE_INTERVALS (BINS) DUE TO SINGULARITY...
# Four subplots of specified grouping (ie. fin vs. no_fin) in all age_intervals
cph = CoxPHFitter()
cph.fit(dfC, duration_col='week', event_col='arrest')#, strata=['wexp'])
cph.plot()
coxGroups = ['fin', 'mar', 'wexp', 'race']
#cph.print_summary()
from matplotlib import colors as colorsMapping
for i, age_interval in enumerate(intervals):
    i = i+1
    age_idx = (dfC['age_bin'] == i) 
    cph = CoxPHFitter(penalizer=1)
    fit_dfC = dfC.loc[age_idx]
    fit_dfC.pop('age_bin')
    cph.fit(fit_dfC, duration_col='week', event_col='arrest')
    colorsMap =  colorsMapping.ListedColormap([colors[i], dcolors[i]])
    ax = plt.subplot(2, 2, i)
    #cph.plot_covariate_groups(grp, [0,1], ax = ax, legend=True, figsize=(8,8), cmap = colorsMap)
    cph.plot_covariate_groups('wexp', [1,0], ax = ax, legend=True, figsize=(8,8), cmap = colorsMap)#, plot_baseline = False)
    ax.legend(['wexp', 'no ' + 'wexp', 'baseline'])
    if i==1 or i==3:
        plt.ylabel('Survival probability') 
    if i==3 or i==4:
        plt.xlabel('Weeks')
    else:
        plt.xlabel('')
    # Log Rank titles from previous plot...
    #LOG_RANK_TITLE = groups_logRank_dict[grp] 
    #plt.title(LOG_RANK_TITLE)
    plt.xlim(0, 52)

# PERHAPS: np.mean(cph.predict_survival_function(dfC.loc[age_idx]),axis=1).plot()
#################################################
'''
# Four subplots, grouped by all four grouping variables (fin, mar, work, race) of all ages in each plot
groups_index = ['Financial aid', 'Marital status', 'Work experience', 'Race (black/other)']
rank_df = pd.DataFrame(index=groups_index,columns=['t-value','p-value'])
groups_logRank_dict = {"fin" : 0 , "mar": 0, "wexp": 0, "race": 0} # Save log rank titles for Cox plots later...
for i, grp in enumerate(groups):
    g1 = grp[0]
    g1_name = grp[0].name
    g2 = grp[1]
    g2_name = "no " + grp[0].name
    # When race, fix group labels and coloring:
    color1 = colors[i]
    color2 = dcolors[i]
    if i == 3:
        g1_name = "black"
        g2_name = "other"
        color1 = dcolors[i]
        color2 = colors[i]
    
    ax = plt.subplot(2, 2, i + 1)
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    T1 = df['week'].loc[g1]
    E1 = df['arrest'].loc[g1]
    T2 = df['week'].loc[g2]
    E2 = df['arrest'].loc[g2]
    kmf1.fit(T1, E1, label = g1_name)
    kmf2.fit(T2, E2, label = g2_name)
    #Log Rank Test
    from lifelines.statistics import logrank_test
    rankT = logrank_test(T1,T2,E1,E2)
    
    pval = float(round(rankT.p_value, 4))
    statval = float(round(rankT.test_statistic, 4))
    rank_df['p-value'][i] = pval
    rank_df['t-value'][i] = statval
    kmf1.plot(ax = ax, legend=True, figsize=(8,8), label = g1_name, color = color1, ci_show=False)
    kmf2.plot(ax = ax, legend=True, figsize=(8,8), label = g2_name, color = color2, ci_show=False)
    plt.xlim(0, 52)
    if i == 0 or i == 1:
        plt.xlabel('')
    else:
        plt.xlabel('Weeks')
    
    pval = float(round(rankT.p_value, 3))
    statval = float(round(rankT.test_statistic, 2))
    if pval == 0.0 and len(str(statval)) != 5:
        pval = "0.000"
    elif len(str(statval)) == 5:
        pval = "0.00"
    elif len(str(pval)) != 5:
        pval = str(pval) + "0"
    LOG_RANK_TITLE = str(statval) + " (" + str(pval) + ")"
    groups_logRank_dict[grp[0].name] = LOG_RANK_TITLE
    plt.title(LOG_RANK_TITLE)
    #plt.title('p-value: ' + str(pval))
    #plt.figure(dpi=100)
    if i==0 or i==2:
        plt.ylabel('Survival probability') 
    
print(rank_df)
#rank_df.to_latex(r'C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\logrankResultsKaplanMeierGroups.tex')

plt.suptitle('Kaplan-Meier', size=14)
plt.subplots_adjust(top=0.91)
plt.savefig(r'C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\allGroupsKM.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

# Compute y-axis Recidivism Rates as Hazard functions instead of survival functions

# Cox Proportional Hazard

cph = CoxPHFitter()
cph.fit(dfC, duration_col='week', event_col='arrest')#, strata=['wexp'])
cph.plot()
coxGroups = ['fin', 'mar', 'wexp', 'race']
#cph.print_summary()
from matplotlib import colors as colorsMapping
for i, grp in enumerate(coxGroups):
    colorsMap =  colorsMapping.ListedColormap([colors[i], dcolors[i]])
    ax = plt.subplot(2, 2, i + 1)
    #cph.plot_covariate_groups(grp, [0,1], ax = ax, legend=True, figsize=(8,8), cmap = colorsMap)
    if i in [0,1,2]:
        cph.plot_covariate_groups(grp, [1,0], ax = ax, legend=True, figsize=(8,8), cmap = colorsMap)#, plot_baseline = False)
        ax.legend([grp, 'no ' + grp, 'baseline'])
    else:
        cph.plot_covariate_groups(grp, [0,1], ax = ax, legend=True, figsize=(8,8), cmap = colorsMap)#, plot_baseline = False)
        #ax.legend(['black', 'white/other', 'baseline'])
        handles, labels = ax.get_legend_handles_labels() # Retrieve colorbars in legend (called "handles") to change their order..
        labels = ['black', 'other', 'baseline']
        handles = handles[0:2][::-1] + [handles[-1]]
        ax.legend(handles, labels)
    if i==0 or i==2:
        plt.ylabel('Survival probability') 
    if i==2 or i==3:
        plt.xlabel('Weeks')
    else:
        plt.xlabel('')
    # Log Rank titles from previous plot...
    LOG_RANK_TITLE = groups_logRank_dict[grp] 
    plt.title(LOG_RANK_TITLE)
    plt.xlim(0, 52)
plt.suptitle('Cox Proportional Hazards', size=14)
plt.subplots_adjust(top=0.91)
#plt.savefig('allGroupsCox.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.savefig(r'C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\allGroupsCox.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

# age_interval split into age_bins of [1,2,3,4] representing the age_intervals of 17-19, 20-24, .. ,etc.
# as to comply with the Cox Proportional Model
cph.plot_covariate_groups('age_bin',[1,2,3,4], color = colors)
#ax.ylabel('Survival probability')
plt.xlabel('Weeks')
intervals
plt.legend([str(intervals[0]), str(intervals[1]), str(intervals[2]), str(intervals[3]), 'baseline'], title = "Ages")
plt.title('Cox Proportional Hazards')
plt.xlim(0, 52)
plt.savefig(r'C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\allAgesCox.pdf', bbox_inches = 'tight', pad_inches = 0)
#plt.savefig('allAgesCox.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()

cph.check_assumptions(dfC)
#'wexp' fails the non-proportional test => Dangerous to make inference from wexp
cph.print_summary()
#cph.baseline_cumulative_hazard_.plot()


# LATEST: MAR vs. NO_MAR Kaplan-Meier:
g1_mar = mar
g2_mar = no_mar
# Four subplots of specified grouping (ie. fin vs. no_fin) in all age_intervals
intervals = np.sort(df['age_interval'].unique())
for i, age_interval in enumerate(intervals):
    #ax = plt.subplot(2, 2, i + 1, figsize=(15,15))
    ax = plt.subplot(2, 2, i + 1)
    idx = df['age_interval'] == age_interval
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    T1 = df['week'].loc[idx].loc[g1_mar]
    E1 = df['arrest'].loc[idx].loc[g1_mar]
    T2 = df['week'].loc[idx].loc[g2_mar]
    E2 = df['arrest'].loc[idx].loc[g2_mar]
    if age_interval != '17-19':
        kmf1.fit(T1, E1, label = age_interval)
    kmf2.fit(T2, E2, label = age_interval)
    
    rankT = logrank_test(T1,T2,E1,E2)
    yes_label = "1" + " ("+legends[i]+")"
    no_label = "0" + " ("+legends[i]+")"
    if age_interval != '17-19':
        kmf1.plot(ax = ax, legend=True, figsize=(8,8), label = yes_label, color = colors[i], ci_show=False)
    kmf2.plot(ax = ax, legend=True, figsize=(8,8), label = no_label, color = dcolors[i], ci_show=False)
    plt.xlim(0, 52)
    if i == 2 or i == 3:
        plt.xlabel('Weeks')
    else:
        plt.xlabel('')
    #plt.figure(dpi=100)
    if i==0 or i==2:
        plt.ylabel('Survival probability')
    
    pval = float(round(rankT.p_value, 3))
    statval = float(round(rankT.test_statistic, 2))
    if pval == 0.0 and len(str(statval)) != 5:
        pval = "0.000"
    elif len(str(statval)) == 5:
        pval = "0.00"
    elif len(str(pval)) != 5:
        pval = str(pval) + "0"
    LOG_RANK_TITLE = str(statval) + " (" + str(pval) + ")"
    if age_interval != '17-19':
        plt.title(LOG_RANK_TITLE)
plt.suptitle('Married (1) vs. Not married (0)', size=14)
plt.subplots_adjust(top=0.91)    
plt.savefig(r'C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\allAgesKM_mar.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()


# COUNTING CHARACTERISTICS SUBGROUPS:
# Count how many in Group 1, for age interval:
    # wor vs. no_wor for instance...
g1_c = race_b
g2_c = race_o
df['arrest'].loc[g1_c].value_counts().sum()
# Count how many in Group 2, for age interval:
df['arrest'].loc[g2_c].value_counts().sum()

g1_wor = wor
g2_wor = no_wor

# Making table to count work experience and marital status in each age group:
#countTable = np.zeros((4,4), dtype=int)
groups_index = ['17-19', '20-24', '25-30', '31-44']
countTable = pd.DataFrame(index=groups_index,columns=['Work Experience (1)', 'No Work Experience (0)', 'Married (1)', 'Not Married (0)'])
for i, age_interval in enumerate(intervals):
    idx = df['age_interval'] == age_interval
    countTable['Work Experience (1)'][i] = df['week'].loc[idx].loc[g1_wor].value_counts().sum()
    countTable['No Work Experience (0)'][i] = df['week'].loc[idx].loc[g2_wor].value_counts().sum()
    countTable['Married (1)'][i] = df['week'].loc[idx].loc[g1_mar].value_counts().sum()
    countTable['Not Married (0)'][i] = df['week'].loc[idx].loc[g2_mar].value_counts().sum()
print(countTable)
countTable.to_latex(r'C:\Users\jacob\Dropbox\1KU ECONOMICS\8. Semester\Demography\Code\countTableOfWorkExpAndMaritalStatus.tex')
