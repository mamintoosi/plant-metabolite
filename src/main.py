# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    BSD license.
"""Algorithms used in bio-graphs."""
## کاوش الگوهای پرتکرار

from pandas import ExcelWriter
import pandas as pd
import numpy as np
from orangecontrib.associate.fpgrowth import *
from itertools import tee
from tqdm.notebook import tqdm
import networkx as nx
from pandas import DataFrame 
pd.options.display.float_format = "{:.2f}".format
import random
import argparse
from bio_graph_utils import *
from fim_utils import fim_bio
import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type = str, default = 'stomach')
    parser.add_argument('--min_count', type = int, default = 1)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # برای اجرای محلی 
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # در اجرای محلی پارامترهای رشته‌ای ارسالی به مین نباید داخل تک کوتیشن باشند

    # global args 
    args = get_args()

    data_dir = 'data/'
    output_dir = 'results/'
    working_dir = 'data/'


    # فایل سرطان معده حاوی گیاهانی است که خاصیت ضدسرطانی آنها اثبات شده است.
    # فایل ای‌سی لیست متابولیت‌هایی است که خاصیت ضد سرطانی آنها مشخص شده
    # به همراه گیاهانی که این متابولیت‌ها در آنها هستند و این لیست برای همه سرطانهاست
    # و در سایت اصلی تفکیک نشده اند
    working_file_name = 'AC.xlsx'

    # فایلهای زیر حاوی گیاهانی هست که خاصیت ضد سرطانی آنها اثبات است
    # GT_file_name = 'LR_Met_Plant.xlsx'
    if args.dataset_name.lower() == 'stomach':
        GT_file_name = 'Stomach.xlsx'
    elif 'wound' in args.dataset_name.lower():
        GT_file_name = 'Wound_healing.xlsx'
    elif 'breast' in args.dataset_name.lower():
        GT_file_name = 'Breast.xlsx'
    else:
        print('Unknown dataset')
        exit(1)
    node_objects = 'Plant'
    edge_objects = 'Met'

    working_file = working_dir+working_file_name
    df = pd.read_excel(working_file, engine="openpyxl") 
    # for col in df.columns:
    #    df[col] = df[col].apply(lambda x: float(x) if not pd.isna(x) else x)

    GT_file = working_dir+GT_file_name
    true_df = pd.read_excel(GT_file, engine="openpyxl") 
    true_list = list(true_df[node_objects].unique())
    # true_list = list(true_df.keys().values)

    # در فرم نود=گیاه، گیاهانی که حداقل مین‌کانت بار در دیتابیس اومدن، نگه داشته میشن
    # به عبارت دیگه حداقل مین‌کانت متابولیت داشتن
    # نگهداری ستونهای با بیش از یا مساوی با ۵ عنصر
    min_count = args.min_count
    w_flag = True #False
    # حداقل تعداد تکرار برای مجموعه اقلام مکرر
    # حد زیر برای نگهداری متابولیت‌هایی است که حداقل در پنج گیاه آمده‌اند.
    # minFreq = 2

    # print('Table info: ', df.shape)
    # print('Number of nodes in main file: ', len(df[node_objects].unique()))

    if min_count>1:
        s = df[node_objects].value_counts()
        df = df[df[node_objects].isin(s[s >= min_count].index)]
        # print('Number of nodes after prunning those nodes which have < min count={} elements is: {}'.format(min_count,len(df[node_objects].unique())))

    G,dfct,bow = make_graph_from_df(df,node_objects,edge_objects)

    # اضافه کردن وزن لبه‌ی متابولیت‌های مشترک
    # در حال حاضر فقط یک واحد اضافه میشه که میشه به نسبت تکرارش در مجموعه دوم اضافه بشه
    # print('Number of Metabolites in dfct: ', len(dfct.columns))
    second_graph_edges = list(true_df[edge_objects].unique())
    # print('Number of Metabolites in Wound healing: ',len(second_graph_edges))
    intersected_edges = set(dfct.columns).intersection(set(second_graph_edges))
    # print('Number of intersected Meabolites:', len(intersected_edges))
    edges_value_counts = true_df[edge_objects].value_counts()
    for row in range(dfct.shape[0]):
        for e in intersected_edges:
            if dfct.iloc[row][e] != 0:
                dfct.iloc[row][e] += edges_value_counts[e]
                        
    # پیدا کردن بزرگترین زیرگراف همبند
    # print('Computing the largest connected graph...\n')
    subG = largest_con_com(dfct, G)
    # print('Number of sub graph nodes:', len(subG.nodes()))
    # # nx.draw_shell(subG,with_labels=True)
    subG_ix = list(subG.nodes())
    dfct_subG = dfct.loc[subG_ix]
    # print('Sub graph info before dropping: ', dfct_subG.shape)
    # drop columns with zero sum
    dfct_subG = dfct_subG.loc[:, (dfct_subG != 0).any(axis=0)]
    # dfct = pd.crosstab(df[node_objects], df[edge_objects])
    # print('Sub graph info: ', dfct_subG.shape)

    # print('Computing the graph features...\n')
    gf_df, gf_df_sorted = rank_using_graph_features(subG, weight='weight')
    # , min_count, node_objects, \
    #     edge_objects, data_dir, output_dir, working_file_name)
    # # df[list(subG)]
    # # مجموعه اقلام مکرر
    # df_subG = df[list(subG)]
    # # T, bow, featureNames = bow_nodes(df_subG)


    index = np.arange(1,50,2)
    [apk_gf,ark_gf] = compute_metrics(true_list, list(gf_df_sorted.index.values))

    # node_names = dfct_subG.index.format()
    # random_order = node_names.copy()
    # random.shuffle(random_order)
    # [apk_rand,ark_rand] = compute_metrics(true_list, random_order)

    # method_names = ['Graph Features']#, 'Random']#, 'RT']
    # scores = [apk_gf] #, apk_rand]
    jadval = pd.DataFrame(
        {'k': index,
        'min_freq': '',
        'Method Name': 'Graph Features',
        'AP@k': apk_gf,
        'AR@k': ark_gf
        })
    # print('Computing frequent itemsets...\n')

    # # sorted_nodes_idx, sorted_nodes_idx_w, G, degreeG = fim_bio(minFreq,T,bow,featureNames)
    # The new weights will be stored in subG
    min_freq_list = [2,4,6,8,10,12,14,16,18,20] #list(np.arange(2,20,2)) #
    min_max_scaler = preprocessing.MinMaxScaler()

    minFreq = min(min_freq_list)
    bow = dfct_subG.values
    T = [[i for i,x in enumerate(row) if x != 0] for row in bow]
    ItemSets = frequent_itemsets(T, minFreq)
    # print(type(itemsets),itemsets)

    with tqdm(total=len(min_freq_list)) as progress_bar:
        for min_freq in min_freq_list:
            # مجبوریم کپی درست کنیم وگرنه با یک دور مرور ایتریتور خالی میشود
            ItemSets,itemsets = tee(ItemSets)
            sorted_nodes_idx, sorted_nodes_idx_w, degreeG_fim, degreeG_fim_w = fim_bio(subG, \
                dfct_subG, min_freq, itemsets, T, bow)
                # , node_objects, edge_objects, output_dir, working_file_name)

            if not w_flag:
                node_names = [dfct_subG.index.format()[x] for x in sorted_nodes_idx]
                degreeG = degreeG_fim.copy()
                pasvand = ''
            else:
                node_names = [dfct_subG.index.format()[x] for x in sorted_nodes_idx_w]
                degreeG = degreeG_fim_w.copy()
                pasvand = '_w'
            method_name = 'Frequent Itemset'+pasvand
            # #استفاده از گراف مجموعه اقلام مکرر
            gf_fim_df = gf_df.loc[:, gf_df.columns != 'features_sum']
            # numpy_matrix = df.values
            X = min_max_scaler.fit_transform(np.array(degreeG).reshape(-1, 1))
            gf_fim_df['degree_fim'] = X
            features_sum = gf_fim_df.sum(axis=1)
            gf_fim_df['features_sum'] = features_sum
            # gf_fim_df
            gf_fim_df_sorted = gf_fim_df.sort_values(by='features_sum', ascending=False)

            [apk_fim,ark_fim] = compute_metrics(true_list, node_names)
            # method_names.append('fim_mc'+str(min_count))
            # scores.append(apk_fim)
            tmp_df = pd.DataFrame(
                {'k': index,
                'min_freq': str(min_freq),
                'Method Name': method_name,
                'AP@k': apk_fim,
                'AR@k': ark_fim
                })
            jadval = jadval.append(tmp_df, ignore_index=True)

            [apk_gf_fim,ark_gf_fim] = compute_metrics(true_list, list(gf_fim_df_sorted.index.values))
            # method_names.append('H_mc'+str(min_count))
            # scores.append(apk_gf_fim)
            tmp_df = pd.DataFrame(
                {'k': index,
                'min_freq': str(min_freq),
                'Method Name': 'Hybrid',
                'AP@k': apk_gf_fim,
                'AR@k': ark_gf_fim
                })
            jadval = jadval.append(tmp_df, ignore_index=True)

            progress_bar.update(1) # update progress

    plt.figure()
    sns.lineplot(x="k", y="AP@k",
                hue="Method Name",# style="min_freq",
                data=jadval)
    png_file_name = 'results/{}_{}_{}_{}_mc{:d}_mf{:d}-{:d}{}_apk.png'.format(working_file_name[:2], GT_file_name[:2], \
        node_objects, edge_objects, min_count, min_freq_list[0], min_freq_list[-1],pasvand)
    plt.savefig(png_file_name)

    plt.figure()
    sns.lineplot(x="k", y="AR@k",
                hue="Method Name",# style="min_freq",
                data=jadval)
    png_file_name = 'results/{}_{}_{}_{}_mc{:d}_mf{:d}-{:d}{}_ark.png'.format(working_file_name[:2], GT_file_name[:2], \
        node_objects, edge_objects, min_count, min_freq_list[0], min_freq_list[-1],pasvand)
    plt.savefig(png_file_name)

    # Write to Excel File
    xls_file_name = 'results/{}_{}_{}_{}_mc{:d}_mf{:d}-{:d}{}.xlsx'.format(working_file_name[:2], GT_file_name[:2], \
        node_objects, edge_objects, min_count, min_freq_list[0], min_freq_list[-1],pasvand)
    writer = ExcelWriter(xls_file_name)
    gf_df_sorted.to_excel(writer, sheet_name='gf_df_sorted')  # , index=False)
    gf_fim_df_sorted.to_excel(writer, sheet_name='gf_fim_df_sorted')  # , index=False)
    jadval.to_excel(writer, sheet_name='jadval')  # , index=False)
    gf_df.to_excel(writer, sheet_name='gf_df')  # , index=False)
    writer.save()

    # fig = plt.figure(figsize=(10, 4))
    # recmetrics.mapk_plot(scores, model_names=names, k_range=index)
    # png_file_name = 'results/{}_{}_{}_{}_{}_{}_apk.png'.format(working_file_name[:2], GT_file_name[:2], \
    #     node_objects, edge_objects, str(min_count), str(minFreq))
    # fig.savefig(png_file_name)

    # scores = [ark_gf, ark_fim, ark_fim_w, ark_gf_fim, ark_gf_fim_w]#, ark_rand]
    # fig = plt.figure(figsize=(10, 4))
    # recmetrics.mark_plot(scores, model_names=names, k_range=index)
    # png_file_name = 'results/{}_{}_{}_{}_{}_{}_ark.png'.format(working_file_name[:2], GT_file_name[:2], \
    #     node_objects, edge_objects, str(min_count), str(minFreq))
    # fig.savefig(png_file_name)


    # file_name = 'results/{}_{}_{}_{}_{}_{}.xlsx'.format(working_file_name[:2], GT_file_name[:2], \
    #     node_objects, edge_objects, str(min_count), str(minFreq))
    # writer = ExcelWriter(file_name)
    # gf_df_sorted.to_excel(writer, sheet_name='gf_df_sorted')  # , index=False)
    # gf_fim_df_sorted.to_excel(writer, sheet_name='gf_fim_df_sorted')  # , index=False)
    # gf_fim_w_df_sorted.to_excel(writer, sheet_name='gf_fim_w_df_sorted')  # , index=False)
    # gf_df.to_excel(writer, sheet_name='gf_df')  # , index=False)
    # writer.save()

            # s1 = gf_fim_df_sorted['degree_cent']
            # s2 = gf_fim_df_sorted['degree_fim']
            # print('Correlation of degree and degree_fim:',s1.corr(s2))
