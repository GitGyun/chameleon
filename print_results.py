import numpy as np
import torch
import pandas as pd
import os
import glob
import argparse
from functools import cmp_to_key
from downstream.davis2017.dataset import DAVIS2017
from downstream.ap10k.dataset import AP10K
from downstream.linemod.dataset import LINEMOD
from args import DOWNSTREAM_DATASETS
import itertools


dataset_dict = {
    'ap10k': [('animalkp', 'Animal Keypoint', 'AP (↑)'),
                      ('animalkp', 'Animal Keypoint', 'AR (↑)'),],
    'davis2017': [('vos', 'Video Object Segmentation', 'J&F Mean (↑)'),
                  ('vos', 'Video Object Segmentation', 'J Score (↑)'),
                  ('vos', 'Video Object Segmentation', 'F Score (↑)')],
    'eigen': [('depth_kitti', 'Monocular Depth Estimation', 'Abs. Rel. (↓)')],
    'nyud2': [('depth_nyu', 'Monocular Depth Estimation', 'RMSE (↓)'),
              ('depth_nyu', 'Monocular Depth Estimation', 'Abs. Rel. (↓)')],
    'linemod': [('pose_6d', '6D Pose Estimation', 'ADD 0.1 (↓)')],
    'isic2018': [('segment_medical', 'Medical Segmentation', 'F1 (↑)'),],
                #  ('segment_medical', 'Medical Segmentation', 'Iou (↑)'),],
}

shots_dict = {
    'ap10k': [10, 20, 50, 100],
    'davis2017': [1, 2, 3, 4],
    'eigen': [20],
    'nyud2': [20, 1000, 2000],
    'linemod': [20, 50, 80, 200],
    'isic2018': [5, 20, 80],
}
shots_dict = {k: [str(i) for i in v] for k, v in shots_dict.items()}

imsize_dict = {
    'ap10k': [224],
    'davis2017': [224, 384, 392],
    'eigen': [224],
    'nyud2': [224],
    'linemod': [224],
    'isic2018': [224, 384, 392],
}
imsize_dict = {k: [str(i) for i in v] for k, v in imsize_dict.items()}

class_dict = {
    'ap10k': AP10K.SPECIES,
    'davis2017': DAVIS2017.CLASS_NAMES,
    'linemod': LINEMOD.CLASS_NAMES,
}

single_class_dict = {
    'davis2017': [class_name for class_name, n_instance in zip(DAVIS2017.CLASS_NAMES, DAVIS2017.NUM_INSTANCES) if n_instance == 1],
    'linemod': ['ape', 'benchviseblue', 'cam', 'cat']
}

multi_class_dict = {
    'ap10k': AP10K.SPECIES,
    'davis2017': [class_name for class_name, n_instance in zip(DAVIS2017.CLASS_NAMES, DAVIS2017.NUM_INSTANCES) if n_instance > 1],
}


def safe_listdir(path):
    try:
        return os.listdir(path)
    except:
        return []


def add_results(table, row, col, value, tag, mode='max', sid=0):
    if row not in table.index:
        table = pd.concat([table, pd.DataFrame(index=[row], columns=table.columns)])

    if not isinstance(table.loc[row][col], dict):
        table.loc[row][col] = {'mode': mode}

    if tag not in table.loc[row][col]:
        table.loc[row][col][tag] = {}

    table.loc[row][col][tag][sid] = value

    return table


def process_dups(table, compact=False, subcompact=False, average=False, all_support_id=False):
    for row in table.index:
        for col in table.columns:
            if isinstance(table.loc[row][col], dict):
                res_dict = table.loc[row][col]
                mode = res_dict.pop('mode')
                if all_support_id:
                    aggregate = []
                    for sid in range(5):
                        candidates = []
                        for tag in res_dict:
                            if sid in res_dict[tag]:
                                candidates.append(res_dict[tag][sid])
                        if len(candidates) > 0:
                            if mode == 'max':
                                aggregate.append(max(candidates))
                            else:
                                aggregate.append(min(candidates))
                    
                    if compact:
                        table.loc[row][col] = f'{np.mean(aggregate):.04f}'
                    else:
                        table.loc[row][col] = (f'{np.mean(aggregate):.04f} \u00B1 {np.std(aggregate):.04f}',
                                               f'{len(aggregate)} seeds', f'over {len(res_dict)} hps', [f'{x:.04f}' for x in aggregate])
                else:
                    res_dict = {tag: list(res_dict[tag].values()) for tag in res_dict}
                    aggregate = {}
                    for tag in res_dict:
                        aggregate[tag] = (np.mean(res_dict[tag]), np.std(res_dict[tag]), len(res_dict[tag]))
                    if mode == 'max':
                        best_tag = max(aggregate, key=lambda x: aggregate[x][0])
                    else:
                        best_tag = min(aggregate, key=lambda x: aggregate[x][0])
                    all_results = res_dict[best_tag]
                    all_results = [f'{x:.04f}' for x in all_results]
                    if compact:
                        table.loc[row][col] = f'{aggregate[best_tag][0]:.04f}'
                    elif subcompact:
                        table.loc[row][col] = f'{aggregate[best_tag][0]:.04f}', best_tag
                    else:
                        table.loc[row][col] = (f'{aggregate[best_tag][0]:.04f} \u00B1 {aggregate[best_tag][1]:.04f}',
                                               f'{aggregate[best_tag][2]} seeds', f'{best_tag} over {len(res_dict)} hps', all_results)
    
    if average:
        groups = sorted(list(set([c[:5] for c in table.columns])))
        columns = pd.MultiIndex.from_tuples(groups, names=['Dataset', 'Shot', 'Image Size', 'Task', 'Metric'])
        avg_table = pd.DataFrame(index=table.index, columns=columns)
        for column, group in zip(columns, groups):
            for i, row in enumerate(table.index):
                target_columns = [c[:5] == group[:5] for c in table.columns]
                if isinstance(table.loc[row][target_columns].values[0], float) and np.isnan(table.loc[row][target_columns].values[0]):
                    continue
                if compact or subcompact:
                    avg_table.loc[row][column] = f'{np.array([float(v) for v in table.loc[row][target_columns].values]).mean():.04f}'
                else:
                    avg_mean = np.array([float(v[0].split(' \u00B1 ')[0]) for v in table.loc[row][target_columns].values]).mean()
                    avg_std = np.array([float(v[0].split(' \u00B1 ')[1]) for v in table.loc[row][target_columns].values]).mean()
                    avg_table.loc[row][column] = f'{avg_mean:.04f} \u00B1 {avg_std:.04f}'

        return avg_table
    else:
        return table


def merge_subname(subname, merge_args, full=False):
    if full:
        return subname
    else:
        return '_'.join([prefix for prefix in subname.split('_') if prefix.split(':')[0] not in merge_args])
    

def extract_arg_from_subname(subname, arg):
    return int(subname.split(f'_{arg}:')[1].split('_')[0])
    

def create_database(args, task_dict):
    result_root = os.path.join(args.root_dir, args.result_dir)
    
    # create indices with model names
    if args.exp_name is not None:
        exp_names = args.exp_name
    else:
        exp_names = sorted(safe_listdir(result_root))
        if args.exp_pattern is not None:
            exp_names = [exp_name for exp_name in exp_names if (args.exp_pattern in exp_name)]
        if args.exp_exclude is not None:
            exp_names = [exp_name for exp_name in exp_names if (exp_name not in args.exp_exclude)]
        if args.exp_exclude_pattern is not None:
            exp_names = [exp_name for exp_name in exp_names if sum([eep in exp_name for eep in args.exp_exclude_pattern]) == 0]
    
    # create multi-columns with dataset, task, and metric
    keys = []
    for dataset in datasets:
        shots = shots_dict[dataset] if args.shot is None else args.shot
        imsizes = imsize_dict[dataset] if args.img_size is None else args.img_size

        for shot, imsize in itertools.product(shots, imsizes):
            if args.class_wise:
                if args.class_name is not None:
                    class_names = args.class_name
                elif args.n_classes < 0:
                    if args.single_channel_only:
                        class_names = single_class_dict[dataset]
                    elif args.multi_channel_only:
                        class_names = multi_class_dict[dataset]
                    else:
                        class_names = class_dict[dataset]
                else:
                    if args.single_channel_only:
                        class_names = single_class_dict[dataset][args.n_classes*args.chunk_id:args.n_classes*(args.chunk_id+1)]
                    elif args.multi_channel_only:
                        class_names = multi_class_dict[dataset][args.n_classes*args.chunk_id:args.n_classes*(args.chunk_id+1)]
                    else:
                        class_names = class_dict[dataset][args.n_classes*args.chunk_id:args.n_classes*(args.chunk_id+1)]

                keys += [(dataset.upper(), task_name, shot, imsize, metric, dataset, class_name, task)
                        for task, task_name, metric in (task_dict[dataset][:1] if args.rep_score_only else task_dict[dataset])
                        for class_name in class_names]
            else:
                keys += [(dataset.upper(), task_name, shot, imsize, metric, dataset, task)
                        for task, task_name, metric in (task_dict[dataset][:1] if args.rep_score_only else task_dict[dataset])]
    if args.class_wise:
        columns = [(*key[:5], key[6]) for key in keys]
        columns = pd.MultiIndex.from_tuples(columns, names=['Dataset', 'Task', 'Shot', 'Image Size', 'Metric', 'Class'])
    else:
        columns = [key[:5] for key in keys]
        columns = pd.MultiIndex.from_tuples(columns, names=['Dataset', 'Task', 'Shot', 'Image Size', 'Metric'])

    # construct a database
    database = pd.DataFrame(index=exp_names, columns=columns)
    for exp_name in exp_names:
        for meta in keys:
            if args.class_wise:
                *column, dataset, class_name, task = meta
                column = (*column, class_name)
            else:
                *column, dataset, task = meta

            column = tuple(column)
            exp_dir = os.path.join(result_root, exp_name)
            if not os.path.exists(exp_dir):
                continue

            # list subnames
            exp_subnames = safe_listdir(exp_dir)

            # exclude subnames
            if args.subname_exclude_pattern is not None:
                exp_subnames = [exp_subname for exp_subname in exp_subnames
                                if sum([se in exp_subname for se in args.subname_exclude_pattern]) == 0]

            # include subnames
            if args.subname_prefix is not None:
                exp_subnames_filtered = []
                for exp_subname in exp_subnames:
                    target_subname = merge_subname(exp_subname, args.merge_args, args.full)
                    add = True
                    for i, sprf in enumerate(args.subname_prefix.split('*')):
                        if i == 0 and not target_subname.startswith(sprf):
                            add = False
                        elif i == len(args.subname_prefix.split('*')) - 1 and not target_subname.endswith(sprf):
                            add = False
                        elif sprf not in target_subname:
                            add = False
                    if add:
                        exp_subnames_filtered.append(exp_subname)
                exp_subnames = exp_subnames_filtered

            # merge subnames to rows
            rowname_list = list(set([merge_subname(exp_subname, args.merge_args, args.full) for exp_subname in exp_subnames]))
            shot, img_size = column[2:4]
            for rowname in sorted(rowname_list):
                # list subnames
                exp_subnames_final = [exp_subname for exp_subname in exp_subnames
                                      if merge_subname(exp_subname, args.merge_args, args.full) == rowname]
                
                # filter shot
                if not args.full or args.shot is not None:
                    exp_subnames_final = [exp_subname for exp_subname in exp_subnames_final
                                          if f'shot:{shot}' in exp_subname]
                    if '_shot:' in rowname:
                        if extract_arg_from_subname(rowname, 'shot') == int(shot):
                            vis_rowname = rowname.replace(f'_shot:{shot}', '')
                        else:
                            continue
                    else:
                        vis_rowname = rowname
                else:
                    vis_rowname = rowname

                # merge task name
                vis_rowname = vis_rowname.replace(f'task:{task}', '')

                # filter image size
                if not args.full or args.img_size is not None:
                    exp_subnames_final = [exp_subname for exp_subname in exp_subnames_final
                                          if f'is:{img_size}' in exp_subname]
                    if '_is:' in rowname:
                        if extract_arg_from_subname(rowname, 'is') == int(img_size):
                            vis_rowname = vis_rowname.replace(f'_is:{img_size}', '')
                        else:
                            continue

                # merge seeds
                if not args.full:
                    for i in range(5):
                        vis_rowname = vis_rowname.replace(f'_sid:{i}', '')

                # merge class names
                if args.class_wise:
                    vis_rowname = vis_rowname.replace(f'_class:{class_name}', '')

                # get results
                for exp_subname in exp_subnames_final:
                    result_dir = os.path.join(exp_dir, exp_subname, 'logs')
                    for result_path in glob.glob(os.path.join(result_dir, f'{dataset}_{task}*')):
                        if dataset.startswith('ap10k') and not result_path.endswith('.pth'):
                            continue
                        if dataset.startswith('ap10k'):
                            if args.class_wise:
                                if class_name == 'none':
                                    skip = False
                                    for class_name_ in AP10K.SPECIES:
                                        if class_name_ in result_path:
                                            skip = True
                                            break
                                    if skip:
                                        continue
                                elif (class_name not in result_path):
                                    continue
                        if dataset == 'linemod' and args.class_wise:
                            if class_name not in result_path:
                                continue

                        sid = int(exp_subname.split('_sid:')[1].split('_')[0])

                        result_name = os.path.split(result_path)[1]
                        if args.result_include is not None:
                            if sum([ri in result_name for ri in args.result_include]) < len(args.result_include):
                                continue
                        if args.result_exclude is not None:
                            if sum([re in result_name for re in args.result_exclude]) > 0:
                                continue

                        if dataset == 'davis2017':
                            result_ptf = result_name.replace(f'{dataset}_{task}_results_shot:{shot}', '').strip('_')
                            if args.class_wise:
                                result_path = os.path.join(result_path, 'per-sequence_results-val.csv')
                            else:
                                result_path = os.path.join(result_path, 'global_results-val.csv')
                        elif dataset == 'linemod':
                            result_ptf = result_name.replace(f'{dataset}_{task}', '').replace(f'_class:{class_name}', '').strip('_').replace('.pth', '')
                        else:
                            result_ptf = result_name.replace(f'{dataset}_{task}', '').strip('_').replace('.pth', '')

                        if result_ptf != '':
                            vis_rowname_ = vis_rowname + f', {result_ptf}'
                        else:
                            vis_rowname_ = vis_rowname

                        row = f'{exp_name} ({vis_rowname_})' if vis_rowname_ != '' else exp_name
                        if os.path.exists(result_path):
                            if dataset in ['eigen', 'nyud2']:
                                mode = 'min'
                            else:
                                mode = 'max'
                            if result_path.endswith('.csv'):
                                result = pd.read_csv(result_path)
                            else:
                                result = torch.load(result_path)

                            if dataset.startswith('ap10k'):
                                if args.class_wise:
                                    metric = column[-2]
                                else:
                                    metric = column[-1]
                                if metric == 'AP (↑)':
                                    value = result[0][1]
                                elif metric == 'AR (↑)':
                                    value = result[5][1]
                            elif dataset == 'davis2017':
                                if args.class_wise:
                                    value = 0
                                    n = 0
                                    for i in range(5):
                                        if f'{class_name}_{i+1}' in result['Sequence'].values:
                                            if column[-2] == 'J&F Mean (↑)':
                                                value_j = result['J-Mean'].loc[result['Sequence'] == f'{class_name}_{i+1}'].iloc[0]
                                                value_f = result['F-Mean'].loc[result['Sequence'] == f'{class_name}_{i+1}'].iloc[0]
                                                value += (value_j + value_f) / 2
                                            elif column[-2] == 'J Score (↑)':
                                                value += result['J-Mean'].loc[result['Sequence'] == f'{class_name}_{i+1}'].iloc[0]
                                            elif column[-2] == 'F Score (↑)':
                                                value += result['F-Mean'].loc[result['Sequence'] == f'{class_name}_{i+1}'].iloc[0]
                                            n += 1
                                    if n > 0:
                                        value /= n
                                    else:
                                        continue
                                else:
                                    if column[-1] == 'J&F Mean (↑)':
                                        value = result['J&F-Mean'].loc[0]
                                    elif column[-1] == 'J Score (↑)':
                                        value = result['J-Mean'].loc[0]
                                    elif column[-1] == 'F Score (↑)':
                                        value = result['F-Mean'].loc[0]
                            elif dataset == 'eigen':
                                value = result[1].cpu().item()
                            elif dataset == 'nyud2':
                                if column[-1] == 'Abs. Rel. (↓)':
                                    value = result[1].cpu().item()
                                elif column[-1] == 'RMSE (↓)':
                                    value = result[3].cpu().item()
                                else:
                                    raise NotImplementedError
                            elif dataset == 'linemod':
                                value = result[0].cpu().item()
                            elif dataset == 'isic2018':
                                if column[-1] == 'F1 (↑)':
                                    value = result[0] if isinstance(result, list) else result
                                elif isinstance(result, list):
                                    value = result[1]
                                else:
                                    continue
                            else:
                                raise NotImplementedError

                            if args.full:
                                tag = exp_subname
                                for i in range(5):
                                    tag = tag.replace(f'_sid:{i}', '')
                            else:
                                tag = exp_subname
                                for tag_ in rowname.split('_'):
                                    tag = tag.replace(f'{tag_}', '')
                                tag = tag.strip('_')

                            if args.lr is None or tag == f'lr:{args.lr}':
                                database = add_results(database, row, column, value, tag, mode, sid)

    database = process_dups(database, args.compact, args.subcompact, args.average, args.all_support_id)

    return database


def name_sort_function(index):
    priority = []
    for name in index:
        if name.startswith('VTM'):
            p1 = 0
        elif name.startswith('CVTM'):
            p1 = 1
        else:
            p1 = 2
        
        if 'LARGE' in name:
            p2 = 1
        else:
            p2 = 0

        if 'mv2:True' in name:
            p3 = 1
        else:
            p3 = 0

        priority.append((p1, p2, p3, name))
    
    return priority


def column_sort_function(index):
    if index.name == 'Class':
        return index
    else:
        return [0 for _ in index]


if True:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', default=None, choices=DOWNSTREAM_DATASETS)
    parser.add_argument('--root_dir', '-root', type=str, default='')
    parser.add_argument('--result_dir', '-rd', type=str, default='results')
    parser.add_argument('--exp_name', type=str, default=None, nargs='+')
    parser.add_argument('--exp_pattern', type=str, default=None)
    parser.add_argument('--exp_exclude', '-ee', type=str, default=None, nargs='+')
    parser.add_argument('--exp_exclude_pattern', '-eep', type=str, default=None, nargs='+')
    parser.add_argument('--result_include', '-ri', type=str, default=None, nargs='+')
    parser.add_argument('--result_exclude', '-re', type=str, default=None, nargs='+')
    parser.add_argument('--table_num', type=int, default=None)
    parser.add_argument('--n_classes', '-nc', type=int, default=-1)
    parser.add_argument('--chunk_id', '-cid', type=int, default=0)
    parser.add_argument('--shot', type=int, default=None, nargs='+')
    parser.add_argument('--img_size', type=int, default=None, nargs='+')
    parser.add_argument('--class_name', '-class', type=str, default=None, nargs='+')
    parser.add_argument('--subname_prefix', '-sprf', type=str, default=None)
    parser.add_argument('--subname_exclude_pattern', '-sep', type=str, default=None, nargs='+')
    parser.add_argument('--verbose', '-v', default=False, action='store_true')
    parser.add_argument('--compact', '-cp', default=False, action='store_true')
    parser.add_argument('--subcompact', '-scp', default=False, action='store_true')
    parser.add_argument('--full', '-f', default=False, action='store_true')
    parser.add_argument('--average', '-avg', default=False, action='store_true')
    parser.add_argument('--lr', type=str, default=None)
    parser.add_argument('--merge_args', type=str, default=['lr'], nargs='+')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--class_wise', '-cw', action='store_true', default=False)
    parser.add_argument('--single_channel_only', '-sco', action='store_true', default=False)
    parser.add_argument('--multi_channel_only', '-mco', action='store_true', default=False)
    parser.add_argument('--rep_score_only', '-rso', action='store_true', default=False)
    parser.add_argument('--all_support_id', '-asi', action='store_true', default=False)
    args = parser.parse_args()

    result_dir = os.path.join(args.root_dir, args.result_dir)

    # choose datasets to show
    if args.dataset is not None:
        datasets = args.dataset
    else:
        datasets = list(set([
            log.split('_')[0]
            for exp_name in safe_listdir(result_dir)
            for exp_subname in safe_listdir(os.path.join(result_dir, exp_name))
            for log in safe_listdir(os.path.join(result_dir, exp_name, exp_subname, 'logs'))
        ]))
        datasets = [dataset.replace('result', 'taskonomy') for dataset in datasets]
        datasets = [dataset for dataset in datasets if dataset in DOWNSTREAM_DATASETS]

    # construct a task dictionary
    task_dict = {dataset: dataset_dict[dataset] for dataset in datasets}

    try:
        pd.set_option('max_columns', None)
    except:
        pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.colheader_justify', 'left')
    database = create_database(args, task_dict)
    database = database[database.columns[database.isna().sum(axis=0) < len(database.index)]]
    database = database.loc[database.index[database.isna().sum(axis=1) < len(database.columns)]]
    database = database.sort_index(axis=0, key=name_sort_function)
    database = database.sort_index(axis=1, key=column_sort_function)
    database = database.reindex(sorted(database.columns,
                                       key=cmp_to_key(lambda x, y: DOWNSTREAM_DATASETS.index(x[0].lower()) - 
                                                                   DOWNSTREAM_DATASETS.index(y[0].lower()))), axis=1)
    
    print(database.to_string(justify='right'))
    if args.save:
        # current time
        os.makedirs('results', exist_ok=True)
        database.to_csv(f'results/result_temp.csv', sep='\t')
