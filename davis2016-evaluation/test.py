import argparse
import os
import subprocess
import glob
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', '-td', type=str, default='TEST')
parser.add_argument('--dataset', type=str, default='davis2017')
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--exp_subname', '-sname', type=str, default=None)
parser.add_argument('--result_dir', '-rdir', type=str, default=None)
parser.add_argument('--shot', type=int, default=None)
parser.add_argument('--test_anyway', '-ta', default=False, action='store_true')
parser.add_argument('--reset_mode', '-reset', default=False, action='store_true')
args = parser.parse_args()


with open('data_paths.yaml') as f:
    path_dict = yaml.safe_load(f)

year = args.dataset.strip('davis')
args.gt_dir = path_dict[f'davis{year}'].replace(year, '')
if year == '2016':
    n_classes = 20
    subdir = ''
elif year == '2017':
    n_classes = 30
    subdir = 'trainval'


def check_dir(path):
    if args.test_anyway or args.reset_mode:
        return True
    
    if len(os.listdir(path)) != n_classes:
        return False
    
    for class_name in os.listdir(path):
        n_src = len(os.listdir(os.path.join(path, class_name)))
        n_tgt = len(os.listdir(os.path.join(args.gt_dir, year, subdir, 'JPEGImages', '480p', class_name))) 
        if n_src != n_tgt:
            return False
        
    return True


def check_result_dir(result_dir):
    if args.shot is not None:
        target_name = f'{args.dataset}_vos_results_shot:{args.shot}'
        if result_dir == target_name:
            return True
        else:
            return False
    else:
        target_name = f'{args.dataset}_vos_results_shot:'
        if result_dir.startswith(target_name):
            return True
        else:
            return False


if args.result_dir is None:
    if args.exp_name is None:
        exp_names = os.listdir(os.path.join('experiments', args.test_dir))
        exp_names = [exp_name for exp_name in exp_names
                    if len(glob.glob(os.path.join('experiments', args.test_dir, exp_name, '*', 'logs', f'{args.dataset}*'))) > 0]
    else:
        exp_names = [args.exp_name]

    for exp_name in exp_names:
        exp_dir = os.path.join('experiments', args.test_dir, exp_name)
        if args.exp_subname is not None:
            exp_subnames = [args.exp_subname]
        else:
            exp_subnames_all = os.listdir(exp_dir)
            exp_subnames = []
            for exp_subname in exp_subnames_all:
                for result_dir in os.listdir(os.path.join(exp_dir, exp_subname, 'logs')):
                    if check_result_dir(result_dir):
                        if check_dir(os.path.join(exp_dir, exp_subname, 'logs', result_dir)):
                            exp_subnames.append((exp_subname, result_dir))

        os.chdir('davis2016-evaluation')
        for exp_subname, result_dir in exp_subnames:
            print(f'Processing {exp_subname}')
            command = f"python evaluation_method.py --results_path {os.path.join('..', exp_dir, exp_subname, 'logs', result_dir)} -y {year}"
            if args.reset_mode:
                command += ' -reset'
            print(command)
            subprocess.call(command.split())
        os.chdir('..')
else:
    os.chdir('davis2016-evaluation')
    command = f"python evaluation_method.py --results_path {os.path.join('..', args.result_dir)} -y {year}"
    if args.reset_mode:
        command += ' -reset'
    print(command)
    subprocess.call(command.split())
    os.chdir('..')
