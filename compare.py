import os

n_last_epoch = 10
exp_dir = 'exps/MSRVTT_jsfusion_clspace_cls_lambd_30_check'

log_path = os.path.join(exp_dir, 'log.txt')
with open(log_path, 'r') as f:
    raw_lines = f.readlines()

    t2v_metrics = {
        "R1": [],
        "R5": [],
        "R10": [],
        "R50": [],
        "MedR": [],
        "MeanR": [],
        "geometric_mean_R1-R5-R10": []
    }
    v2t_metrics = {
        "R1": [],
        "R5": [],
        "R10": [],
        "R50": [],
        "MedR": [],
        "MeanR": [],
        "geometric_mean_R1-R5-R10": []
    }

    for line in raw_lines:
        log_info = line.lstrip().rstrip('\n')
        if "MSRVTT_jsfusion_test/t2v_metrics" in log_info:
            tag, score = log_info.split(': ')
            tag = tag.split('/')[-1]
            score = float(score)
            t2v_metrics[tag].append(score)
        elif "MSRVTT_jsfusion_test/v2t_metrics" in log_info:
            tag, score = log_info.split(': ')
            tag = tag.split('/')[-1]
            score = float(score)
            v2t_metrics[tag].append(score)
    
    print(os.path.basename(exp_dir) + ':')
    for key in t2v_metrics:
        print('\tt2v_metrics/{}: {:.2f}'.format(key, sum(t2v_metrics[key][-n_last_epoch:]) / n_last_epoch))
    for key in v2t_metrics:
        print('\tt2v_metrics/{}: {:.2f}'.format(key, sum(v2t_metrics[key][-n_last_epoch:]) / n_last_epoch))
