import os
import pandas as pd
import matplotlib.pyplot as plt


def save_precision_recall_curves(precision_recall_dict, dataloader_idx):
    fig, ax = plt.subplots()

    for key, value in precision_recall_dict.items():
        df = value['df']
        auc = value['auc']
        ax.plot(df['Recall'], df['Precision'], label=f"{key} (AUC = {auc:.3f})")
    
    if dataloader_idx == 0:
        dataset_balance = 0.5
    elif dataloader_idx == 1:
        dataset_balance = 0.1 #TODO

    ax.plot([0, 1], [dataset_balance, dataset_balance], linestyle='--', color='gray', label='Random Classifier')

    ax.set_ylim(0, 1)
    ax.set_title(f'Precision Recall T{dataloader_idx+1}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='best', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'./combined_reports/precision_recall_curves_T{dataloader_idx+1}')

def save_roc_curves(precision_recall_dict, dataloader_idx):
    fig, ax = plt.subplots()

    for key, value in precision_recall_dict.items():
        df = value['df']
        auc = value['auc']
        ax.plot(df['FPR'], df['TPR'], label=f"{key} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    ax.set_ylim(0, 1)
    ax.set_title(f'ROC T{dataloader_idx+1}')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(loc='best', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'./combined_reports/roc_curves_T{dataloader_idx+1}')

def main():
    current_directory = os.getcwd()

    if not os.path.exists('combined_reports'):
        os.mkdir('combined_reports')
    
    pr_T1_dict = {}
    pr_T2_dict = {}
    roc_T1_dict = {}
    roc_T2_dict = {}

    for model in os.listdir(current_directory):
        item_path = os.path.join(current_directory, model)
        if not os.path.isdir(item_path):
            continue

        if model == 'combined_reports':
            continue

        metrics = pd.read_csv(f'./{model}/reports/human/metrics.csv')
        pr_T1_dict[model] = {
            'df': pd.read_csv(f'./{model}/reports/human/precision_recall_curve_data_T1.csv'),
            'auc': metrics['auprc_T1'].item()
        }
        pr_T2_dict[model] = {
            'df': pd.read_csv(f'./{model}/reports/human/precision_recall_curve_data_T2.csv'),
            'auc': metrics['auprc_T2'].item()
        }
        roc_T1_dict[model] = {
            'df': pd.read_csv(f'./{model}/reports/human/roc_curve_data_T1.csv'),
            'auc': metrics['auroc_T1'].item()
        }
        roc_T2_dict[model] = {
            'df': pd.read_csv(f'./{model}/reports/human/roc_curve_data_T2.csv'),
            'auc': metrics['auroc_T2'].item()
        }

    save_precision_recall_curves(pr_T1_dict, 0)
    save_precision_recall_curves(pr_T2_dict, 1)

    save_roc_curves(roc_T1_dict, 0)
    save_roc_curves(roc_T2_dict, 1)

if __name__ == '__main__':
    main()