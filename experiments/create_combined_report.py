import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

def save_precision_recall_curves(precision_recall_dict, dataloader_idx):
    fig, ax = plt.subplots()

    for key, value in precision_recall_dict.items():
        df = value['df']
        auc = value['auc']
        ax.plot(df['Recall'], df['Precision'], label=f"{key} (AUC = {auc:.3f})")
    
    if dataloader_idx == 0:
        dataset_balance = 0.5 # TODO: dont hardcode this
    elif dataloader_idx == 1:
        dataset_balance = 0.0909 # TODO: dont hardcode this

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

def save_metric_table(df, dataloader_idx):
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed

    # Hide the axis
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    ax.set_frame_on(False)

    # Add the row names (index) as a column in the table
    df_with_index = df.reset_index()

    # Create a table
    tbl = table(ax, df_with_index, loc='center', cellLoc='center', colWidths=[0.2]*len(df_with_index.columns))

    # Style the table (optional)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)  # Scale table size (optional)
    plt.savefig(f'./combined_reports/metrics_T{dataloader_idx+1}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

def main():
    current_directory = os.getcwd()

    if not os.path.exists('combined_reports'):
        os.mkdir('combined_reports')
    
    pr_T1_dict = {}
    pr_T2_dict = {}
    roc_T1_dict = {}
    roc_T2_dict = {}

    metric_data_T1 = {}
    metric_data_T2 = {}

    metric_index = [
        'accuracy',
        'auprc',
        'auroc',
        'f1',
        'false_negatives',
        'false_positives',
        'precision',
        'recall',
        'true_negatives',
        'true_positives',
        'weighted_f1'
    ]

    for model in os.listdir(current_directory):
        item_path = os.path.join(current_directory, model)
        if not os.path.isdir(item_path):
            continue

        if model == 'combined_reports':
            continue

        metrics = pd.read_csv(f'./{model}/reports/human/metrics.csv')
        metric_data_T1[model] = [round(metrics[f'{metric}_T1'].item(), 3) for metric in metric_index]
        metric_data_T2[model] = [round(metrics[f'{metric}_T2'].item(), 3) for metric in metric_index]

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

    save_metric_table(
        pd.DataFrame(metric_data_T1, index=metric_index),
        0
    )
    save_metric_table(
        pd.DataFrame(metric_data_T2, index=metric_index),
        1
    )

    save_precision_recall_curves(pr_T1_dict, 0)
    save_precision_recall_curves(pr_T2_dict, 1)

    save_roc_curves(roc_T1_dict, 0)
    save_roc_curves(roc_T2_dict, 1)

if __name__ == '__main__':
    main()