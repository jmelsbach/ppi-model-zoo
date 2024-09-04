import pandas as pd
import random


def create_test_split(dataset: pd.DataFrame, protein_hubs: list, test_protein_rate: float = 0.07, complete_overlap_rate: float = 0.05, seed: int = 864) -> tuple:
    random.seed(seed)
    # todo(diego): comment and explain 
    all_prot_ids = dataset['uniprotID_A'].unique(
    ).tolist() + dataset['uniprotID_B'].unique().tolist()
    hubs = sorted(list(set(protein_hubs) & set(all_prot_ids)))
    non_hubs = sorted(list(set(all_prot_ids) - set(protein_hubs)))

    test_hubs = random.sample(
        hubs,
        int(test_protein_rate * len(hubs))
    )
    test_non_hubs = random.sample(
        non_hubs,
        int(test_protein_rate * len(non_hubs))
    )
    test_proteins = test_hubs + test_non_hubs

    a_is_test_prot = dataset['uniprotID_A'].isin(test_proteins).astype(int)
    b_is_test_prot = dataset['uniprotID_B'].isin(test_proteins).astype(int)
    test_mask = a_is_test_prot + b_is_test_prot

    test = dataset.loc[test_mask >= 1]
    # no or partial overlap
    train = dataset.loc[test_mask == 0]

    # add complete overlap
    complete_overlap_subset = train.sample(
        n=int(complete_overlap_rate * len(train)),
        replace=False
    )

    train = train.drop(complete_overlap_subset.index)
    test = pd.concat([test, complete_overlap_subset], axis=0)

    return train, test
