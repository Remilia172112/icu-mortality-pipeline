# Ablation Study Results

## Missing Mask

| Factor | Value | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | Params | EpochsUsed | Time_s |
|---|---|---|---|---|---|---|---|---|---|---|
| Missing Mask | OFF | 0.9214 | 0.7019 | 0.618 | 0.6639 | 0.9395 | 0.5781 | 599297 | 12 | 46.1 |
| Missing Mask | ON | 0.9249 | 0.7114 | 0.6236 | 0.6982 | 0.9325 | 0.5635 | 669953 | 14 | 59.8 |

## pos_weight

| Factor | Value | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | Params | EpochsUsed | Time_s |
|---|---|---|---|---|---|---|---|---|---|---|
| pos_weight | 1.0 | 0.922 | 0.702 | 0.6045 | 0.5082 | 0.9784 | 0.7459 | 669953 | 15 | 64.8 |
| pos_weight | 3.0 | 0.9249 | 0.7123 | 0.6241 | 0.6892 | 0.9352 | 0.5703 | 669953 | 11 | 45.6 |
| pos_weight | 5.0 | 0.9233 | 0.7103 | 0.5816 | 0.7894 | 0.8845 | 0.4604 | 669953 | 13 | 55.5 |
| pos_weight | 8.0 | 0.9236 | 0.7093 | 0.5531 | 0.8232 | 0.856 | 0.4164 | 669953 | 13 | 55.2 |

## Hidden Size

| Factor | Value | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | Params | EpochsUsed | Time_s |
|---|---|---|---|---|---|---|---|---|---|---|
| Hidden Size | 64 | 0.9218 | 0.704 | 0.6192 | 0.6496 | 0.944 | 0.5915 | 203905 | 13 | 54.1 |
| Hidden Size | 128 | 0.9236 | 0.7115 | 0.6197 | 0.704 | 0.9291 | 0.5535 | 669953 | 11 | 47.7 |
| Hidden Size | 256 | 0.9232 | 0.7136 | 0.631 | 0.6607 | 0.9459 | 0.604 | 2388481 | 11 | 123.4 |

## n_layers

| Factor | Value | AUROC | AUPRC | F1 | Sensitivity | Specificity | PPV | Params | EpochsUsed | Time_s |
|---|---|---|---|---|---|---|---|---|---|---|
| n_layers | 1 | 0.925 | 0.7128 | 0.6317 | 0.6549 | 0.9478 | 0.6101 | 274689 | 15 | 46.9 |
| n_layers | 2 | 0.9228 | 0.7104 | 0.6283 | 0.6807 | 0.9393 | 0.5834 | 669953 | 13 | 53.2 |
| n_layers | 3 | 0.9225 | 0.7083 | 0.6212 | 0.6876 | 0.9343 | 0.5665 | 1065217 | 12 | 67.7 |

