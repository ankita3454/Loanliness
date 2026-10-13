# Loanliness: Predicting Loan Repayment Ability

UE24CS352A Machine Learning mini-project. This project replicates and extends
[*Loanliness: Predicting Loan Repayment Ability by Using Machine Learning Methods*](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26644913.pdf)
(Liang, Jin, Wang, Stanford CS229, 2019).

**Problem.** Banks judge loan applicants mostly by credit history, so students
and other people without that history get shut out. Can other information
(employment, income, previous applications, payment behaviour) predict who
will have trouble repaying?

**Data.** Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
has 307,511 labelled applicants across 7 linked tables. `TARGET = 1` means the
client had payment difficulties (about 8% of applicants).

## What we do

| | Paper | This project |
|---|---|---|
| Rows | Drops any row with a missing value (307K → 102K) | **Replication:** same recipe. **Improved:** keeps all 307K; LightGBM handles NaN natively |
| Features | Concatenate 7 tables, one-hot, polynomial | Same, plus engineered ratios (loan/income, repayment/income, late-payment days, card utilisation) |
| Imbalance | Down-sampling (best), SMOTE, class weights | All three compared on the *natural* test set |
| Metric | Accuracy on a balanced test set | That (to compare with the paper), plus **ROC-AUC / PR-AUC** on the real 8% default rate |
| Models | LR, RF, Naive Bayes, MLP, LightGBM, K-means + per-cluster LightGBM | Same, plus tuned LightGBM with 5-fold CV and early stopping |
| Extra | PCA / t-SNE, RF feature importance | Same, plus ablations (no external credit scores, application table only) and a **thin-file** analysis of applicants with no credit-bureau record |
| Demo | None | Streamlit app that scores an applicant and explains the score |

Final numbers are written to [`results/SUMMARY.md`](results/SUMMARY.md) after a run.

## Setup

Requires Python 3.10 or newer.

```bash
git clone <this-repo> && cd loanliness
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**macOS:** LightGBM needs OpenMP. If `import lightgbm` fails, run `brew install libomp`.

### Get the data (about 2.5 GB unzipped)

1. Log in on Kaggle, open the [competition](https://www.kaggle.com/c/home-credit-default-risk/rules) and accept the rules.
2. Download the data, either with the CLI (needs `~/.kaggle/kaggle.json`):
   ```bash
   pip install kaggle
   kaggle competitions download -c home-credit-default-risk -p data
   unzip data/home-credit-default-risk.zip -d data
   ```
   or by downloading the zip from the Data tab and unzipping it into `data/`.

`data/` should then contain `application_train.csv`, `bureau.csv`,
`bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`,
`installments_payments.csv` and `credit_card_balance.csv`. The other files are not used.

## Run

```bash
python run_all.py --sample 60000 --fast   # quick check, about 5 min
python run_all.py                          # full run, about 30-60 min on a laptop
```

Or run each step on its own:

```bash
python -m loanliness.features     # join 7 tables -> artifacts/features.pkl
python -m loanliness.replicate    # Experiment 1: paper replication
python -m loanliness.visualize    # PCA, t-SNE, random-forest importance
python -m loanliness.improve      # Experiment 2: improved LightGBM, ablations, demo model
```

Outputs:
- `results/SUMMARY.md`: paper vs ours, headline numbers
- `results/replication.md`, `results/improved.md`: full metric tables
- `results/figures/*.png`: ROC/PR curves, feature importance, PCA/t-SNE, K-means, ablation chart

### Live demo

```bash
streamlit run app.py
```

Enter an applicant's details to see the predicted probability of payment difficulty
and the factors that pushed it up or down (LightGBM SHAP contributions). Untick
the external scores to simulate a thin-file applicant. The **Results** tab shows all figures and tables.

### Tests

```bash
pytest -q
```

The tests build small synthetic tables with the same schema as the Kaggle data,
so they run in seconds without the download. The synthetic numbers are random,
so never report them.

## Repository layout

```
loanliness/
  config.py      paths, seed, paper settings
  features.py    load + aggregate the 7 tables into one row per applicant
  replicate.py   Experiment 1: paper cleaning, 5 models, balancing, K-means
  visualize.py   PCA, t-SNE, random-forest importance
  improve.py     Experiment 2: LightGBM CV, logistic baseline, ablations, demo model
  common.py      metrics (paper-style tables), plotting
app.py           Streamlit demo
run_all.py       runs everything and writes results/SUMMARY.md
tests/           synthetic-data generator + pytest smoke tests
```

## Method notes

- **No leakage:** the train/test split comes first. Scaling, imputation, SMOTE and
  down-sampling are fit on training data only. The decision threshold is picked
  on out-of-fold training predictions, never on the test set.
- **Why ROC-AUC:** with 8% defaults, predicting "everyone repays" scores 92%
  accuracy and catches nobody. AUC measures how well the model *ranks* risky
  applicants regardless of threshold.
- **`DAYS_EMPLOYED = 365243`** is a placeholder for pensioners and unemployed applicants.
  It is converted to NaN plus a flag column instead of being treated as 1,000 years of employment.
- **`EXT_SOURCE_1/2/3`** are normalised scores from external credit bureaus. They are
  the strongest predictors but also exactly what thin-file applicants lack, so the ablation
  reports performance with and without them.
- **Fairness:** the demo model deliberately leaves out gender.

## Team

| Member | SRN | Contribution |
|---|---|---|
| Anisha | PES1UG24CS064 | |
| Ankita Pramod | PES1UG24CS065 | |

## References

1. Liang, Jin, Wang. *Loanliness: Predicting Loan Repayment Ability by Using Machine Learning Methods.* CS229, Stanford, 2019.
2. Home Credit Default Risk, Kaggle, 2018.
3. Ke et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
4. Chawla et al. *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR 2002.
