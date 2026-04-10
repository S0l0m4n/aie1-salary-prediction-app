Data Science Jobs EDA
=====================
Let's go through the EDA checklist.

* Raw data: `df = pd.read_csv("w1_ds_salaries.csv")`

* df.info()
```
RangeIndex: 607 entries, 0 to 606
Data columns (total 12 columns):
 #   Column              Non-Null Count  Dtype      Description
---  ------              --------------  -----      -----------
 0   Unnamed: 0          607 non-null    int64      index
 1   work_year           607 non-null    int64      year salary was paid
 2   experience_level    607 non-null    object     EN = entry, MI = mid, SE = senior, EX = exec
 3   employment_type     607 non-null    object     PT = part, FT = full, CT = contract, FL = freelance
 4   job_title           607 non-null    object     name
 5   salary              607 non-null    int64      amount in currency (next)
 6   salary_currency     607 non-null    object     currency
 7   salary_in_usd       607 non-null    int64      adjusted salary in USD
 8   employee_residence  607 non-null    object     employee location (ISO country code)
 9   remote_ratio        607 non-null    int64      0 = on-site, 50 = hybrid, 100 = fully remote
 10  company_location    607 non-null    object     company location (ISO country code)
 11  company_size        607 non-null    object     S = <50, M = 50-250, L = >250
dtypes: int64(5), object(7)
```
No data is missing. Data is either string or int type.

* dup_count = df.duplicated().sum()
There are 42/607 duplicated rows, or 7 %. Should we remove them? It is plausible that some jobs might have identical data in terms of title, salary, location etc.

I decided to remove the duplicate rows where the salary (in the original currency) was very specific, not a multiple of 1000:
```
specific_salary = (df['salary'] % 1000 != 0)
specific_salary.sum()  # 142 specific salaries
dupes = df.duplicated(keep='first')
df_clean = df[~(specific_salary & dupes)]  # removed 16 duplicate entries, instead of 42
```

* df.describe()
We get statistics on the four int columns. (Not counting the first column, it's just an index.)
Use the USD salary column, as the other one uses various currencies.
Mean salary = $112k, std dev = $72k, min = $3k, max = $600k
The work year is not really useful, more informational.

* df['job_title'].value_counts(normalize=False)
Most common jobs are:
- data scientist (141)
- data engineer (128)
- data analyst (89)

* sns.histplot(df['salary_in_usd'], bins=30, kde=True)
Shows that the distribution skews to the right, median salary is $100k, mean of $112 skewed by a few large salaries.
