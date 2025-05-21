import pandas as pd


def gen_agg_cols(df: pd.DataFrame) -> pd.DataFrame:
    cache = df['-cache:dl1'].apply(lambda x: x.split(':'))
    df['cache ul1: nsets'] = cache.apply(lambda x: int(x[1])).astype(int)
    df['cache ul1: blocksize'] = cache.apply(lambda x: int(x[2])).astype(int)
    df['cache ul1: associativity'] = cache.apply(lambda x: int(x[3])).astype(int)
    return df
    
df = gen_agg_cols(pd.read_csv('out.csv'))
sep = ["benchmark", "cache ul1: nsets","cache ul1: blocksize","cache ul1: associativity"]
dup = df.drop_duplicates(sep)
print("df:", df.shape)
print("fil:", dup.shape)