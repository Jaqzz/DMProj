import pandas as pd
import ast as ast
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('C:/Users/Elijah/Desktop/manga/manga.csv')
#not for a school project, thanks
df = df[df['sfw'] != False]
df2 = df
#remove irrelevant data to keep python from getting angry in the for loop
df.drop(['title', 'score', 'sfw', 'themes', 'manga_id', 'type', 'scored_by', 'status', 'volumes', 'chapters', 'start_date', 'end_date', 'members', 'favorites', 'approved', 'created_at_before', 'updated_at', 'real_start_date', 'real_end_date', 'demographics', 'authors', 'serializations', 'synopsis', 'background', 'main_picture', 'url', 'title_english', 'title_japanese', 'title_synonyms', 'jikan'], axis=1, inplace=True)

#whoever made this dataset and disguised the genres column as a column of lists when its entries are actually strings is going to hell
for i in df.index:
    df['genres'][i] = df['genres'][i].strip('][').split(', ')
lol = df['genres'].values.tolist()

#This code taken heavily from mlxtend documentation at https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
#formatting input
te = TransactionEncoder()
te_ary = te.fit(lol).transform(lol)
tfdf = pd.DataFrame(te_ary, columns=te.columns_)

apdf = apriori(tfdf, min_support=0.03, use_colnames=True)
apdf.to_csv('C:/Users/Elijah/Desktop/manga/apriori.csv')
ardf = association_rules(apdf, min_threshold=0.1)
ardf.to_csv('C:/Users/Elijah/Desktop/manga/association.csv')