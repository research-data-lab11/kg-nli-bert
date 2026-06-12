# -*- coding: utf-8 -*-
import pandas as pd


df = pd.read_csv('/path/to/your/workspace/aristo-tuple-kb-v4-mar2017/aristo-tuple-kb-v4-mar2017/COMBINED-KB.tsv', sep='\t')


spo_data = df[['Arg1', 'Pred', 'Arg2']]


spo_data.to_csv('AristoTuple_to_spo.spo', header=False, index=False, sep='\t')

