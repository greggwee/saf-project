# Usage: python -I data/scripts/build_sp500_pit_ids.py <dir with msp500/secinfo/ccm/ibes/ciq .parquet> data/universe
# Inputs = Drive files renamed: crsp_msp500list_v2->msp500, crsp_stksecurityinfohist->secinfo,
#   crsp_ccmxpf_lnkhist->ccm, wrdsapps_ibcrsphist->ibes, ciq_wrds_gvkey->ciq
import pandas as pd, sys
D=sys.argv[1]; O=sys.argv[2]
P=lambda c: pd.to_datetime(c, errors='coerce')
m=pd.read_parquet(f'{D}/msp500.parquet'); s=pd.read_parquet(f'{D}/secinfo.parquet')
c=pd.read_parquet(f'{D}/ccm.parquet'); i=pd.read_parquet(f'{D}/ibes.parquet'); q=pd.read_parquet(f'{D}/ciq.parquet')
for df,cols in [(m,['mbrstartdt','mbrenddt']),(s,['secinfostartdt','secinfoenddt']),(c,['linkdt','linkenddt']),(i,['sdate','edate']),(q,['startdate','enddate'])]:
    for k in cols: df[k]=P(df[k])
FAR=pd.Timestamp('2262-01-01')
m=m[m.mbrenddt>='2005-01-01'].copy()
m['still_member']=m.mbrenddt>=pd.Timestamp('2026-06-30')
c=c[c.linktype.isin(['LU','LC'])].copy(); c['linkenddt']=c.linkenddt.fillna(FAR)
q['enddate']=q.enddate.fillna(FAR); q['startdate']=q.startdate.fillna(pd.Timestamp('1900-01-01'))
s['secinfoenddt']=s.secinfoenddt.fillna(FAR)
# segments: membership spell x secinfo interval
x=m.merge(s,on='permno')
x['seg_start']=x[['mbrstartdt','secinfostartdt']].max(axis=1); x['seg_end']=x[['mbrenddt','secinfoenddt']].min(axis=1)
x=x[x.seg_start<=x.seg_end].copy().reset_index(drop=True)
def asof(seg, tbl, key, start, end, rank):
    # pick best row overlapping segment, preferring one covering seg_end
    t=seg[['permno','seg_start','seg_end']].reset_index().merge(tbl,on=key)
    t=t[(t[start]<=t.seg_end)&(t[end]>=t.seg_start)].copy()
    t['_cov']=(t[start]<=t.seg_end)&(t[end]>=t.seg_end)
    t=t.sort_values(['index','_cov']+rank,ascending=False).drop_duplicates('index')
    return t.set_index('index')
cc=asof(x,c,'permno','linkdt','linkenddt',[])  # LC before LU, P before C handled below
cc=c.assign(_r=c.linktype.map({'LC':1,'LU':0}).fillna(0)+c.linkprim.map({'P':2,'C':1,'J':0.5}).fillna(0))
g=asof(x,cc,'permno','linkdt','linkenddt',['_r'])
x['gvkey']=g['gvkey']
ib=asof(x,i.assign(score=i.score.fillna(9)*-1),'permno','sdate','edate',['score'])
x['ibes_ticker']=ib['ticker']; x['ibes_ncusip']=ib['ncusip']
x2=x.reset_index().rename(columns={'index':'i'}).merge(q[['gvkey','companyid','startdate','enddate','primaryflag']],on='gvkey',how='left')
x2=x2[x2.companyid.isna()|((x2.startdate<=x2.seg_end)&(x2.enddate>=x2.seg_start))].sort_values(['i','primaryflag'],ascending=[True,False]).drop_duplicates('i')
x=x.reset_index(drop=True); x['ciq_companyid']=x2.set_index('i')['companyid'].reindex(x.index).astype('Int64') if False else x2.set_index('i')['companyid'].reindex(range(len(x))).values
x['ciq_companyid']=pd.array(x['ciq_companyid'],dtype='Int64')
out=x[['permno','permco','gvkey','ciq_companyid','ibes_ticker','ibes_ncusip','ticker','tradingsymbol','cusip','cusip9','hdrcusip','hdrcusip9','securitynm','issuernm','primaryexch','shareclass','siccd','seg_start','seg_end','mbrstartdt','mbrenddt','still_member','securityhdrflg']].rename(columns={'ticker':'crsp_ticker','tradingsymbol':'crsp_tradingsymbol','cusip':'cusip8','seg_start':'valid_from','seg_end':'valid_to','mbrstartdt':'member_from','mbrenddt':'member_to'})
out=out.sort_values(['permno','valid_from']).reset_index(drop=True)
idc=['gvkey','ciq_companyid','ibes_ticker','ibes_ncusip','crsp_ticker','crsp_tradingsymbol','cusip8','cusip9']
fb=q.sort_values(['primaryflag','enddate'],ascending=False).drop_duplicates('gvkey').set_index('gvkey')['companyid']
out['ciq_companyid']=out['ciq_companyid'].fillna(out['gvkey'].map(fb)).astype('Int64')
out['ids_filled']=out[idc].isna().any(axis=1)
out[idc]=out.groupby('permno')[idc].transform(lambda v:v.ffill().bfill())
out.loc[out.valid_to>=pd.Timestamp('2026-06-30'),'valid_to_open']=True
out.to_csv(f'{O}/sp500_pit_identifiers_segments.csv',index=False,date_format='%Y-%m-%d')
# permno-level summary
agg=out.groupby('permno').agg(permco=('permco','last'),gvkey=('gvkey',lambda v:'|'.join(pd.unique(v.dropna()))),ciq_companyid=('ciq_companyid',lambda v:'|'.join(map(str,pd.unique(v.dropna())))),ibes_ticker=('ibes_ticker',lambda v:'|'.join(pd.unique(v.dropna()))),
  tickers=('crsp_ticker',lambda v:'|'.join(pd.unique(v.dropna()))),cusip8=('cusip8',lambda v:'|'.join(pd.unique(v.dropna()))),cusip9=('cusip9',lambda v:'|'.join(pd.unique(v.dropna()))),
  last_name=('securitynm','last'),first_member=('member_from','min'),last_member=('member_to','max')).reset_index()
agg.to_csv(f'{O}/sp500_pit_identifiers_by_permno.csv',index=False,date_format='%Y-%m-%d')
# daily-membership helper: spells
m[['permno','mbrstartdt','mbrenddt','mbrflg','still_member']].to_csv(f'{O}/sp500_membership_spells.csv',index=False,date_format='%Y-%m-%d')
print(len(out),out.permno.nunique(),len(agg))
for k in ['permco','gvkey','ciq_companyid','ibes_ticker','crsp_ticker','cusip8','cusip9']: print(k,'null%',round(out[k].isna().mean()*100,1))
print(agg.permno.nunique(), (agg.gvkey=='').sum(),(agg.ciq_companyid=='').sum(),(agg.ibes_ticker=='').sum())
