import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'repo/src/py')
import vertical_competition_analysis as v
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=v.OUT
root=json.loads((out/'warm_boundary_root.json').read_text())
allrows=[]
for f in out.glob('probabilities-*.csv'):
 df=v.pd.read_csv(f);df['source']=f.name;allrows.append(df)
summary=v.pd.concat(allrows,ignore_index=True)
summary.to_csv(out/'numerical_refinement_summary.csv',index=False)
obs=v.pd.read_csv(v.p.REPO/'data/analysis/left-boundary-timescales/timescales.csv')
v.p.style()
fig,axes=plt.subplots(1,2,figsize=(8.6/2.54,5.5/2.54),layout='constrained')
for ax,temp,letter in zip(axes,[300,.1],['a)','b)']):
 for dx,color,style in [(.01,'.6',':'),(.005,'darkgoldenrod','--'),(.0025,'darkslateblue','-'),(.00125,'#aa3333','-')]:
  g=summary.query('temperature_K == @temp and dx_um == @dx').sort_values(['tau_s','dt_s']).drop_duplicates('tau_s')
  g=g[g.p_vertical_final>1e-8].sort_values('omega_rad_s')
  if not len(g):continue
  ax.plot(g.omega_rad_s,g.p_vertical_final,style,color=color,lw=.9,marker='.',ms=2,label=rf'$\delta s={dx:g}$')
 data=obs.query('temperature_K == @temp and fv_final > 0 and omega < .25')
 ax.plot(data.omega,data.fv_final,'v',color='black',ms=3,label='saved data')
 ax.axhline(.01,color='.3',ls='--',lw=.6)
 ax.set_yscale('log');ax.set_ylim(1e-6,1);ax.set_xlim(0,.22 if temp==300 else .07)
 ax.set_xlabel(r'$\omega\,[\mathrm{rad/s}]$',fontsize=9)
 ax.set_title(rf'$T={temp:g}\,\mathrm{{K}}$',fontsize=10)
 ax.text(.04,.94,letter,transform=ax.transAxes,fontsize=10)
 ax.legend(frameon=False,fontsize=5.5,loc='lower left',handlelength=1)
axes[0].set_ylabel(r'$p_v(t_{\rm obs})$',labelpad=0)
axes[0].axvline(root['omega_rad_s'],color='darkslateblue',lw=.5,ls=':')
for ext in ['pdf','png']:fig.savefig(out/f'vertical_switching_probability.{ext}',dpi=300,bbox_inches='tight')
print('Saved probability comparison; warm model omega:',root['omega_rad_s'])
