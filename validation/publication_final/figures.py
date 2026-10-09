"""Six publication figures regenerated solely from public aggregate tables."""
import argparse
import json
from pathlib import Path
import shutil
import re
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE='#0072B2';ORANGE='#D55E00';GRAY='#666666';GREEN='#009E73'
CONDITIONS=['independent','linear','nonlinear_marker','heteroskedastic','correlated','z_dependent_censoring','nonph_linear','nonph_nonlinear','nonph_heteroskedastic','nonlinear_clinical_risk']
LABELS=['Independent','Linear marker mean','Nonlinear marker mean','Heteroskedastic markers','Correlated markers','Z-dependent censoring','Time reversal + linear','Time reversal + nonlinear','Time reversal + heteroskedastic','Nonlinear clinical risk']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':320})


def pair(value):return json.loads(value) if isinstance(value,str) else None
def interval(k,n):return [0 if k==0 else beta.ppf(.025,k,n-k+1),1 if k==n else beta.ppf(.975,k+1,n-k)]
def panel(ax,label,title):
    ax.set_title(label+'  '+title,loc='left',pad=11,fontweight='bold');ax.spines[['top','right']].set_visible(False)
def save_figure(fig,out,index,table):
    for suffix in ('png','pdf','svg'):fig.savefig(out/f'Figure{index}.{suffix}',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else None)
    svg = out/f'Figure{index}.svg'
    svg.write_text(re.sub(r'<!DOCTYPE[^>]*>', '', svg.read_text(), flags=re.S))
    table.to_csv(out/f'Figure{index}-source.csv',index=False);plt.close(fig)


def run(inputs,output):
    output.mkdir(parents=True,exist_ok=False)
    v2=pd.read_csv(inputs/'v2.csv');v3=pd.read_csv(inputs/'v3_main.csv');tasks=pd.read_csv(inputs/'tool_tasks.csv')
    # Figure 1 is a state-flow schematic; source rows define every labelled edge.
    fig,ax=plt.subplots(figsize=(8.4,5.0));ax.set(xlim=(-.2,10.2),ylim=(0,6));ax.axis('off')
    nodes={
      'training':(1.5,5,'Training data\nDeclare event and variable roles'),
      'transform':(5,5,'Training transformations\nImpute, encode and fix knots'),
      'diagnosis':(8.5,5,'Model and residual diagnostics\nCheck method qualification'),
      'allowed':(8.5,3.2,'Diagnostics do not reject\nQualified method profile required\nAssumption-dependent inference'),
      'held':(5,3.2,'Test or qualification fails\nInference withheld'),
      'raw':(1.5,3.2,'Exploratory calculations retained\nStandard p and q are null'),
      'export':(3.3,1.1,'API, screen, CSV and report\nState and reasons remain visible'),
      'recipe':(7.5,1.1,'Locked external prediction\nFixed state and transformations\nWithholding remains')}
    for key,(x,y,label) in nodes.items():
        ax.add_patch(FancyBboxPatch((x-1.55,y-.48),3.1,.96,boxstyle='round,pad=.03',facecolor='#FFF3EC' if key in ('held','raw') else '#F0F6FA',edgecolor=ORANGE if key in ('held','raw') else BLUE,lw=1.0))
        ax.text(x,y,label,ha='center',va='center',fontsize=8.4)
    edges=[('training','transform'),('transform','diagnosis'),('diagnosis','allowed'),('diagnosis','held'),('held','raw'),('held','export'),('raw','export'),('allowed','recipe'),('export','recipe')]
    for left,right in edges:
        x,y,_=nodes[left];u,v,_=nodes[right];dx=u-x;dy=v-y
        sx=x+(1.6*np.sign(dx) if abs(dx)>abs(dy) else 0);sy=y+(.53*np.sign(dy) if abs(dy)>=abs(dx) else 0)
        ex=u-(1.6*np.sign(dx) if abs(dx)>abs(dy) else 0);ey=v-(.53*np.sign(dy) if abs(dy)>=abs(dx) else 0)
        ax.annotate('',xy=(ex,ey),xytext=(sx,sy),arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.1))
    ax.text(5,.08,'An estimable prediction does not establish valid marker inference.',ha='center',fontsize=9)
    save_figure(fig,output,1,pd.DataFrame([dict(source=l,target=r,source_label=nodes[l][2],target_label=nodes[r][2]) for l,r in edges]))
    # Figure 2 retains all ten v2 global-null conditions and both partial nulls.
    main=v2[(v2.stage=='main')&(v2.n==180)&(v2.p==30)]
    fig,axes=plt.subplots(2,2,figsize=(9.3,8.0),gridspec_kw={'width_ratios':[1.25,1]});y=np.arange(10)
    methods=[('legacy_linear',GRAY,'Legacy raw'),('guarded_linear',BLUE,'v2 linear'),('guarded_spline',ORANGE,'v2 spline')]
    for method,color,label in methods:
        rows=main.set_index(['condition','method']);data=[rows.loc[(c,method)] for c in CONDITIONS];offset=(-.17 if method=='legacy_linear' else 0 if method=='guarded_linear' else .17)
        val=np.array([r.fwer for r in data])*100;limits=np.array([pair(r.fwer_mc95) for r in data])*100
        axes[0,0].errorbar(val,y+offset,xerr=np.maximum(0,np.vstack([val-limits[:,0],limits[:,1]-val])),fmt='o',ms=3,color=color,label=label,lw=.8)
        if method=='legacy_linear':continue
        valid=[i for i,r in enumerate(data) if pd.notna(r.conditional_fwer)]
        value=np.array([data[i].conditional_fwer for i in valid])*100;ci=np.array([pair(data[i].conditional_mc95) for i in valid])*100
        axes[0,1].errorbar(value,y[valid]+offset,xerr=np.maximum(0,np.vstack([value-ci[:,0],ci[:,1]-value])),fmt='o',ms=3,color=color,lw=.8)
        for i in set(range(10))-set(valid):axes[0,1].text(82,y[i]+offset,'NA',color=color,fontsize=7,va='center')
        value=np.array([r.allowed_fraction for r in data])*100;ci=np.array([interval(int(r.allowed),int(r.planned)) for r in data])*100
        axes[1,0].errorbar(value,y+offset,xerr=np.maximum(0,np.vstack([value-ci[:,0],ci[:,1]-value])),fmt='o',ms=3,color=color,lw=.8,label=label)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_yticks(y,LABELS if ax is not axes[0,1] else []);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0,0].set_xscale('symlog',linthresh=1);axes[0,0].set_xticks([0,1,5,20,100],['0','1','5','20','100']);axes[0,0].set(xlabel='FWER (%)',xlim=(-.1,110));axes[0,0].legend(fontsize=7,loc='upper right')
    axes[0,1].set_xscale('symlog',linthresh=1);axes[0,1].set_xticks([0,1,5,20,100],['0','1','5','20','100']);axes[0,1].set(xlabel='FWER among allowed analyses (%)',xlim=(-.1,105));axes[0,1].axvline(6,color=GRAY,ls='--',lw=.7)
    axes[1,0].set(xlabel='Allowed analyses (%)',xlim=(-2,102));axes[1,0].axvline(80,color=GRAY,ls='--',lw=.7)
    power=pd.read_csv(inputs/'power-retention-uncertainty.csv')
    for j,(method,color,label) in enumerate(methods[1:]):
        r=power[power.method==method].set_index('condition').loc[['partial_weak','partial_strong']]
        values=r.ratio.to_numpy()*100;ci=r[['paired_delta_MC95_lower','paired_delta_MC95_upper']].to_numpy()*100
        axes[1,1].errorbar([0+j*.15,1+j*.15],values,yerr=np.vstack([values-ci[:,0],ci[:,1]-values]),fmt='o',color=color,label=label,capsize=3)
    axes[1,1].set(xticks=[.075,1.075],xticklabels=['Weak signal','Strong signal'],ylabel='Power retained vs legacy (%)',ylim=(70,100));axes[1,1].axhline(90,color=GRAY,ls='--',lw=.7);axes[1,1].legend(fontsize=8)
    for ax,label,title in zip(axes.flat,'ABCD',['Raw and reported errors','Errors among allowed analyses','Normal and unsuitable analyses retained','Partial-null power retention']):panel(ax,label,title)
    fig.tight_layout(w_pad=2.0,h_pad=2.4);save_figure(fig,output,2,main)
    # Figure 3 presents the v3 improvement and its prespecified sensitivity failure.
    fig,axes=plt.subplots(2,2,figsize=(9.3,7.1));healthy=CONDITIONS[:2]+['correlated','z_dependent_censoring'];used=[]
    for method,color,label in [('v2_linear',BLUE,'v2 linear'),('v3_linear',ORANGE,'v3 linear')]:
        chosen=v3[(v3.method==method)&v3.condition.isin(healthy)].sort_values(['p','condition']);labels=[f'{LABELS[CONDITIONS.index(r.condition)]} / P{r.p}' for r in chosen.itertuples()]
        values=chosen.allowed_fraction.to_numpy()*100;ci=np.array([interval(int(r.allowed),int(r.planned)) for r in chosen.itertuples()])*100
        axes[0,0].errorbar(values,np.arange(8)+(-.12 if method=='v2_linear' else .12),xerr=np.maximum(0,np.vstack([values-ci[:,0],ci[:,1]-values])),fmt='o',ms=3,color=color,label=label);used.append(chosen)
        bad=v3[(v3.method==method)&(v3.condition=='heteroskedastic')].sort_values('p');values=bad.withhold_fraction.to_numpy()*100;ci=np.array([interval(int(r.planned-r.allowed),int(r.planned)) for r in bad.itertuples()])*100
        axes[0,1].errorbar([0,1],values,yerr=np.maximum(0,np.vstack([values-ci[:,0],ci[:,1]-values])),fmt='o-',color=color,label=label,ms=4,capsize=2);used.append(bad)
        powerrows=v3[(v3.method==method)&v3.condition.str.startswith('partial_')].sort_values(['p','condition']);ratio=[pair(r.power_ratio) for r in powerrows.itertuples()]
        vals=np.array([r['point'] for r in ratio])*100;ci=np.array([r['mc95'] for r in ratio])*100
        axes[1,0].errorbar(np.arange(4)+(-.07 if method=='v2_linear' else .07),vals,yerr=np.maximum(0,np.vstack([vals-ci[:,0],ci[:,1]-vals])),fmt='o',color=color,ms=4,capsize=2,label=label);used.append(powerrows)
    axes[0,0].set(yticks=range(8),yticklabels=labels,xlabel='Allowed analyses (%)',xlim=(74,100));axes[0,0].invert_yaxis();axes[0,0].legend(fontsize=8,loc='upper left')
    axes[0,1].set(xticks=[0,1],xticklabels=['30 markers','300 markers'],ylabel='Withheld for misspecification (%)',ylim=(40,103));axes[0,1].legend(fontsize=8)
    axes[1,0].set(xticks=range(4),xticklabels=['Strong\nP30','Weak\nP30','Strong\nP300','Weak\nP300'],ylabel='Power retained vs legacy (%)',ylim=(70,102));axes[1,0].axhline(90,color=GRAY,ls='--',lw=.7)
    completeness=pd.read_csv(inputs/'v3_completeness.csv');bottom=np.zeros(4)
    for col,color,label in [('validated_paired_rows',GREEN,'Readable paired result'),('unusable',ORANGE,'Unreadable'),('absent','#DDDDDD','No stored result')]:
        vals=completeness[col].to_numpy()/completeness.planned.to_numpy()*100;axes[1,1].barh(range(4),vals,left=bottom,color=color,label=label);bottom+=vals
    axes[1,1].set(yticks=range(4),yticklabels=['Main: 120,000','Extension: 12,000','Large: 1,500','Stress: 16,000'],xlabel='Planned indices (%)',xlim=(0,100));axes[1,1].invert_yaxis();axes[1,1].legend(fontsize=7,loc='lower right')
    for ax,label,title in zip(axes.flat,'ABCD',['Healthy-condition availability','Missed heteroskedasticity increases','Power improves','Confirmation remains incomplete']):panel(ax,label,title)
    fig.tight_layout(w_pad=2.2,h_pad=2.5);save_figure(fig,output,3,pd.concat(used).drop_duplicates(['condition','p','method']))
    # Figure 4 encodes execution evidence, not a feature superiority score.
    tools=['SurvStudio','R survival/Hmisc/rms','lifelines','mlsurv','surviveR','KM Plotter'];names=['Event coding','Kaplan-Meier','Adjusted HR and reference','Misspecification response','Duplicates and outcome role','Selection and internal assessment','Fixed external validation','Endpoint pairing']
    codes={'N':('#DCF0EA','Executed native calculation'),'S':('#DEEDF8','Executed added script'),'D':('#FFF0DC','Different target or supplied-input response'),'P':('#EAEAEA','Execution unverified'),'L':('#D8D8D8','Access restricted')}
    fig,ax=plt.subplots(figsize=(9.3,4.9));ax.set(xlim=(0,6),ylim=(8,0));ax.set_xticks(np.arange(6)+.5,['SurvStudio','R workflow','lifelines','mlsurv','surviveR','KM Plotter']);ax.xaxis.tick_top();ax.set_yticks(np.arange(8)+.5,names);ax.tick_params(length=0);ax.spines[:].set_visible(False)
    for j,tool in enumerate(tools):
        part=tasks[tasks.tool==tool]
        for i,row in enumerate(part.itertuples()):
            code=row.display_code;ax.add_patch(plt.Rectangle((j,i),1,1,facecolor=codes[code][0],edgecolor='white',lw=2));ax.text(j+.5,i+.5,code,ha='center',va='center',fontsize=11)
    fig.subplots_adjust(left=.30,right=.99,top=.85,bottom=.25)
    for i,(code,(_,desc)) in enumerate(codes.items()):fig.text(.06,.19-i*.027,code+'  '+desc,fontsize=8)
    save_figure(fig,output,4,tasks)
    # Figure 5 separates numerical agreement from exact artifact reproduction.
    checks=pd.read_csv(inputs/'independent_numeric_checks.csv');bench=pd.read_csv(inputs/'benchmark.csv');groups=[]
    for name in ('eight_task','native_lifelines','public_case','v3_aggregation'):
        part=checks[checks.validation_group==name]
        columns=[c for c in ('maximum_difference','absolute_difference','difference') if c in part]
        vals=[float(v) for c in columns for v in part[c].dropna()]
        groups.append(dict(group=name,checks=len(part),maximum_difference=max(vals,default=0)))
    groups.append(dict(group='common_KM_Cox',checks=len(bench),maximum_difference=float(bench.numeric_maximum_difference.max())))
    readable={'eight_task':'Eight-task R reference','native_lifelines':'lifelines reference','public_case':'Public case R reference','v3_aggregation':'v3 aggregate R check','common_KM_Cox':'Common KM and Cox'}
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.1),gridspec_kw={'width_ratios':[1.15,1]});labels=[f"{readable[r['group']]} ({r['checks']:,} checks)" for r in groups]
    axes[0].barh(range(5),[max(r['maximum_difference'],1e-16) for r in groups],color=BLUE);axes[0].set_xscale('log');axes[0].axvline(1e-6,color=ORANGE,ls='--');axes[0].set(yticks=range(5),yticklabels=labels,xlabel='Maximum absolute difference',xlim=(1e-16,1e-5));axes[0].invert_yaxis()
    fresh=json.loads((inputs/'fresh_environment-verification.json').read_text());files=list(fresh['file_hash_checks']);axes[1].set(xlim=(0,1),ylim=(len(files),0));axes[1].axis('off')
    for i,name in enumerate(files):axes[1].text(.02,i+.5,name,fontsize=8.5);axes[1].text(.75,i+.5,'Exact hash match' if fresh['file_hash_checks'][name] else 'Different',color=GREEN,fontsize=8.5)
    panel(axes[0],'A','Numerical agreement');panel(axes[1],'B','Nine files match on a fresh server');fig.tight_layout(w_pad=3);save_figure(fig,output,5,pd.DataFrame(groups))
    # Figure 6 retains endpoints and the prespecified public-case sensitivities.
    old=pd.read_csv(inputs/'existing_external_cases.csv');pub=pd.read_csv(inputs/'public_external_case.csv');primary=pub[(pub.horizon_years==5)&pub['case'].str.startswith('conservative_RFS-restricted')].iloc[0]
    fig,axes=plt.subplots(2,2,figsize=(9.3,7.3));source=[]
    labels=['Case I / OS','Case IV / OS','Case V / RFS','Case V / DMFS','Rotterdam to GBSG / RFS']
    for ax,metric in [(axes[0,0],'delta_c'),(axes[0,1],'calibration_slope')]:
        for i,(case,endpoint) in enumerate([('I','os'),('IV','os'),('V','rfs'),('V','dmfs')]):
            r=old[(old['case']==case)&(old.endpoint==endpoint)&(old.clinical_basis=='linear')&(old.scaling=='as_measured')&(old.metric==metric)].iloc[0]
            val=r.estimate;lo=r.hksj_ci_lower;hi=r.hksj_ci_upper
            ax.errorbar(val,i,xerr=[[val-lo],[hi-val]],fmt='o',color=BLUE,ms=4,capsize=2);source.append(dict(label=labels[i],metric=metric,estimate=val,ci_lower=lo,ci_upper=hi,n=int(r.n),events=int(r.events),interval='modified HKSJ',inference='withheld'))
        key='delta_c_index' if metric=='delta_c' else metric;val=primary[key];lo,hi=pair(primary[key+'_ci'])
        ax.errorbar(val,4,xerr=[[val-lo],[hi-val]],fmt='s',color=ORANGE,ms=4,capsize=2);source.append(dict(label=labels[4],metric=metric,estimate=val,ci_lower=lo,ci_upper=hi,n=686,interval='paired external bootstrap',inference='withheld'))
        ax.set(yticks=range(5),yticklabels=labels if metric=='delta_c' else [],xlabel='Gain in Harrell C' if metric=='delta_c' else 'Calibration slope');ax.invert_yaxis();ax.axvline(0 if metric=='delta_c' else 1,color=GRAY,ls='--',lw=.8)
    public=pub[pub.horizon_years==5];labels2=[]
    for i,r in enumerate(public.itertuples()):
        label=('Conservative' if r.case.startswith('conservative') else 'Liberal')+' / '+('RCS' if 'restricted' in r.case else 'linear');labels2.append(label)
        for ax,key in [(axes[1,0],'delta_c_index'),(axes[1,1],'calibration_slope')]:
            val=getattr(r,key);lo,hi=pair(getattr(r,key+'_ci'));ax.errorbar(val,i,xerr=[[val-lo],[hi-val]],fmt='o',color=ORANGE,ms=4,capsize=2)
            source.append(dict(label=label,metric=key,estimate=val,ci_lower=lo,ci_upper=hi,n=686,interval='paired external bootstrap',inference=r.inference_status))
    for ax in axes[1]:ax.set(yticks=range(4),yticklabels=labels2 if ax is axes[1,0] else []);ax.invert_yaxis()
    axes[1,0].set_xlabel('Gain in Harrell C');axes[1,0].axvline(0,color=GRAY,ls='--',lw=.8);axes[1,1].set_xlabel('Calibration slope');axes[1,1].axvline(1,color=GRAY,ls='--',lw=.8)
    for ax,label,title in zip(axes.flat,'ABCD',['External gain remains case-specific','Calibration and uncertainty','Prespecified endpoint\nand basis sensitivities','All four benchmark analyses\nremain withheld']):panel(ax,label,title)
    fig.tight_layout(w_pad=2.1,h_pad=2.5);save_figure(fig,output,6,pd.DataFrame(source))
    # Keep every displayed ancillary exact count and source accessible.
    shutil.copyfile(inputs/'v3_completeness.csv',output/'Figure3-completeness-source.csv')
    shutil.copyfile(inputs/'power-retention-uncertainty.csv',output/'Figure2-power-source.csv')
    shutil.copyfile(inputs/'fresh_environment-verification.json',output/'Figure5-fresh-source.json')
    print('Generated six figures with source tables')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.inputs,a.output)
