"""Regenerate evidence-bounded publication figures from aggregate/synthetic outputs."""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

LABELS={'independent':'Independent markers','linear':'Linear marker relation','nonlinear_marker':'Nonlinear marker relation',
        'heteroskedastic':'Heteroskedastic markers','correlated':'Correlated markers',
        'z_dependent_censoring':'Z-dependent censoring','nonph_linear':'Time reversal\n+ linear markers',
        'nonph_nonlinear':'Time reversal\n+ nonlinear markers','nonph_heteroskedastic':'Time reversal\n+ heteroskedastic markers',
        'nonlinear_clinical_risk':'Nonlinear clinical risk','partial_weak':'Weak partial null','partial_strong':'Strong partial null'}
COLORS={'legacy_linear':'#424242','guarded_linear':'#0072B2','guarded_spline':'#D55E00'}
METHODS={'legacy_linear':'Legacy linear calculation','guarded_linear':'Guarded linear','guarded_spline':'Guarded spline'}


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=300,facecolor='white')
    fig.savefig(out/(name+'.svg'),facecolor='white',metadata={'Date':None})
    svg=out/(name+'.svg')
    # Matplotlib adds an external SVG DTD declaration; standalone exports need no DTD.
    standalone=re.sub(r'<!DOCTYPE[^>]*>\s*','',svg.read_text())
    svg.write_text('\n'.join(line.rstrip() for line in standalone.splitlines())+'\n')
    fig.savefig(out/(name+'.pdf'),facecolor='white',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(length=3)
    ax.grid(axis='x',color='#e5e5e5',lw=.55,zorder=0)


def run(root):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9.5,'axes.labelsize':9,
        'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'SurvStudio-inference-v2-1ed6c10',
        'axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8})
    summary=json.loads((root/'confirmation-v2/confirmation-summary.json').read_text());data=summary['summaries']
    # Figure 1 is a process schematic, with no quantitative or causal claims.
    fig,ax=plt.subplots(figsize=(7.4,5.5));ax.set(xlim=(0,10),ylim=(0,9.2));ax.axis('off')
    nodes=[('input',5,8.3,7.7,'Declare event, endpoint, marker and clinical roles'),
           ('basis',5,7.1,7.7,'Prespecify clinical basis: linear default or five-knot spline'),
           ('train',5,5.9,7.7,'Fit transformations within training rows; retain a fixed recipe'),
           ('diagnostics',5,4.5,7.7,'Clinical PH and form; marker residual mean and variance\nSeparate Holm families, fixed 1% withholding rule'),
           ('qualification',5,3.2,7.7,'Apply the method engineering gate\nFailed profiles remain exploratory'),
           ('raw',2.5,1.9,4.2,'Retain raw calculations\nExploratory p/q and fitted prediction'),
           ('state',7.4,1.9,4.2,'Carry inference state and reasons\nAPI, screen, figures, CSV, reports'),
           ('external',5,.5,7.7,'Apply fixed external transformations and coefficients\nPreserve state; report endpoints separately')]
    for key,x,y,w,text in nodes:
        ax.add_patch(FancyBboxPatch((x-w/2,y-.4),w,.8,boxstyle='round,pad=.06,rounding_size=.06',fc='#f5f7f8',ec='#667680',lw=.8))
        ax.text(x,y,text,ha='center',va='center',fontsize=9)
    for x1,y1,x2,y2 in [(5,7.86,5,7.54),(5,6.66,5,6.34),(5,5.46,5,4.94),(5,4.06,5,3.64),
                          (4,2.76,2.5,2.34),(6,2.76,7.4,2.34),(2.5,1.46,4,.94),(7.4,1.46,6,.94)]:
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=.85,color='#56616b'))
    ax.text(.05,9.12,'SurvStudio conditional marker workflow',fontsize=11,fontweight='bold',va='top')
    fig.subplots_adjust(.015,.02,.99,.96);save(fig,out,'Figure_1_workflow')
    pd.DataFrame(nodes,columns=['node','x','y','width','text']).to_csv(out/'Figure_1_source.csv',index=False)

    # Main global null, all ten conditions; zero allowed gives NA conditional FWER.
    conditions=[c for c in summary['protocol']['conditions'] if not c.startswith('partial_')]
    main=[r for r in data if r['stage']=='main'];lookup={(r['condition'],r['method']):r for r in main}
    fig,axes=plt.subplots(2,2,figsize=(8.1,7.1),sharey=True)
    panels=[('fwer_raw_lower','raw_fwer_mc95','A  Raw FWER',('legacy_linear','guarded_spline')),
            ('fwer','fwer_mc95','B  FWER after withholding',('guarded_linear','guarded_spline')),
            ('conditional_fwer','conditional_mc95','C  FWER in allowed analyses',('guarded_linear','guarded_spline')),
            ('allowed_fraction',None,'D  Analysis allowed',('guarded_linear','guarded_spline'))]
    for ax,(metric,interval,title,methods) in zip(axes.flat,panels):
        for mi,method in enumerate(methods):
            for j,c in enumerate(conditions):
                row=lookup[c,method];y=j+(-.14 if mi==0 else .14);value=row[metric]
                if value is None:
                    ax.text(99 if metric=='conditional_fwer' else 5.9,y,'NA (0)',ha='right',va='center',color=COLORS[method],fontsize=6.5)
                    continue
                value*=100
                if interval:
                    lo,hi=row[interval];error=np.array([[value-100*lo],[100*hi-value]])
                    ax.errorbar(value,y,xerr=error,fmt='o' if mi==0 else 's',ms=3.3,color=COLORS[method],lw=.7,capsize=1.7,zorder=3)
                else: ax.plot(value,y,'o' if mi==0 else 's',ms=3.6,color=COLORS[method],zorder=3)
        ax.set_title(title,loc='left',fontweight='bold');ax.set_yticks(range(len(conditions)),[LABELS[c] for c in conditions])
        ax.set_ylim(len(conditions)-.55,-.55);style(ax)
        if metric in ('fwer_raw_lower','conditional_fwer'):
            ax.set_xscale('symlog',linthresh=1);ax.set_xlim(-.05,105);ax.set_xticks([0,1,5,20,100],['0','1','5','20','100'])
        elif metric=='fwer': ax.set_xlim(-.15,6.2);ax.set_xticks([0,2,4,6])
        else: ax.set_xlim(-2,104);ax.set_xticks([0,25,50,75,100])
        ax.axvline(80 if metric=='allowed_fraction' else 5,color='#737373',ls='--',lw=.65,zorder=1)
        ax.set_xlabel('Datasets allowed (%)' if metric=='allowed_fraction' else 'FWER (%)')
    handles=[plt.Line2D([],[],marker=marker,color=COLORS[m],ls='',label=METHODS[m],ms=4) for m,marker in [('legacy_linear','o'),('guarded_linear','o'),('guarded_spline','s')]]
    fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.01),frameon=False)
    fig.text(.5,.07,'5,000 datasets per condition; n = 180, 30 markers, 999 permutations. Whiskers: 95% Monte Carlo intervals.\nv2 linear passes declared main support only; spline remains exploratory. Extensions do not establish broader qualification.',ha='center',fontsize=7.5)
    fig.subplots_adjust(left=.30,right=.985,top=.955,bottom=.15,wspace=.14,hspace=.4)
    save(fig,out,'Figure_2_error_and_availability')
    pd.DataFrame([r for r in main if r['condition'] in conditions]).drop(columns=['source_hashes']).to_csv(out/'Figure_2_source.csv',index=False)

    fig=plt.figure(figsize=(7.4,5.8));gs=fig.add_gridspec(2,2,height_ratios=[1,1.1]);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    for i,method in enumerate(summary['protocol']['methods']):
        rows=[lookup[cond,method] for cond in ('partial_weak','partial_strong')]
        y=np.array([r['power'] for r in rows])*100;ci=np.array([r['power_mc95'] for r in rows])*100
        x=np.arange(2)+(i-1)*.19
        a.errorbar(x,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='o',color=COLORS[method],ms=4,capsize=2,lw=1)
    for i,method in enumerate(('guarded_linear','guarded_spline')):
        retention=[100*lookup[cond,method]['power']/lookup[cond,'legacy_linear']['power'] for cond in ('partial_weak','partial_strong')]
        uncertainty=pd.read_csv(root/'power-retention-uncertainty.csv').set_index(['condition','method'])
        bounds=np.array([[100*uncertainty.loc[(cond,method),'paired_delta_MC95_lower'],100*uncertainty.loc[(cond,method),'paired_delta_MC95_upper']] for cond in ('partial_weak','partial_strong')])
        error=np.stack([np.array(retention)-bounds[:,0],bounds[:,1]-np.array(retention)])
        b.bar(np.arange(2)+(i-.5)*.3,retention,width=.28,color=COLORS[method],label=METHODS[method],yerr=error,capsize=2,error_kw={'elinewidth':.8})
    a.set_title('A  Power over all planned datasets',loc='left',fontweight='bold');a.set_ylabel('True markers rejected (%)');a.set_ylim(0,55)
    b.set_title('B  Power retained versus original',loc='left',fontweight='bold');b.set_ylabel('Power ratio (%)');b.axhline(90,color='#737373',ls='--',lw=.8);b.set_ylim(0,105)
    for ax in (a,b): ax.set_xticks([0,1],['Weak signal','Strong signal']);style(ax)
    shapes={'independent':'o','linear':'s','correlated':'^'}
    for cond in shapes:
        for method in ('guarded_linear','guarded_spline'):
            selected=[r for r in data if r['n']==180 and r['condition']==cond and r['method']==method]
            selected=sorted(selected,key=lambda r:r['p'])
            c.plot([r['p'] for r in selected],[r['allowed_fraction']*100 for r in selected],marker=shapes[cond],
                   ls='-' if method=='guarded_linear' else '--',color=COLORS[method],ms=4,lw=1)
    c.set_xscale('log');c.set_xticks([30,300,3000],['30','300','3,000']);c.set_ylim(0,105);c.set_ylabel('Datasets allowed (%)');c.set_xlabel('Number of markers (n = 180)')
    c.axhline(80,color='#737373',ls=':',lw=.8);c.set_title('C  Prespecified marker-dimension extensions',loc='left',fontweight='bold');style(c)
    condition_handles=[plt.Line2D([],[],marker=s,color='#555555',ls='',label=LABELS[cond],ms=4) for cond,s in shapes.items()]
    c.legend(handles=condition_handles,ncol=3,loc='lower left',frameon=False,fontsize=7.5)
    power_handles=[plt.Line2D([],[],color=COLORS[m],ls=ls,label=METHODS[m],marker='o' if m=='legacy_linear' else None,ms=4)
                   for m,ls in [('legacy_linear',''),('guarded_linear','-'),('guarded_spline','--')]]
    fig.legend(handles=power_handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.005),frameon=False)
    fig.text(.5,.075,'Power whiskers: 95% Monte Carlo intervals; retention whiskers: paired delta intervals (descriptive).\nThe retention gate uses point estimates. Linear passes declared main support only; spline remains exploratory.',ha='center',fontsize=7.5)
    fig.subplots_adjust(left=.1,right=.98,top=.95,bottom=.20,wspace=.36,hspace=.48);save(fig,out,'Figure_3_power_and_dimension')
    pd.DataFrame([r for r in data if r['condition'].startswith('partial_') or (r['n']==180 and r['condition'] in shapes)]).drop(columns=['source_hashes']).to_csv(out/'Figure_3_source.csv',index=False)
    uncertainty.reset_index().to_csv(out/'Figure_3_ratio_source.csv',index=False)

    reference=json.loads((root/'independent_R_full_v2/verification.json').read_text())
    comparison=pd.read_csv(root/'tool-comparison-v2/numerical-comparison.csv')
    fig,(a,b)=plt.subplots(1,2,figsize=(7.9,4.5),gridspec_kw={'width_ratios':[1.2,1]})
    core=list(reference['differences'].items());source=[]
    for i,(name,diff) in enumerate(core):
        a.plot(max(diff,1e-16),i,'o',color='#0072B2',ms=4)
        source.append({'panel':'A','quantity':name,'absolute_difference':diff,'tolerance':reference['tolerances'][name]})
    a.set_yticks(range(len(core)),[k.replace('_',' ') for k,v in core]);a.invert_yaxis();a.set_title('A  Spline and diagnostic checks',loc='left',fontweight='bold')
    groups=[('Residual HC3 statistics',comparison.quantity.str.startswith('diagnostic_')),
            ('KM survival estimates',comparison.quantity.str.contains(' survival',regex=False)),
            ('Adjusted Cox fit',comparison.quantity.str.startswith('adjusted_')),
            ('Held-out concordance',comparison.quantity=='inner_heldout_C'),
            ('Fixed external C and calibration',comparison.quantity.str.startswith('locked_external_'))]
    for i,(label,mask) in enumerate(groups):
        diff=comparison.loc[mask,'absolute_difference'].max();b.plot(max(diff,1e-16),i,'s',color='#D55E00',ms=4)
        source.append({'panel':'B','quantity':label,'n_quantities':int(mask.sum()),'absolute_difference':diff,'tolerance':1e-6})
    b.set_yticks(range(len(groups)),[f'{label}\n(n = {int(mask.sum())})' for label,mask in groups]);b.invert_yaxis();b.set_title('B  Eight workflow tasks',loc='left',fontweight='bold')
    for ax in (a,b):
        ax.set_xscale('log');ax.set_xlim(3e-17,2e-6);ax.set_xticks([1e-16,1e-12,1e-8,1e-6]);ax.axvline(1e-6,color='#555',ls='--',lw=.8);style(ax);ax.set_xlabel('Maximum absolute difference')
    fig.text(.5,.045,'14 corrected-source reference quantities and 84 current workflow quantities passed. Six coding/selection checks agreed exactly.\nValues below 10⁻¹⁶ are drawn at 10⁻¹⁶. KM Plotter execution remains unverified.',ha='center',fontsize=7.5)
    fig.subplots_adjust(left=.21,right=.985,top=.92,bottom=.22,wspace=1.02);save(fig,out,'Figure_4_independent_numerical_validation')
    pd.DataFrame(source).to_csv(out/'Figure_4_source.csv',index=False)

    # Full grid prevents selecting only favorable expansion conditions for presentation.
    columns=[('main',180,30),('extension',500,30),('extension',180,300),('large',180,3000)]
    conds=summary['protocol']['conditions'];fig,axes=plt.subplots(1,2,figsize=(7.3,5.4),sharey=True)
    for ax,method in zip(axes,('guarded_linear','guarded_spline')):
        matrix=np.full((len(conds),len(columns)),np.nan)
        for j,cell in enumerate(columns):
            for i,cond in enumerate(conds):
                rows=[r for r in data if (r['stage'],r['n'],r['p'])==cell and r['condition']==cond and r['method']==method]
                if rows:matrix[i,j]=rows[0]['allowed_fraction']*100
        image=ax.imshow(matrix,vmin=0,vmax=100,cmap='cividis',aspect='auto')
        for (i,j),value in np.ndenumerate(matrix):
            label='NA' if np.isnan(value) else f'{value:.2f}' if 0<value<.1 else f'{value:.1f}'
            ax.text(j,i,label,ha='center',va='center',fontsize=7.5,color='black' if value>=60 or np.isnan(value) else 'white')
        ax.set_xticks(range(4),['180 / 30','500 / 30','180 / 300','180 / 3,000'],rotation=35,ha='right');ax.set_xlabel('Patients / markers')
        ax.set_yticks(range(len(conds)),[LABELS[c] for c in conds]);ax.set_title(METHODS[method],fontweight='bold')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Datasets allowed (%)',fraction=.025,pad=.025)
    fig.text(.5,.035,'NA: cell not prespecified. Main: 5,000; extensions: 1,000; large-marker cells: 500 datasets per condition.\nv2 linear passes declared main support only; spline remains exploratory. Extensions do not establish broader qualification.',ha='center',fontsize=7.5)
    fig.subplots_adjust(left=.31,right=.865,top=.93,bottom=.21,wspace=.1);save(fig,out,'Figure_S1_full_availability')
    pd.DataFrame(data).drop(columns=['source_hashes']).to_csv(out/'Figure_S1_source.csv',index=False)
    cases=root/'cases-v2/pooled-case-results.csv'
    if cases.exists():
        table=pd.read_csv(cases)
        primary=table[table.scaling.eq('as_measured') & table.metric.isin(['delta_c','calibration_slope'])].copy()
        order=[('I','os'),('IV','os'),('V','rfs'),('V','dmfs')]
        fig,axes=plt.subplots(1,2,figsize=(7.6,4.25),sharey=True)
        for ax,metric,title,reference in zip(axes,['delta_c','calibration_slope'],
                                             ['A  External gain in concordance','B  External calibration slope'],[0,1]):
            for i,(case,endpoint) in enumerate(order):
                for offset,basis,color,shape in [(-.13,'linear','#0072B2','o'),(.13,'restricted_cubic_spline','#D55E00','s')]:
                    row=primary[primary.case.eq(case)&primary.endpoint.eq(endpoint)&primary.clinical_basis.eq(basis)&primary.metric.eq(metric)].iloc[0]
                    ax.errorbar(row.estimate,i+offset,xerr=np.array([[row.estimate-row.hksj_ci_lower],[row.hksj_ci_upper-row.estimate]]),
                                fmt=shape,color=color,ms=4,capsize=2,lw=.8)
            ax.set_yticks(range(4),['Case I: OS (k = 7)','Case IV: OS (k = 3)','Case V: RFS (k = 3)','Case V: DMFS (k = 2)'])
            ax.set_ylim(3.45,-.45);ax.set_title(title,loc='left',fontweight='bold');style(ax)
            ax.axvline(reference,color='#555',ls='--',lw=.8);ax.set_xlabel('ΔC' if metric=='delta_c' else 'Slope')
        handles=[plt.Line2D([],[],marker=shape,color=color,ls='',label=label) for shape,color,label in
                 [('o','#0072B2','Linear primary'),('s','#D55E00','Spline sensitivity')]]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.02),ncol=2,frameon=False)
        fig.text(.5,.115,'Existing external data reanalysis; frozen development clinical transforms and as-measured markers.\nWhiskers: 95% HKSJ intervals. All predictions and comparisons remain exploratory.',ha='center',fontsize=7.5)
        fig.subplots_adjust(left=.23,right=.985,top=.93,bottom=.25,wspace=.27)
        save(fig,out,'Figure_5_external_reanalysis');primary.to_csv(out/'Figure_5_source.csv',index=False)
    print(json.dumps({'figures':6 if cases.exists() else 5,'directory':str(out)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);run(p.parse_args().results)
