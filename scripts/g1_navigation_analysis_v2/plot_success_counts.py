#!/usr/bin/env python3
"""Plot explicit aggregate counts without fabricating individual trial outcomes."""
import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wilson(k,n):
    z=1.959963984540054
    p=k/n; d=1+z*z/n
    m=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,m-h),min(1,m+h)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts',type=Path,default=Path(__file__).with_name('success_counts.csv'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('figures'))
    args=parser.parse_args()
    with args.counts.open(newline='') as f:
        rows=list(csv.DictReader(f))
    lookup={}
    for r in rows:
        key=(r['scenario'],float(r['speed_mps']),r['configuration'])
        if key in lookup:
            raise ValueError('Duplicate group: '+str(key))
        k,n=int(r['successes']),int(r['trials'])
        if not 0<=k<=n or n==0:
            raise ValueError('Invalid counts: '+str(key))
        lookup[key]=(k,n)
    order=['calibrated','Odometry-only','dog','g1_nav']
    labels=['Odom +\ncommand','Odometry\nonly','Dog\nodometry','g1_nav']
    colors=['#0072B2','#AA4499','#E69F00','#009E73']
    panels=[('corridor_with_doors',.12,'Fire doors'),('corridor_with_doors',.30,'Fire doors'),
            ('corridor',.12,'Office corridor'),('corridor',.30,'Office corridor')]
    fig,axes=plt.subplots(2,2,figsize=(10.8,6.8),sharey=True)
    fig.suptitle('Navigation success rate\nAll reported trials; Wilson 95% intervals',fontsize=13,y=.99)
    for ax,(scenario,speed,label) in zip(axes.flat,panels):
        for i,config in enumerate(order):
            k,n=lookup[(scenario,speed,config)]
            percent=100*k/n;low,high=wilson(k,n)
            ax.bar(i,percent,width=.60,color=colors[i])
            ax.errorbar(i,percent,yerr=[[max(0,percent-low*100)],[max(0,high*100-percent)]],fmt='none',color='#333333',capsize=4,linewidth=1.2)
            ax.text(i,108,f'{k}/{n}',ha='center',fontsize=11,fontweight='bold')
        ax.set_title(f'{label} | {speed:.2f} m/s',fontsize=11)
        ax.set_xticks(range(4),labels,fontsize=10)
        ax.set_ylim(0,120);ax.set_yticks(range(0,101,20))
        ax.set_ylabel('Success rate (%)',fontsize=10)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(rect=(0,0,1,.93))
    args.output.mkdir(parents=True,exist_ok=True)
    with matplotlib.rc_context({'pdf.fonttype':42,'svg.fonttype':'none'}):
        for extension in ('png','pdf','svg'):
            fig.savefig(args.output/('success_rate.'+extension),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f'Figures saved to {args.output}')

if __name__=='__main__':
    main()
