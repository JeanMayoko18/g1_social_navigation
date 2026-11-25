"""Regression checks for annotations, shared exports and replot dispatch."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from g1_analysis.discovery import annotations
from g1_analysis.aggregate import summarize
from g1_analysis.count_validation import validate_counts
from g1_analysis.replot import run_replot
from g1_analysis.reporting import csv_file
from g1_analysis.paper_plots import success_plot, command_windows_overview

ROOT = Path(__file__).resolve().parents[1]

class PublicationTests(unittest.TestCase):
    def rows(self):
        config = yaml.safe_load((ROOT/'config/campaign.yaml').read_text())
        rows=[]
        # Synthetic scalar values are used only to test software behavior.
        for tid in annotations(ROOT/'config/navigation_annotations.csv'):
            campaign, configuration, number=tid.split('__')
            rows.append(dict(trial_id=tid, campaign=campaign, configuration=configuration,
                             trial_number=number.removeprefix('trial_'), outcome='UNKNOWN',
                             command_window_s=100.0, **config['campaigns'][campaign]))
        return rows,config

    def test_complete_annotations(self):
        manual=annotations(ROOT/'config/navigation_annotations.csv')
        self.assertEqual(len(manual),64)
        self.assertFalse(any(r['outcome'] in ('','UNKNOWN') for r in manual.values()))
        rows,_=self.rows()
        for row in rows: row['outcome']=manual[row['trial_id']]['outcome']
        validate_counts(rows,ROOT/'config/success_counts.csv')
        totals={c:sum(r['outcome']=='SUCCEEDED' for r in rows if r['configuration']==c)
                for c in ('calibrated','Odometry-only','dog','g1_nav')}
        self.assertEqual(totals,{'calibrated':15,'Odometry-only':11,'dog':8,'g1_nav':7})
        self.assertEqual(sum(r['outcome']=='CANCELED' for r in rows),1)
        broken=[dict(r) for r in rows];broken[0]['outcome']='SUCCEEDED' if broken[0]['outcome']!='SUCCEEDED' else 'ABORTED'
        with self.assertRaises(ValueError): validate_counts(broken,ROOT/'config/success_counts.csv')

    def test_replot_resolves_before_all_exports(self):
        rows,config=self.rows()
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'metrics.csv';csv_file(source,rows)
            out=Path(folder)/'out'
            with patch('g1_analysis.replot.campaign_figures') as render, patch('g1_analysis.aggregate_outcomes.export_declared_counts'):
                run_replot(source,out,config,ROOT/'config/navigation_annotations.csv',ROOT/'config/success_counts.csv')
            resolved=render.call_args.args[0]
            self.assertEqual(len(resolved),64)
            self.assertEqual(sum(r['outcome']=='UNKNOWN' for r in resolved),0)
            self.assertTrue(all(r['command_window_s']==100 for r in resolved))
            self.assertEqual(sum(r['n_success'] for r in summarize(resolved)),41)
            with patch('g1_analysis.paper_plots.save') as save:
                command_windows_overview(resolved,out)
                fig=save.call_args.args[0]
                labels=[t.get_text() for legend in fig.legends for t in legend.get_texts()]
                self.assertNotIn('Unknown',labels)
                import matplotlib.pyplot as plt
                plt.close(fig)
                success_plot(resolved,out)
                fig=save.call_args.args[0]
                labels=[t.get_text() for ax in fig.axes for t in ax.texts]
                self.assertTrue(all('*' not in t for t in labels))
                plt.close(fig)

    def test_cli_aliases_skip_discovery(self):
        from g1_analysis.cli import main
        for option in ('--result','--results'):
            with patch('sys.argv',['main.py',option,'input.csv','--output','out']), patch('g1_analysis.cli.discover',side_effect=AssertionError('Bag discovery called')), patch('g1_analysis.replot.run_replot',return_value=0) as run:
                self.assertEqual(main(),0)
                self.assertEqual(run.call_args.args[0],Path('input.csv'))

if __name__=='__main__':
    unittest.main()
