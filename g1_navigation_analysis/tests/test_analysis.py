"""Regression tests for interpretation-critical behavior, independent of ROS."""
import unittest
import tempfile
from pathlib import Path
import yaml
from g1_analysis.discovery import discover
from g1_analysis.outcomes import resolve
from g1_analysis.numerics import consistency, distance_to_polyline, wilson
from g1_analysis.metrics import analyze
from g1_analysis.aggregate import summarize

CONFIG = yaml.safe_load((Path(__file__).parents[1]/'config/campaign.yaml').read_text())

class AnalysisTests(unittest.TestCase):
    def test_discovery_all_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            for campaign in CONFIG['campaigns']:
                bag = Path(directory)/campaign/'Odometry-only'/f'{campaign}_Odometry-only_trial_02'/'navigation'
                bag.mkdir(parents=True); (bag/'metadata.yaml').write_text('{}')
            trials = discover(Path(directory),CONFIG)
            self.assertEqual(len(trials),4)
            self.assertEqual({r['speed_mps'] for r in trials},{.12,.30})
    def test_old_terminal_not_current_success(self):
        data = dict(bag_start=100,bag_end=200,statuses=[dict(t=110,goal_stamp=10,goal_id='abc',status=4)])
        self.assertEqual(resolve(data,{})['outcome'],'UNKNOWN')
    def test_first_terminal_time(self):
        data = dict(bag_start=100,bag_end=200,statuses=[dict(t=t,goal_stamp=105,goal_id='abc',status=s) for t,s in [(106,2),(150,4),(190,4)]])
        self.assertEqual(resolve(data,{})['end_s'],150)
    def test_multiple_goals_ambiguous(self):
        data = dict(bag_start=100,bag_end=200,statuses=[dict(t=110,goal_stamp=105,goal_id=g,status=4) for g in ['abc','def']])
        self.assertEqual(resolve(data,{})['outcome'],'UNKNOWN')
    def test_no_motion_retained(self):
        data = dict(bag_start=100,bag_end=200,statuses=[],streams={},counts={},types={},warnings=[])
        result,_ = analyze(dict(trial_id='test'),data,CONFIG['analysis'],{})
        self.assertEqual(result['outcome'],'UNKNOWN')
        self.assertEqual(result['active_command_samples'],0)
        self.assertIsNone(result['terminal_position_residual_m'])
    def test_se2_alignment(self):
        a = [dict(t=i,x=i,y=0,yaw=0) for i in range(3)]
        b = [dict(t=i,x=10,y=20+i,yaw=1.5707963267948966) for i in range(3)]
        self.assertAlmostEqual(consistency(a,b,.1)['position_rmse_m'],0)
    def test_polyline_segments(self):
        self.assertAlmostEqual(distance_to_polyline(5,2,[(0,0),(10,0)]),2)
    def test_unknown_not_failure(self):
        rows = [dict(scenario='a',speed_mps=.12,configuration='dog',outcome=o) for o in ['SUCCEEDED','UNKNOWN']]
        summary = summarize(rows)[0]
        self.assertEqual(summary['success_rate_known'],1)
        self.assertEqual(summary['success_fraction_lower_bound'],.5)
    def test_export_preserves_decimal_speed_in_filename(self):
        from g1_analysis.plots import save
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            for speed in ('0.12', '0.30'):
                fig, ax = plt.subplots()
                ax.plot([0,1],[0,1])
                save(fig, Path(directory)/('success_'+speed))
                for extension in ('png','pdf','svg'):
                    self.assertTrue((Path(directory)/('success_'+speed+'.'+extension)).is_file())

    def test_wilson_nonzero_width_for_all_success(self):
        low,high = wilson(3,3)
        self.assertLess(low,1)
        self.assertAlmostEqual(high,1)

if __name__ == '__main__':
    unittest.main()
