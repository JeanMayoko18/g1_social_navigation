"""Regression checks for legacy windows, proxy goals and original CSV import."""
import unittest
import tempfile
from pathlib import Path
from g1_analysis.legacy_metrics import publication_metrics, distance_metrics
from g1_analysis.discovery import annotations
from g1_analysis.replot import load_metrics

SETTINGS=dict(maximum_command_gap_s=.5,linear_activity_threshold_mps=.01,
              angular_activity_threshold_radps=.02,minimum_command_stop_s=1.,map_frame='map')

class PublicationTests(unittest.TestCase):
    def test_zero_commands_are_included_in_publication_window(self):
        cmd=[dict(t=i/4,vx=0.,vy=0.,wz=0.) for i in range(9)]
        amcl=[dict(t=i/4,x=i/4,y=0.,yaw=0.,frame='map') for i in range(9)]
        plan=dict(t=0.,points=[(0.,0.),(3.,0.)],endpoint_yaw=0.,frame='map')
        data=dict(streams={'/cmd_vel':cmd,'/amcl_pose':amcl,'/plan':[plan]})
        result=publication_metrics(data,SETTINGS,{})
        self.assertEqual(result['command_window_s'],2.)
        self.assertEqual(result['legacy_commanded_stop_count'],1)
        self.assertEqual(result['final_position_residual_m'],1.)
        self.assertEqual(result['goal_reference'],'initial_plan_endpoint_proxy')

    def test_gap_invalidates_complete_distance(self):
        rows=[dict(t=t,x=t/10,y=0.,yaw=0.,frame='odom') for t in (0,.2,2,2.2)]
        result=distance_metrics(rows,.5)
        self.assertIsNone(result['odom_distance_5hz_m'])
        self.assertAlmostEqual(result['odom_partial_distance_5hz_m'],.04)

    def test_stale_terminal_pose_does_not_create_residual(self):
        data=dict(streams={'/cmd_vel':[dict(t=t,vx=0.,vy=0.,wz=0.) for t in (0,10)],
            '/amcl_pose':[dict(t=0,x=0.,y=0.,yaw=0.,frame='map')],
            '/plan':[dict(t=0,points=[(0.,0.),(1.,0.)],endpoint_yaw=0.,frame='map')]})
        self.assertNotIn('final_position_residual_m',publication_metrics(data,SETTINGS,{}))

    def test_original_annotations_preserve_trial_token(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.csv'
            path.write_text('campaign,config,trial,status\nc,dog,04,Reached\n')
            self.assertEqual(annotations(path)['c__dog__trial_04']['outcome'],'SUCCEEDED')

    def test_original_csv_is_imported_without_changing_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'m.csv'
            path.write_text('campaign,scenario,config,speed_m_s,trial,status,used_odom_amcl_rmse_m\nc,Office corridor,dog,0.12,01,Reached,0.42\n')
            row=load_metrics(path)[0]
            self.assertEqual(row['legacy_used_odom_amcl_rmse_m'],.42)
            self.assertEqual(row['trial_number'],'01')
            self.assertEqual(row['scenario'],'corridor')

if __name__=='__main__': unittest.main()
