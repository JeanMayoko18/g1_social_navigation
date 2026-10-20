"""Synthetic regression checks, not experimental evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest

p=Path(__file__).resolve().parents[1]/'analyze_social_bags.py'
spec=importlib.util.spec_from_file_location('analysis',p)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AnalysisTests(unittest.TestCase):
    def test_observation_gap_is_not_stationary_time(self):
        rows=[dict(t=t,x=0.,y=0.,speed=0.,wz=0.,frame='odom',header_stamp=None)
              for t in [0.,.25,.5,10.,10.25,10.5]]
        _,m,_=a.motion_metrics(rows,0.,10.5,a.DEFAULTS)
        self.assertAlmostEqual(m['stationary_time_observed_s'],1.)
        self.assertLess(m['motion_time_coverage'],.1)

    def test_cancel_reissue_does_not_infer_mission_failure(self):
        d={'end':10.,'statuses':[
          {'topic':a.NAV_STATUS,'goal_id':'a','status':'executing','t':1.},
          {'topic':a.NAV_STATUS,'goal_id':'a','status':'canceled','t':3.},
          {'topic':a.NAV_STATUS,'goal_id':'b','status':'executing','t':5.},
          {'topic':a.NAV_STATUS,'goal_id':'b','status':'succeeded','t':9.}]}
        self.assertEqual(a.window_for(d,{})[2:],('recording_only','unknown'))

    def test_single_public_goal_inference(self):
        d={'end':10.,'statuses':[
          {'topic':a.NAV_STATUS,'goal_id':'a','status':'executing','t':1.},
          {'topic':a.NAV_STATUS,'goal_id':'a','status':'succeeded','t':9.}]}
        self.assertEqual(a.window_for(d,{}),(1.,9.,'inferred_single_public_goal_status','nav2_succeeded'))

    def test_unknown_human_schema_does_not_produce_valid_metrics(self):
        d={'origin':100.,'humans':[{'t':1.,'id':'p','x':.5,'y':0.,'frame':'base_footprint',
                                   'observation_stamp':101.,'collection_stamp':101.}]}
        _,m,_=a.human_distances(d,a.pd.DataFrame(),0,2,a.DEFAULTS,False)
        self.assertIsNone(m['human_min_distance_estimate_m'])
        _,m,_=a.human_distances(d,a.pd.DataFrame(),0,2,a.DEFAULTS,True)
        self.assertEqual(m['human_min_distance_estimate_m'],.5)
        d['humans'][0]['observation_stamp']=90.
        _,m,_=a.human_distances(d,a.pd.DataFrame(),0,2,a.DEFAULTS,True)
        self.assertIsNone(m['human_min_distance_estimate_m'])

    def test_generated_text_is_not_spoken_evidence(self):
        d={'events':[], 'hri':[{'t':1.,'topic':'/explainability/text','role':'robot_text',
            'text':'Synthetic test message','speech_id':'id1','status':'','data':{}}]}
        _,rows=a.evidence_examples(d,a.DEFAULTS)
        self.assertEqual(rows[0]['delivery_evidence'],'unverified')
        d['hri'].append({'t':2.,'topic':'/explainability/speech_status','role':'speech_status',
                         'text':'','speech_id':'wrong_id','status':'completed','data':{}})
        _,rows=a.evidence_examples(d,a.DEFAULTS)
        self.assertEqual(rows[0]['delivery_evidence'],'unverified')

if __name__=='__main__':unittest.main()
