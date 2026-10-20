#!/usr/bin/env python3
"""Generate clearly labelled synthetic bags for integration testing only."""
from pathlib import Path
import argparse
import numpy as np
import yaml
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores,get_typestore,get_types_from_msg

def generate(root):
    s=get_typestore(Stores.ROS2_FOXY)
    defs={
      'demo_msgs/msg/Person':'std_msgs/Header header\nstring track_id\ngeometry_msgs/Point position\n',
      'demo_msgs/msg/People':'std_msgs/Header header\ndemo_msgs/Person[] persons\n',
      'demo_msgs/msg/Speech':'string utterance_id\nstring text\nstring status\n',
      'demo_msgs/msg/Event':'string state\n'}
    reg={}
    for name,d in defs.items():reg.update(get_types_from_msg(d,name))
    s.register(reg)
    # External definitions also exercise legacy message registration.
    for name,d in defs.items():
        p=root/'definitions'/('demo_msgs/msg/'+name.split('/')[-1]+'.msg');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(d)
    def m(name,**kw):return s.types[name](**kw)
    def time(t):return m('builtin_interfaces/msg/Time',sec=int(t),nanosec=int(round((t-int(t))*1e9)))
    def header(t,frame):return m('std_msgs/msg/Header',stamp=time(t),frame_id=frame)
    def point(x,y):return m('geometry_msgs/msg/Point',x=float(x),y=float(y),z=0.)
    def vec(x,y,z=0.):return m('geometry_msgs/msg/Vector3',x=float(x),y=float(y),z=float(z))
    for conf in ['A','D']:
        folder=root/f'technical_{conf}_S1a_R01';folder.mkdir(parents=True,exist_ok=True)
        (folder/'trial_metadata.yaml').write_text(yaml.safe_dump({'trial_id':f'SYNTHETIC_{conf}_S1a_R01','trial_type':'synthetic',
           'configuration':{'id':conf},'scenario':{'id':'S1a'},'outcome':{'intervention_required':False}}))
        with Writer(folder/'navigation',version=8) as w:
            types={'/odom':'nav_msgs/msg/Odometry','/navigate_to_pose/_action/status':'action_msgs/msg/GoalStatusArray'}
            if conf=='D':types.update({'/social_perception/tracked_persons':'demo_msgs/msg/People',
                                     '/explainability/text':'demo_msgs/msg/Speech',
                                     '/explainability/speech_status':'demo_msgs/msg/Speech',
                                     '/social_navigation/navigation_state':'demo_msgs/msg/Event',
                                     '/humans/voices/pc_user/speech':'std_msgs/msg/String'})
            conn={topic:w.add_connection(topic,typ,typestore=s) for topic,typ in types.items()}
            def write(topic,t,obj):w.write(conn[topic],int(round((1000+t)*1e9)),s.serialize_cdr(obj,types[topic]))
            def status(t,code):
                info=m('action_msgs/msg/GoalInfo',goal_id=m('unique_identifier_msgs/msg/UUID',uuid=np.array([1]*16,dtype=np.uint8)),stamp=time(1000))
                write('/navigate_to_pose/_action/status',t,m('action_msgs/msg/GoalStatusArray',status_list=[m('action_msgs/msg/GoalStatus',goal_info=info,status=code)]))
            status(0,2)
            x=0.
            for i in range(101):
                t=i/10;v=.12 if not(conf=='D' and 3<=t<=5) else 0.
                x+=v*.1
                pose=m('geometry_msgs/msg/Pose',position=point(x,.15*np.sin(t/3) if conf=='D' else 0),orientation=m('geometry_msgs/msg/Quaternion',x=0.,y=0.,z=0.,w=1.))
                tw=m('geometry_msgs/msg/Twist',linear=vec(v,0),angular=vec(0,0))
                od=m('nav_msgs/msg/Odometry',header=header(1000+t,'odom'),child_frame_id='base_footprint',
                     pose=m('geometry_msgs/msg/PoseWithCovariance',pose=pose,covariance=np.zeros(36)),
                     twist=m('geometry_msgs/msg/TwistWithCovariance',twist=tw,covariance=np.zeros(36)))
                write('/odom',t,od)
                if conf=='D':
                    person=m('demo_msgs/msg/Person',header=header(1000+t,'base_footprint'),track_id='synthetic_person',position=point(1.5-.5*np.sin(t/3),.2))
                    write('/social_perception/tracked_persons',t,m('demo_msgs/msg/People',header=header(1000+t,'base_footprint'),persons=[person]))
                    if i in [0,30,50]:write('/social_navigation/navigation_state',t,m('demo_msgs/msg/Event',state={0:'NAVIGATING',30:'WAITING',50:'RESUMING'}[i]))
                    if i==31:write('/explainability/text',t,m('demo_msgs/msg/Speech',utterance_id='test1',text='SYNTHETIC TEST: I am waiting.',status='generated'))
                    if i==32:write('/explainability/speech_status',t,m('demo_msgs/msg/Speech',utterance_id='test1',text='',status='started'))
                    if i==40:write('/explainability/speech_status',t,m('demo_msgs/msg/Speech',utterance_id='test1',text='',status='completed'))
                    if i==45:write('/humans/voices/pc_user/speech',t,m('std_msgs/msg/String',data='SYNTHETIC TEST: You can continue.'))
            status(10.1,4)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    if args.output.exists():parser.error('Output must not already exist')
    generate(args.output)
