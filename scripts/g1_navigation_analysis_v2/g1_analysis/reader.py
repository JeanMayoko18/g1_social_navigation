"""ROS-dependent boundary. Metrics and plots operate on ordinary Python records.

Retain header timestamps and frames. Count every recorded topic, but deserialize
only supported messages. No TF transformation or clock-offset estimation is hidden
inside this reader. Frame-incompatible metrics are refused downstream.
"""
from collections import Counter
import math
import yaml

COMMANDS = ['/cmd_vel', '/cmd_vel_nav_scaled', '/cmd_vel_mux', '/cmd_vel_rlsar']
ODOMETRY = ['/odom', '/dog_odom', '/dog_odom_2d', '/odom_dog_calibrated',
            '/odom_unitree', '/odom_mola_nav']


def yaw(q):
    norm = math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
    if not math.isfinite(norm) or norm < 1e-9:
        return float('nan')
    w,x,y,z = q.w/norm,q.x/norm,q.y/norm,q.z/norm
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def stamp(s):
    return float(s.sec) + float(s.nanosec) * 1e-9


def read_bag(path, settings):
    from rclpy.serialization import deserialize_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rosidl_runtime_py.utilities import get_message

    metadata = yaml.safe_load((path / 'metadata.yaml').read_text())
    storage = metadata['rosbag2_bagfile_information']['storage_identifier']
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(path), storage_id=storage),
                ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    status_topics = {topic for topic, typename in types.items() if typename == 'action_msgs/msg/GoalStatusArray' and ('navigate_to_pose' in topic)}
    status_topics.add(settings['status_topic'])
    statuses_by_topic = {topic: [] for topic in status_topics}
    wanted = set(COMMANDS + ODOMETRY + ['/amcl_pose', '/goal_pose', '/plan', settings['status_topic']])
    wanted.update(status_topics)
    classes = {t: get_message(types[t]) for t in wanted if t in types}
    data = dict(streams={t: [] for t in wanted}, statuses=[], counts=Counter(),
                types=types, bag_start=None, bag_end=None, warnings=[], storage=storage)
    while reader.has_next():
        topic, raw, ns = reader.read_next()
        bt = ns * 1e-9
        data['counts'][topic] += 1
        data['bag_start'] = bt if data['bag_start'] is None else min(bt, data['bag_start'])
        data['bag_end'] = bt if data['bag_end'] is None else max(bt, data['bag_end'])
        if topic not in classes:
            continue
        msg = deserialize_message(raw, classes[topic])
        if topic in status_topics:
            for status in msg.status_list:
                statuses_by_topic[topic].append(dict(t=bt, goal_stamp=stamp(status.goal_info.stamp),
                    goal_id=bytes(status.goal_info.goal_id.uuid).hex(), status=int(status.status)))
            continue
        header = getattr(msg, 'header', None)
        ht = stamp(header.stamp) if header else None
        frame = header.frame_id if header else None
        t = ht if settings['time_basis'] == 'header' and ht is not None else bt
        if settings['time_basis'] == 'header' and ht is not None and ht <= 0:
            data['warnings'].append(f'{topic}: zero header timestamp skipped')
            continue
        row = dict(t=t, bag_t=bt, header_t=ht, frame=frame)
        if topic in COMMANDS:
            twist = getattr(msg, 'twist', msg)
            row.update(vx=float(twist.linear.x), vy=float(twist.linear.y), wz=float(twist.angular.z))
        elif topic in ODOMETRY or topic == '/amcl_pose':
            pose = msg.pose.pose
            row.update(x=float(pose.position.x), y=float(pose.position.y), yaw=yaw(pose.orientation))
            if topic in ODOMETRY:
                row.update(child_frame=msg.child_frame_id, vx=float(msg.twist.twist.linear.x),
                           vy=float(msg.twist.twist.linear.y), wz=float(msg.twist.twist.angular.z))
        elif topic == '/goal_pose':
            row.update(x=float(msg.pose.position.x), y=float(msg.pose.position.y), yaw=yaw(msg.pose.orientation))
        elif topic == '/plan':
            row['endpoint_yaw'] = yaw(msg.poses[-1].pose.orientation) if msg.poses else float('nan')
            row['points'] = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]
        data['streams'][topic].append(row)
    available = [topic for topic, rows in statuses_by_topic.items() if rows]
    chosen = settings['status_topic'] if statuses_by_topic.get(settings['status_topic']) else (available[0] if len(available)==1 else None)
    data['statuses'] = statuses_by_topic.get(chosen, [])
    data['status_topic_audit'] = dict(selected_topic=chosen,
        extracted_status_entries={topic: len(rows) for topic, rows in statuses_by_topic.items()},
        recorded_messages={topic: data['counts'].get(topic,0) for topic in status_topics})
    if chosen is None:
        data['warnings'].append('No unambiguous NavigateToPose status stream with entries. Review annotations or status_topic setting.')
    for rows in data['streams'].values():
        rows.sort(key=lambda r: r['t'])
    data['warnings'] = sorted(set(data['warnings']))
    data['counts'] = dict(data['counts'])
    return data
