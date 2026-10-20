#!/usr/bin/env bash

# =====================================================
# OFFLINE ROS 2 HUMBLE ENVIRONMENT
# =====================================================
# ROS/ament setup scripts may reference unset variables.
# Temporarily disable nounset while loading them.

case $- in
    *u*) _OFFLINE_RESTORE_NOUNSET=1 ;;
    *)   _OFFLINE_RESTORE_NOUNSET=0 ;;
esac

set +u

source /opt/ros/humble/setup.bash
source "$HOME/ros4hri_ws/install/setup.bash"
source "$HOME/g1_ws/install/setup.bash"

if [[ "$_OFFLINE_RESTORE_NOUNSET" == 1 ]]; then
    set -u
fi

unset _OFFLINE_RESTORE_NOUNSET

export ROS_DOMAIN_ID=91
unset ROS_LOCALHOST_ONLY
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Force CycloneDDS to use loopback and extend
# the automatic DDS participant index range.
export CYCLONEDDS_URI='<CycloneDDS>
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="lo"/>
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="127.0.0.1"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>'