source /opt/ros/humble/setup.bash
source ~/ros4hri_ws/install/setup.bash
source ~/g1_ws/install/setup.bash

echo "=== DIAGNOSTIC HRI ==="

ros2 topic echo /humans/voices/tracked \
  > /tmp/tracked.log 2>&1 &
PID1=$!

ros2 topic echo /humans/voices/pc_user/speech \
  > /tmp/speech.log 2>&1 &
PID2=$!

ros2 topic echo /intents \
  > /tmp/intents.log 2>&1 &
PID3=$!

echo "Dis maintenant : OUI"
sleep 10

kill $PID1 $PID2 $PID3 2>/dev/null || true

echo
echo "========== TRACKED =========="
cat /tmp/tracked.log

echo
echo "========== SPEECH =========="
cat /tmp/speech.log

echo
echo "========== INTENTS =========="
cat /tmp/intents.log
