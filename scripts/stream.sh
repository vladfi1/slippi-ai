# Streams the X display :99 to twitch.tv. You must run xvfb.sh first and set
# DISPLAY=:99 on the dolphin process.

TWITCH_STREAM=rtmp://live.twitch.tv/app/$TWITCH_KEY

# Note: you need to Ctrl-C twice to exit this loop
while true;
do
  ffmpeg \
    -framerate 60 -f x11grab -i :99 \
    -f alsa -ac 2 -i default \
    -c:v h264_nvenc -g 120 -preset fast \
    -b:v 3000k -maxrate 3000k -bufsize 6000k \
    -c:a aac -ar 44100 \
    -f flv \
    $TWITCH_STREAM

  sleep 60
done
