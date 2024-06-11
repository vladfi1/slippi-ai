# Streams the X display :99 to twitch.tv. You must run xvfb.sh first and set
# DISPLAY=:99 on the dolphin process.

TWITCH_STREAM=rtmp://live.twitch.tv/app/$TWITCH_KEY

# TODO: for some reason the twitch stream is often choppy
# TODO: restart command every 2 days to bypass Twitch stream cap

ffmpeg -report \
  -f x11grab -i :99 \
  -f alsa -ac 2 -i default \
  -c:v h264_nvenc -g 60 -preset fast \
  -b:v 3000k -maxrate 3000k -bufsize 6000k \
  -c:a aac -ar 44100 \
  -f flv \
  $TWITCH_STREAM
