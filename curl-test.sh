#!/bin/bash

for i in {1..10}
do
  curl -X POST https://77myn88rx6j94c-5000.proxy.runpod.net/generate_text \
    -F "prompt=center point coordinate of the sign up button. json output format only x,y" \
    -F "image=@/Users/logankeenan/test-image_original.webp" \
    -w "DNS Lookup Time: %{time_namelookup} ms\nTime to Connect: %{time_connect} ms\nTime to Start Transfer: %{time_starttransfer} ms\nTotal Time: %{time_total} ms\n"

  sleep 0.1
done
