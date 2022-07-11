#!/bin/bash


sudo modprobe msr

if [[ -z $(which rdmsr) ]]; then
    echo "msr-tools is not installed. Run 'sudo apt-get install msr-tools' to install it." >&2
    exit 1
fi



cores=$(cat /proc/cpuinfo | grep processor | awk '{print $3}')
for core in $cores; do
    sudo wrmsr -p${core} 0x1a0 0x4000850089
    state=$(sudo rdmsr -p${core} 0x1a0 -f 38:38)
    if [[ $state -eq 1 ]]; then
        echo "core ${core}: disabled"
    else
        echo "core ${core}: enabled"
    fi
done

sudo su - root <<HERE
/bin/echo "-1" > /proc/sys/kernel/perf_event_paranoid
echo 2 | dd of=/sys/devices/cpu/rdpmc
echo 2 | dd of=/sys/bus/event_source/devices/cpu/rdpmc
wrmsr -a 0x38d 0x0333
HERE

echo fin
