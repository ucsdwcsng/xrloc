# How to use udev

When you connect multiple EVB1000 to single computer, device name such as `/dev/ttyACM0` is confusing.
If you use udev rules, you can set a specific device name to each EVB1000.

## 1. Check serial number of EVB1000

if the target device name is `/dev/ttyACM0`, you can use `udevadm info` like following.

```
$ sudo udevadm info --attribute-walk /dev/ttyACM0 | grep serial
    ATTRS{serial}=="283006434E42323334FFD805"
    ATTRS{serial}=="0000:00:14.0"
$
```

`283006434E42323334FFD805` is the serial number.


## 2. Make `/etc/udev/rules.d/99-myrule.rules`

When you want to make `/dev/muloc01`, the `99-myrule.rules` should be like following.

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", ATTRS{serial}=="283006434E42323334FFD805", SYMLINK+="muloc01", MODE="0777"
```

## 3. Run some command


```
$ sudo udevadm control --reload-rules
$ sudo service udev reload
```

## 4. Unplug and plug

Unplug and plug the target device.

## Appendix

### debug

```
$ sudo udevadm monitor
```

