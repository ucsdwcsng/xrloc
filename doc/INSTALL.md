# install tool chain

You need to install
- compiler: gcc-arm-none-eabi
- flash programmer: stlink-tools

Both of them can be easily installed by apt install in Ubuntu 20.04.

## compiler: gcc-arm-none-eabi

```
$ sudo apt install gcc-arm-none-eabi
```

If the install is succeeded, you can see

```
$ arm-none-eabi-gcc --version
arm-none-eabi-gcc (15:9-2019-q4-0ubuntu1) 9.2.1 20191025 (release) [ARM/arm-9-branch revision 277599]
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

```

Now you can make `uloc_firmware`.

No need to add ppa:terry.guo/gcc-arm-embedded


## flash programmer: stlink-tools


```
$ sudo apt install stlink-tools
```

I have installed v1.6.0.

```
$ st-flash --version
v1.6.0
```


`lsusb` we can see ST-LINK and Virtual COM port like this.

```
$ lsusb
Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 003 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 003: ID 0cf3:e009 Qualcomm Atheros Communications
Bus 001 Device 007: ID 0483:374b STMicroelectronics ST-LINK/V2.1
Bus 001 Device 006: ID 0483:5740 STMicroelectronics Virtual COM Port
Bus 001 Device 005: ID 1a40:0101 Terminus Technology Inc. Hub
Bus 001 Device 004: ID 046d:c52b Logitech, Inc. Unifying Receiver
Bus 001 Device 002: ID 058f:6366 Alcor Micro Corp. Multi Flash Reader
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```

### erase

It takes a few seconds.

```
$ st-flash erase
st-flash 1.6.0
2022-08-12T19:41:30 INFO common.c: Loading device parameters....
2022-08-12T19:41:30 INFO common.c: Device connected is: F1 Connectivity line device, id 0x10016418
2022-08-12T19:41:30 INFO common.c: SRAM size: 0x10000 bytes (64 KiB), Flash: 0x40000 bytes (256 KiB) in pages of 2048 bytes
Mass erasing
```

### write

It takes several seconds.


```
$ st-flash write multiposition.bin 0x8000000
st-flash 1.6.0
2022-08-12T19:41:39 INFO common.c: Loading device parameters....
2022-08-12T19:41:39 INFO common.c: Device connected is: F1 Connectivity line device, id 0x10016418
2022-08-12T19:41:39 INFO common.c: SRAM size: 0x10000 bytes (64 KiB), Flash: 0x40000 bytes (256 KiB) in pages of 2048 bytes
2022-08-12T19:41:39 INFO common.c: Attempting to write 30380 (0x76ac) bytes to stm32 address: 134217728 (0x8000000)
Flash page at addr: 0x08007000 erased
2022-08-12T19:41:39 INFO common.c: Finished erasing 15 pages of 2048 (0x800) bytes
2022-08-12T19:41:39 INFO common.c: Starting Flash write for VL/F0/F3/F1_XL core id
2022-08-12T19:41:39 INFO flash_loader.c: Successfully loaded flash loader in sram
 15/15 pages written
2022-08-12T19:41:40 INFO common.c: Starting verification of write complete
2022-08-12T19:41:40 INFO common.c: Flash written and verified! jolly good!
```


## memo

I have tried to follow this.
- https://ukhas.org.uk/guides:stm32toolchain

But, many things can be installed with apt.


openocd, which is used in uloc_firmware, does not work well.
Especially, it is hard to use Nucleo as a writer programmer.

Nucleo's firmware sometimes has a problem.
When `st-flash` does not run correctly, you should try to update the firmware.
- [STSW-LINK007 - ST-LINK, ST-LINK/V2, ST-LINK/V2-1, STLINK-V3 boards firmware upgrade - STMicroelectronics](https://www.st.com/en/development-tools/stsw-link007.html)


