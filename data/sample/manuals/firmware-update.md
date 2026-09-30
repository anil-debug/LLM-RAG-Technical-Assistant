# AstraRack X11 firmware update

## Release tracking

This guide covers release train ARX-77. ARX-77 is an internal tracking code and is not a firmware version. Do not type ARX-77 into the firmware upload field.

## BMC and BIOS order

Apply BMC firmware 01.73.12 before BIOS 2.8.4. Do not remove power while the BMC flash is running. The SEL marker FW_UPDATE_BMC is written when the BMC flash starts. The BIOS setup password is stored in the BIOS region and survives a BMC factory reset.

## GPU board

GPU board GPU_SXM_1 uses firmware bundle AR-GPU-1.6.0. Update that bundle only when the host is in maintenance mode and GPU_SXM_1_TEMP is below 70 C. The same out-of-band style is used on NVIDIA HGX platforms, but this guide supports only bundle AR-GPU-1.6.0.

## NVMe kit

The Astra NVMe kit firmware is 2.4.1. The drive must sit in a non-RAID slot before the update starts. A RAID member will reject the download. If the update reports NVME-05, the SMART critical warning bit is set. Abort the update and replace the drive. Do not retry NVME-05 on the same device.
