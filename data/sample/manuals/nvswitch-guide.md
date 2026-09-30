# AstraRack X11 NVSwitch fabric

## Fabric module

The Astra fabric module connects four GPU boards. The NVSwitch firmware for this release is AR-NVS-3.2.0. Each GPU board keeps its own bundle, described in the firmware update guide.

## Dropped lane

Error FAB-12 means a lane on GPU_SXM_1 dropped training. Put the host in maintenance mode, reset the switch, and verify the link before returning the host to service. A FAB-12 event is a fabric fault, not a fan fault.

## Bundle compatibility

Do not run NVSwitch firmware AR-NVS-3.2.0 with a GPU bundle older than AR-GPU-1.6.0. Upgrade the GPU board first, then the switch. Mixing an older GPU bundle with AR-NVS-3.2.0 leaves the lane untrained.
