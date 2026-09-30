# AstraRack X11 networking

## Management network

The BMC uses a dedicated NIC. Management traffic is on VLAN 40. The management gateway is 10.40.0.1. Host operating-system routes do not program this NIC.

## BMC link failures

A link-down condition on BMC eth0 raises error NET-4401. Check the cable, then confirm VLAN 40, then check the gateway 10.40.0.1. NET-4401 is not raised by a host bond failure.

## Host storage network

HOST_NIC1 carries storage traffic. The documented MTU for that port is 9000. The host bond is not managed by the BMC. Changing the BMC VLAN does not change the HOST_NIC1 MTU.
