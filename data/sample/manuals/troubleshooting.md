# AstraRack X11 troubleshooting

## Fan stall

Error 0x1F means SYS_FAN1 stalled. Reseat SYS_FAN1 and ask the BMC to refresh sensors. If the SEL records 0x1F again after the reseat, replace FRU FRU-FAN-220. Do not replace a different fan first.

## Reserved code

Error 0x1A is reserved and has no assigned sensor. Treat 0x1A as unassigned. It does not mean SYS_FAN1 stalled. Only 0x1F maps to that fan.

## Memory training

POST code 0xD4 means DIMM training failed on CPU0_DIMM_A1. Reseat CPU0_DIMM_A1, then test the socket with one DIMM. A training failure is not fixed by reflashing the BMC.

## Out of band reachability

When the host cannot boot, use the BMC. IPMI remains available on UDP 623 for older consoles. Prefer the Redfish session API when the console supports it.

## What this guide does not publish

This guide does not publish a default BIOS password. It does not publish a default IPMI cipher suite. It does not state a warranty period, a chassis serial number, an LDAP server address, or a default SNMP community. Those values are site-specific and are absent from this document set. Do not guess them from a product family.
