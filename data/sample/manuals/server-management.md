# AstraRack X11 server management

## The baseboard management controller

The BMC is the baseboard management controller on the AstraRack X11. It stays powered when the host operating system is off. Operators reach it by the dedicated hostname astrarack-bmc.local. The web UI listens on TCP 443 and is independent of the host bond.

## Power sensors

PSU1_VIN is the input-voltage sensor for the first supply. The documented nominal range is 200-240 VAC. A reading outside that range is a site-power fault, not a firmware fault.

## System event log

The SEL stores hardware events even when the host cannot boot. A BMC firmware flash writes the SEL marker FW_UPDATE_BMC when the flash starts. Use that marker to confirm that an update began.

## Factory reset

Hold the chassis ID button for 10 seconds to factory-reset the BMC. The reset clears the BMC network configuration, including the address on the dedicated NIC. It does not clear the BIOS setup password. That password lives in the BIOS region, not in the BMC configuration store.

## Web session length

The BMC web UI ends an idle session after 30 minutes. Opening a new browser tab does not extend a session that has already expired.
