"""Original training prompts. None of these strings are golden questions.

The wrappers in ``finetune.dataset`` expand each prompt into several training
questions. Answers are short statements grounded in the synthetic manuals.
"""

FACTS: list[dict[str, str]] = [
    {
        "prompt": "which sensor lists the nominal range 200-240 VAC?",
        "answer": "PSU1_VIN lists a nominal range of 200-240 VAC.",
        "source": "server-management.md",
    },
    {
        "prompt": "how long does an idle BMC web session last?",
        "answer": "The BMC web UI ends an idle session after 30 minutes.",
        "source": "server-management.md",
    },
    {
        "prompt": "how is a BMC factory reset started from the chassis?",
        "answer": "Hold the chassis ID button for 10 seconds to factory-reset the BMC.",
        "source": "server-management.md",
    },
    {
        "prompt": "which hostname is the DNS fallback for the BMC?",
        "answer": "Use astrarack-bmc.local when the management DNS entry is missing.",
        "source": "readme-notes.txt",
    },
    {
        "prompt": "which Redfish MessageId reports a stalled fan?",
        "answer": "A stalled fan reports MessageId AstraRack.1.0.FanStalled.",
        "source": "redfish-guide.md",
    },
    {
        "prompt": "which HTTP method deletes a Redfish session?",
        "answer": "Delete a session with DELETE /redfish/v1/SessionService/Sessions/{id}.",
        "source": "redfish-guide.md",
    },
    {
        "prompt": "which account role is blocked from firmware updates?",
        "answer": "The Operator role cannot update firmware.",
        "source": "redfish-guide.md",
    },
    {
        "prompt": "which older protocol still listens on UDP 623?",
        "answer": "The platform still accepts IPMI on UDP 623.",
        "source": "redfish-guide.md",
    },
    {
        "prompt": "which NVMe code means the SMART critical warning bit is set?",
        "answer": "NVME-05 means the SMART critical warning bit is set. Abort and replace the drive.",
        "source": "firmware-update.md",
    },
    {
        "prompt": "which tracking code must stay out of the firmware upload field?",
        "answer": "ARX-77 is an internal tracking code and is not a firmware version.",
        "source": "firmware-update.md",
    },
    {
        "prompt": "which BIOS build follows BMC firmware 01.73.12?",
        "answer": "Apply BMC firmware 01.73.12 before BIOS 2.8.4.",
        "source": "firmware-update.md",
    },
    {
        "prompt": "does a BMC factory reset erase the BIOS setup password?",
        "answer": "A BMC factory reset does not clear the BIOS setup password. It lives in the BIOS region.",
        "source": "server-management.md",
    },
    {
        "prompt": "which BMC interface raises NET-4401?",
        "answer": "A link-down condition on BMC eth0 raises error NET-4401.",
        "source": "networking-guide.md",
    },
    {
        "prompt": "which VLAN is checked while diagnosing NET-4401?",
        "answer": "After the cable, confirm VLAN 40 and gateway 10.40.0.1.",
        "source": "networking-guide.md",
    },
    {
        "prompt": "which host port documents an MTU of 9000?",
        "answer": "HOST_NIC1 carries storage traffic and its documented MTU is 9000.",
        "source": "networking-guide.md",
    },
    {
        "prompt": "which POST code points at CPU0_DIMM_A1?",
        "answer": "POST code 0xD4 means DIMM training failed on CPU0_DIMM_A1.",
        "source": "troubleshooting.md",
    },
    {
        "prompt": "which error code is reserved and has no sensor?",
        "answer": "Error 0x1A is reserved and has no assigned sensor.",
        "source": "troubleshooting.md",
    },
    {
        "prompt": "which fan does FRU-FAN-220 replace?",
        "answer": "Replace FRU-FAN-220 when SYS_FAN1 reports 0x1F again after a reseat.",
        "source": "troubleshooting.md",
    },
    {
        "prompt": "which GPU board drops a lane on FAB-12?",
        "answer": "FAB-12 means a lane on GPU_SXM_1 dropped training.",
        "source": "nvswitch-guide.md",
    },
    {
        "prompt": "which NVSwitch firmware ships in this release?",
        "answer": "The NVSwitch firmware for this release is AR-NVS-3.2.0.",
        "source": "nvswitch-guide.md",
    },
    {
        "prompt": "what GPU_SXM_1_TEMP limit gates the GPU bundle update?",
        "answer": "Update AR-GPU-1.6.0 only in maintenance mode when GPU_SXM_1_TEMP is below 70 C.",
        "source": "firmware-update.md",
    },
    {
        "prompt": "which slot type rejects the NVMe firmware download?",
        "answer": "The drive must sit in a non-RAID slot. A RAID member rejects the download.",
        "source": "firmware-update.md",
    },
    {
        "prompt": "does the BMC stay up when the host operating system is off?",
        "answer": "The BMC stays powered when the host operating system is off.",
        "source": "server-management.md",
    },
    {
        "prompt": "which response header carries a new Redfish session token?",
        "answer": "The session token is returned in the X-Auth-Token response header.",
        "source": "redfish-guide.md",
    },
    {
        "prompt": "how many seconds long is the Redfish SessionTimeout property?",
        "answer": "SessionTimeout is 1800 seconds.",
        "source": "redfish-guide.md",
    },
    {
        "prompt": "which TLS versions are rejected on the Redfish port?",
        "answer": "The BMC requires TLS 1.2 or newer. Older TLS versions are rejected.",
        "source": "redfish-guide.md",
    },
    {
        "prompt": "which SEL marker shows that a BMC flash has started?",
        "answer": "A BMC firmware flash writes the SEL marker FW_UPDATE_BMC when the flash starts.",
        "source": "server-management.md",
    },
    {
        "prompt": "what should automation match for a stalled fan instead of free text?",
        "answer": "Automation should match MessageId AstraRack.1.0.FanStalled.",
        "source": "redfish-guide.md",
    },
    {
        "prompt": "which GPU platform is named only as an out-of-band comparison?",
        "answer": "NVIDIA HGX is mentioned as a comparison. This guide supports only bundle AR-GPU-1.6.0.",
        "source": "firmware-update.md",
    },
    {
        "prompt": "what does the SEL record for a BMC eth0 link-down event?",
        "answer": "The sample SEL records NET-4401 on BMC eth0 with VLAN 40.",
        "source": "sel.log",
    },
]

WRAPPERS: tuple[str, ...] = (
    "{q}",
    "In the AstraRack X11 notes, {q}",
    "For the on-site technician, {q}",
    "Looking only at the synthetic manuals, {q}",
    "During a maintenance window, {q}",
    "When reading the platform guide, {q}",
    "A new operator asks: {q}",
    "Before closing the change ticket, {q}",
)
