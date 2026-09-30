# AstraRack X11 Redfish guide

## What Redfish is

Redfish is the HTTPS JSON management API for the AstraRack X11. The service root is /redfish/v1/. Clients send and receive JSON. The BMC requires TLS 1.2 or newer on the Redfish port. Older TLS versions are rejected.

## Session login

Create a session with POST /redfish/v1/SessionService/Sessions. The JSON body contains UserName and Password. On success the session token is returned in the X-Auth-Token response header. Send that same header on later requests. The password is not repeated in the token.

## Session lifetime

The SessionTimeout property on the session service is 1800 seconds. That is the Redfish session length, distinct from the 30 minute BMC web UI idle timer.

## Ending a session

Delete a session with DELETE /redfish/v1/SessionService/Sessions/{id}. A deleted token cannot be reused. Create a new session instead of refreshing the old token.

## Accounts and firmware authority

Accounts are listed under /redfish/v1/AccountService. A user with the Operator role cannot update firmware. A user with the Administrator role can update firmware. Read-only accounts can collect SEL entries and sensor readings.

## Message identifiers

A stalled fan reports the Redfish MessageId AstraRack.1.0.FanStalled. Automation should match that MessageId rather than a free-text sentence.

## Preference over IPMI

IPMI is the older out-of-band protocol. The AstraRack X11 still accepts IPMI on UDP 623. Redfish is the preferred API for new automation because it returns JSON resources instead of raw IPMI completion codes.
