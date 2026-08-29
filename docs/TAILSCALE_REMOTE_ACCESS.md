# Use Maestro Remotely with Tailscale

Tailscale lets you open Maestro securely from another computer, phone, or
tablet—even when you are away from home. Maestro continues running on your own
computer and is available only to devices authorized in your private Tailscale
network.

This setup is optional. Each person uses their own Tailscale account. Maestro
uses **Tailscale Serve**, never the public Tailscale Funnel feature.

## What you need

- The computer that runs Maestro, called the **Maestro computer** below.
- The phone, tablet, or second computer you want to use remotely.
- Tailscale installed on both devices.
- Both devices signed into the same Tailscale account or authorized in the same
  tailnet.
- Maestro running in Pinokio whenever you want to use it remotely.

## One-time setup

### 1. Create a Tailscale account

1. Open [tailscale.com/start](https://login.tailscale.com/start).
2. Sign in with one of the offered account providers.
3. Choose the personal-use option when prompted.

This creates your private Tailscale network, called a *tailnet*.

### 2. Install Tailscale on the Maestro computer

1. Download Tailscale from [tailscale.com/download](https://tailscale.com/download).
2. Install and open it.
3. Sign in with the account you created above.
4. Confirm that Tailscale says it is connected.

### 3. Install Tailscale on the remote device

- **iPhone or iPad:** install Tailscale from the App Store and approve its VPN
  configuration.
- **Android:** install Tailscale from Google Play and approve its VPN
  connection.
- **Mac or Windows:** download it from
  [tailscale.com/download](https://tailscale.com/download).

Sign in using the same Tailscale account as the Maestro computer, then confirm
that Tailscale is connected.

### 4. Enable private Maestro access

1. Start Maestro in Pinokio and wait for **Open Web UI** to appear.
2. Return to Maestro's page in Pinokio.
3. Select **Secure Remote Access (Tailscale)**.
4. Approve the operating-system permission prompts. Windows can show more than
   one approval while Tailscale enables its HTTPS certificate and persistent
   Serve route.
5. If Tailscale opens a browser page asking to enable HTTPS for the tailnet,
   approve it.
6. In Maestro, open **Settings → Notifications**.
7. Under **Private phone access**, confirm that private HTTPS access is enabled.
8. Copy the secure URL or scan the QR code with the remote device.

The address looks similar to:

```text
https://your-computer.your-tailnet.ts.net
```

Only devices and users permitted by that Tailscale network can open the
address. It is not a public Maestro link.

## Everyday remote use

1. Leave the Maestro computer powered on and connected to the internet.
2. Make sure Tailscale is connected on the Maestro computer.
3. Start Maestro normally in Pinokio.
4. Connect Tailscale on the remote device.
5. Open the saved Maestro HTTPS address.

The setup remembers Maestro's selected backend port and normally reuses the
same private address after app or computer restarts. You do not need to run the
setup action or scan the QR code every time. If that port is occupied and
Maestro has to choose another one, run **Secure Remote Access (Tailscale)** once
more to adopt the new port.

If Maestro is stopped or the host computer is off, the remote page is
unavailable. It becomes available again when Maestro is running.

## Install Maestro on an iPhone or iPad Home Screen

The installed web app provides a full-screen interface and is required for
closed-app notifications on iPhone and iPad.

1. Open the Maestro **HTTPS** address in Safari.
2. Tap **Share**.
3. Choose **Add to Home Screen**.
4. Tap **Add**.
5. Close Safari and open Maestro from its Home Screen icon.

If Maestro was previously installed from a local `http://` address, remove that
old Home Screen shortcut and add it again from the Tailscale `https://` address.

## Enable completion notifications

On the remote device:

1. Open **Maestro → Settings → Notifications**.
2. Enable **System notifications**.
3. Approve the browser or operating-system permission request.
4. Choose the Complete, Failed, and Queue events you want.
5. Optionally enable background-only alerts and a chime on that device.
6. Use **Test notification** and **Test closed-app notification**.

On iPhone and iPad, request permission from the installed Maestro Home Screen
app, not from a normal Safari or Chrome tab. Background Web Push requires iOS
or iPadOS 16.4 or later.

## Troubleshooting

### The private address will not open

- Confirm that Maestro is running on the host computer.
- Confirm that Tailscale says **Connected** on both devices.
- Confirm that both devices are in the same tailnet.
- In Maestro, open **Settings → Notifications** and refresh **Private phone
  access**.
- If the route is missing, run **Secure Remote Access (Tailscale)** from
  Maestro's Pinokio page while Maestro is running.

### Maestro says Tailscale is unavailable

- Open the Tailscale desktop app on the Maestro computer.
- Sign in or reconnect it.
- Return to Maestro and press the refresh control beside **Private phone
  access**.

### iPhone or iPad will not enable notifications

- Verify that the address begins with `https://`.
- Verify that Maestro was added to the Home Screen from that secure address.
- Open the installed Home Screen app before enabling notifications.
- Check **iOS Settings → Notifications → Maestro** if permission was denied.
- Confirm that the device uses iOS or iPadOS 16.4 or later.

### Maestro reports only foreground notifications

- Use Maestro's normal **Update** action once so the Web Push dependency is
  installed.
- Restart Maestro.
- Reopen the installed Maestro app and enable **System notifications** again.

### Another Tailscale Serve route is already configured

Maestro will not overwrite a different private service already using Tailscale
Serve on the same computer. Disable or relocate that route before enabling
Maestro private access.

## Disable private access

Turn off **Private HTTPS access** in Maestro's Notifications settings. An
advanced user can also run:

```text
tailscale serve --https=443 off
```

Disabling Tailscale access does not delete Maestro projects, models, outputs,
or notification preferences.

## Privacy and security

- Maestro does not create or manage a Tailscale account.
- Maestro does not operate a cloud relay for remote access.
- Tailscale Serve keeps the Maestro address inside the private tailnet.
- Maestro never enables Tailscale Funnel.
- Anyone allowed to open the Maestro address can control that local Maestro
  installation. Protect the Tailscale account and authorize only trusted
  devices and users.

## Official references

- [Tailscale quickstart](https://tailscale.com/kb/1017/install/)
- [Download and install Tailscale](https://tailscale.com/docs/install)
- [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve)
- [Tailscale pricing](https://tailscale.com/pricing)
- [Install Tailscale on iPhone or iPad](https://tailscale.com/docs/install/ios)
- [Web Push for iPhone and iPad Home Screen apps](https://webkit.org/blog/13878/web-push-for-web-apps-on-ios-and-ipados/)
