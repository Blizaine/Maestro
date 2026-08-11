# Generation Completion Notifications Research and Plan

Status: research complete; implementation deferred

Research date: 2026-08-11

Audited Maestro baseline: `30ff6da` (`v1.7.2`)

## Purpose

Preserve the product and implementation plan for notifying users when a long
generation, queue, Director project, repair, or other top-level task completes
or fails. Maestro should provide useful alerts without notification spam,
exposing private prompts, or weakening its local-first positioning.

The system should be layered because desktop host notifications, browser
notifications, custom sounds, and reliable background mobile delivery have
different platform constraints.

## Executive recommendation

Implement three progressively broader notification levels:

1. **Fully local desktop alerts:** use Pinokio's native `pterm push` command
   with optional sound. This remains useful when the Maestro browser tab is
   minimized or closed because the running backend triggers the host operating
   system notification.
2. **Connected browser alerts:** add an optional in-page chime and standards-
   based browser notification through a service worker while a browser client
   can observe completion events.
3. **Optional background mobile push:** add standards-based Web Push for users
   who want lock-screen delivery when the phone or PWA is suspended. Label this
   clearly because Apple or Google transports the encrypted push message; it
   is not strictly offline/local transport.

Do not require a Maestro cloud account for any level. If Web Push is added,
generate and retain the installation's VAPID keys and subscriptions locally.

## Capability matrix

| Delivery method | Browser can be closed | Mobile support | Strictly local | Important limitation |
|---|---:|---:|---:|---|
| Pinokio host notification | Yes | No | Yes | Alerts the machine running Maestro, not a remote phone |
| In-page custom chime | No | Foreground reliably | Yes | Browser audio must first be unlocked by a user gesture |
| Browser/service-worker notification | Sometimes | Android and iOS PWA with restrictions | Yes when completion is already observed locally | Requires a secure origin and cannot reliably wake a suspended browser without push |
| Standards-based Web Push | Yes | Yes | No | Uses the browser vendor's push service |

## Maestro-specific findings

### Pinokio already provides native notifications

The bundled `pterm push` command sends a desktop notification and accepts:

- `--title`;
- `--subtitle`;
- `--image`;
- `--sound`;
- wait/timeout controls.

This is the smallest reliable desktop implementation. The backend should invoke
the executable as an argument array after feature detection, never by
interpolating prompt text into a shell command. If `pterm` is unavailable in a
standalone Maestro installation, the host-notification path should disable
itself without affecting generation.

Reference: `C:/Users/bliza/pinokio/prototype/PTERM.md`, section `push`.

### Maestro already observes terminal state

The React store detects ordinary job completion, reconnected-job completion,
and Director completion. Several Edit tools also contain their own job polling
loops. Adding notification calls separately to every poller would be fragile
and would produce duplicate alerts.

Introduce a central terminal-event path instead. Both the host notification
and every browser client should consume the same durable, deduplicated event.

### Maestro is not currently a PWA

The current UI has no web-app manifest or service worker. Those are required
before providing a consistent mobile notification experience. They should be
introduced carefully so a stale service-worker cache cannot recreate the UI
asset/version mismatch problems Maestro has previously guarded against.

## Notification event model

Create a backend notification/event broker that records only user-facing parent
operations. A terminal event should include:

- stable event ID;
- top-level job or pipeline ID;
- operation type (`generation`, `director`, `repair`, `download`, or future
  types);
- terminal state (`completed` or `failed`);
- completion timestamp;
- active-generation elapsed time when available;
- output count and a safe output identifier;
- whether the operation belongs to a larger queue;
- privacy-safe display title and body.

Do not emit ordinary completion alerts for:

- individual Director child clips;
- internal sliding windows;
- intermediate image/keyframe tasks;
- model downloads that are part of a still-running generation;
- repair child jobs;
- cancelled jobs unless the user explicitly enables cancellation alerts.

Failures should be eligible for immediate notification because user action may
be required. Successful batches should default to one notification when the
top-level queue becomes idle.

## Durable delivery and deduplication

The backend should keep a small bounded journal of recent terminal events. A
browser requests events newer than its last acknowledged event ID. This solves
several problems:

- a page refresh does not lose a completion that occurred during reload;
- reconnected jobs do not produce a second alert;
- completed jobs can be reported after they disappear from the active-jobs
  endpoint;
- multiple UI pollers do not need notification-specific code;
- future Web Push can subscribe to the same terminal event stream.

Browser clients should persist their last acknowledged event ID locally.
Across multiple tabs, use a `BroadcastChannel`, a service-worker notification
tag, or equivalent leader/deduplication mechanism so only one tab plays the
chime and the OS receives one notification per event.

## Proposed settings

Add a Notifications section in Settings with a master opt-in control. Suggested
controls:

- **Completion alerts** (master toggle, off by default);
- **Host desktop notification**;
- **Browser notification** with permission status and an Enable/Test button;
- **Completion chime**;
- chime volume and Test button;
- notify on completed jobs;
- notify immediately on failures;
- notify after every generation or only when the queue finishes;
- notify only when Maestro is not focused;
- include operation details or use privacy-safe generic text;
- optional cancellation alerts.

The permission and sound Test button must be a direct click/tap. Browsers often
block audio and notification permission requests that are not initiated by a
user gesture.

Suggested privacy-safe default notification:

```text
Maestro generation complete
1 video finished in 12m 43s.
```

Do not include prompts, character names, filenames, workspace names, or mature
content unless the user explicitly enables detailed notifications.

## Sound behavior

Browser notification APIs do not provide a portable custom sound-file option.
The operating system controls the sound associated with a browser notification.
Maestro's custom chime should therefore be implemented separately with Web
Audio or a bundled local audio asset.

The chime should:

- remain optional;
- be unlocked by the user's Test/Enable gesture;
- use a short bundled or synthesized sound with no network dependency;
- catch and report autoplay rejection without affecting the job;
- avoid playing once per child job;
- respect browser focus and queue-level preferences.

Pinokio host notifications can use the native `--sound` option independently
of the browser chime.

## Desktop behavior

On a Pinokio-hosted installation, the backend should send a native host alert
when the configured top-level operation completes. This path should work even
when:

- the Maestro browser tab was closed;
- the user is working in another application;
- the UI client lost its polling connection;
- several different browser devices have connected to Maestro.

Click behavior should open or focus Maestro and, where routing permits, select
the completed output or Director project.

## Mobile behavior

### Foreground mobile browser

An enabled chime and in-app alert can work while Maestro is open and the page
is active. Mobile operating systems may suspend timers, audio, and network
connections when the browser is backgrounded or the phone locks, so this is
not a reliable background-notification solution.

### Android

Use a service worker and `ServiceWorkerRegistration.showNotification()` from a
trusted HTTPS origin. Installation as a PWA improves the experience but is not
the fundamental Web Push transport requirement.

### iPhone and iPad

Standards-based Web Push requires iOS/iPadOS 16.4 or newer and a Maestro web app
added to the Home Screen. Permission must be requested after direct user
interaction. Delivery uses Apple Push Notification service and does not require
an Apple Developer Program membership.

### Local-network HTTP limitation

`http://localhost` can be treated as a secure origin on the same machine, but a
phone loading Maestro from a private LAN IP such as `http://192.168.x.x` is not
generally a secure context. Service workers and browser notifications therefore
need a trusted HTTPS origin on the phone.

The UI should use feature detection (`window.isSecureContext`, service-worker
availability, Notification API availability, and current permission) and show
an exact explanation instead of offering a toggle that cannot work.

## Background mobile push and local-first disclosure

Reliable lock-screen delivery while the browser/PWA is suspended requires Web
Push. The local Maestro backend sends an encrypted request directly to the
browser subscription endpoint, which Apple, Google, Mozilla, or another browser
vendor operates. The vendor routes it to the device and wakes the service
worker.

This mode can remain private in content—the Web Push payload is encrypted—but
it is not offline or strictly local transport. Present it as a separate opt-in:

> Mobile background notifications use your browser provider's encrypted push
> service. Generation, media, prompts, and models remain local.

Do not imply that a local WebSocket can replace Web Push on iOS; the operating
system will suspend it in the background.

## Implementation phases

### Phase 1: Fully local alerts

1. Add the central terminal-event broker and bounded event journal.
2. Emit events from the common job lifecycle and Director pipeline lifecycle.
3. Add queue/parent-child suppression and event deduplication.
4. Add feature-detected Pinokio host notifications with optional native sound.
5. Add Notifications settings and privacy-safe defaults.
6. Add the opt-in in-page chime and Test button.
7. Add regression tests covering success, failure, queue aggregation, Director
   child suppression, reconnects, and duplicate terminal updates.

### Phase 2: Browser and PWA support

1. Add a version-safe web-app manifest and service worker.
2. Add service-worker browser notifications for connected secure clients.
3. Add permission, HTTPS, and platform diagnostics in Settings.
4. Add notification-click focusing/navigation.
5. Validate Chrome, Edge, Firefox, Safari on macOS, Android Chrome, and iOS
   Home Screen behavior.
6. Verify UI updates never leave an old service worker serving incompatible
   asset hashes.

### Phase 3: Optional background mobile Web Push

1. Generate per-installation VAPID credentials locally.
2. Store subscriptions locally and provide per-device removal.
3. Send minimal encrypted terminal-event payloads directly from Maestro.
4. Add retry/expiry handling without blocking job finalization.
5. Add the explicit non-local-transport disclosure.
6. Test browser-closed and phone-locked delivery on Android and iOS.

## Acceptance criteria

- A normal Studio job generates exactly one configured completion alert.
- A Director project generates one alert when the entire pipeline finishes,
  never one per clip or window.
- Failures can alert immediately without a success alert afterward.
- A reconnected or repeatedly polled terminal job is not announced twice.
- Closing the Maestro tab does not prevent a configured Pinokio host alert.
- A user can test browser permission and chime playback before a long run.
- Notification content is generic by default and never leaks prompts.
- Unsupported or insecure mobile contexts receive a clear explanation.
- Disabling notifications prevents all host, browser, sound, and push activity.
- Notification failures never change job completion state.

## Primary references

- Pinokio desktop notification command:
  `C:/Users/bliza/pinokio/prototype/PTERM.md#push`
- MDN, Using the Notifications API:
  <https://developer.mozilla.org/en-US/docs/Web/API/Notifications_API/Using_the_Notifications_API>
- MDN, service-worker `showNotification()`:
  <https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerRegistration/showNotification>
- MDN, secure contexts:
  <https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Secure_Contexts>
- MDN, media/Web Audio autoplay behavior:
  <https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Autoplay>
- WebKit, Web Push for iOS and iPadOS Home Screen web apps:
  <https://webkit.org/blog/13878/web-push-for-web-apps-on-ios-and-ipados/>
- Web Push architecture overview:
  <https://web.dev/articles/push-notifications-overview>
