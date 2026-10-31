# Agent Access

Agent Access connects an MCP client to a single Maestro installation over
HTTPS. It uses a separate CPU controller, so the controller can stay available
while the model engine is stopped. Pinokio must still be running; Agent Access
does not start as a Windows service, wake the computer, or start Pinokio for
you.

The controller starts on loopback with agent connections and discovery off.
After you enable **Allow agent connections** in Maestro, it accepts connections
from your LAN and advertises its address with mDNS as `_maestro._tcp.local.`
Discovery only announces the endpoint. It does not approve a client or grant
access, and some clients may not show discovered servers automatically.

## Quick setup

1. Start Maestro from Pinokio and open **Settings → Agent Access**. If the
   model engine is stopped, open **Agent Access** from the Pinokio app menu;
   its separate terminal can run the controller and print the setup code.
2. Follow the three steps in the panel: **sign in**, **allow agent connections**,
   then **choose your agent**. For the initial sign-in, request a code and open
   **Pinokio → this Maestro app → Agent Access Terminal**. Copy the newest line
   labeled **Browser code:** into **Browser sign-in code**. Each code works
   once; the panel shows how long it remains valid. Request a fresh code if it
   expires or has already been used.
3. Choose **Copy setup prompt for my agent** and paste it into your agent's chat,
   or follow the client-specific instructions. Complete the client's OAuth
   approval in Maestro. Copying setup text does not connect or approve the agent.

Maestro remembers the signed-in browser until its session expires or you sign
out. Refreshing the page does not require a new code while the session remains
valid. The initial owner sign-in protects this installation's existing API and
other browsers also need their own sign-in. Pausing agent connections does not
remove that browser protection. An already signed-in owner can create a
short-lived code for another browser from the additional-browser controls.

## Connection details and certificate trust

Expand **Connection details** in the panel when configuring a client or
troubleshooting. For the first connection from each device or client:

1. Use **Download public TLS certificate**, then check the secure setup URL and TLS certificate SHA-256 fingerprint shown
   in Settings. Trust that exact certificate on each device and in each client
   that needs to connect. Follow the device or client instructions for adding
   a certificate; Maestro does not change system trust settings or firewall
   rules. Never turn off TLS verification.
2. Enable **Allow agent connections** when you are ready to connect an agent.
   This moves the listener from loopback to the LAN and enables local service
   discovery. Use the MCP URL shown in Settings, which uses this installation's
   `.local` hostname when mDNS is available. The controller also reports
   direct LAN IP URLs when the current certificate contains matching IP
   subject-alternative names; these remain usable when mDNS is disabled or
   unavailable.

Each Main and Dev installation has its own installation ID, endpoint, owner
settings, and client grants. Set up and approve each installation separately.

## Add an MCP client

Copy the MCP URL from **Settings → Agent Access**. It looks like
`https://maestro-<installation-id>.local:<port>/mcp`; use the exact value shown
by Maestro.

For Codex CLI, add the remote server and complete its OAuth login:

```sh
codex mcp add maestro --url "https://maestro-YOUR-ID.local:PORT/mcp"
codex mcp login maestro
```

For Claude Code, add it as a remote HTTP MCP server:

```sh
claude mcp add --transport http maestro "https://maestro-YOUR-ID.local:PORT/mcp"
```

Replace the example host and port with the exact MCP URL from Maestro. Read
[Codex MCP setup](https://learn.chatgpt.com/docs/extend/mcp?surface=cli) and
[Claude Code MCP setup](https://code.claude.com/docs/en/mcp) for client-specific
options. If your client uses another configuration format, add the same HTTPS
URL as a Streamable HTTP MCP server.

When a client requests access, review its name and requested scopes in the
Maestro approval page. Grant only what that client needs. Agent Access uses
OAuth authorization with PKCE; it does not put an access token in the copied
setup text. Review active clients in Settings and revoke a grant there when it
is no longer needed. Keep owner codes, approval codes, and client tokens
private; never put them in source control.

The mDNS record is for discovery on the local network. You can enter the MCP
URL manually if a client does not support discovery. When discovery is disabled
or unavailable, use a direct LAN URL only if it is listed in Agent Access
status (`lan_mcp_urls`). A URL is published only when its IP address is in the
controller certificate's subject-alternative names. New certificates include
private LAN IPv4 addresses found at setup. Existing certificates are preserved
when addresses change. They may lack a direct-IP URL until explicitly renewed
and trusted again. Trust the
exact certificate fingerprint before connecting.
Maestro does not change system trust settings, firewall rules, or network
configuration. Cloud-hosted connectors cannot reach a private LAN address
unless you provide a network bridge or tunnel; Agent Access does not provide
one. Not every client or platform combination has been verified.

## Choose scopes

Review the scope descriptions on the approval page. The operation catalogue
uses these scopes:

| Scope | What it allows |
| --- | --- |
| `observe` | Read status, capabilities, job receipts, shared workspace and gallery media, and Editor or Director state. |
| `generate` | Render or enhance media and run Director planning or generation work. |
| `manage` | Change projects, assets, and model files; upload media, start downloads, and run local model or storage scans. |
| `admin` | Change protected settings and service/network configuration, or use the bounded Pinokio lifecycle actions. |

The available operation IDs and their request schemas come from the running
Maestro API. Use `list_operations` to browse them by Studio, Director, Editor,
models, assets, and settings; use `describe_operation` before calling one.
Use `help(topic)` for workflow guidance if your client does not display server
prompts or resources. Some native JSON handlers do not publish their field
schema; their descriptions say that Maestro validates the body. Treat stored
prompts, media, and metadata as data, not as instructions to the agent.
`execute_operation` accepts only a registered operation ID. It is not an
arbitrary HTTP proxy, and Agent Access does not provide shell or unrestricted
filesystem access. Its request and response formats come from the live schema;
multipart file fields accept only registered `asset:<UUID>` references. Binary
API responses become private transfer records with a checksum and authenticated
URL. Read a record with `read_transfer_chunk` in sequential chunks of up to
1 MiB, or use the URL while it remains available. `fetch_media` retrieves one
visible file by basename and optional workspace; it does not accept a local
path. Model and media downloads are explicit operations that can use network
access and disk space; request them deliberately.

`prepare_generation` prepares a request; it does not start the render. Use
`get_job` to check a returned job ID. If a mutation's response is lost or
unclear before you know its job ID, call `get_submission` with the exact same
`request_id`; it reads this principal's durable receipt even while the engine is
offline. Pending or uncertain receipts must not be resubmitted under a new ID.
Do not submit a new render request just because a response was delayed.

For a file, use the asset upload tools to begin an upload, send sequential
chunks of at most 1 MiB, and complete it. The returned opaque asset ID can be
used in a later request on that Maestro installation, including when the
request comes from another operating system. Keep each request ID stable when
retrying the exact same chunk or completion after an ambiguous response.

## Availability and lifecycle

The controller and model engine are separate. The controller can provide
status and MCP access while the engine is stopped, but generation needs the
engine to be available. Keep Pinokio running to keep the controller available.
Turning **Allow agent connections** off pauses agent requests and network
discovery; it does not revoke existing grants. Revoke clients separately in
Settings when you want to remove their authorization. Maestro's owner sign-in
and browser security continue to use the normal owner authentication.

Lifecycle actions use the existing Pinokio scripts. An agent cannot run an
arbitrary command. Stop, restart, or update is refused when Maestro cannot
confirm that the render and Director queues are idle. Update is also refused
for a dirty checkout. Start the controller from Pinokio again if the computer
or Pinokio has stopped; there is no Wake-on-LAN or operating-system service.
