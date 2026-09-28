# Security Policy

## Reporting a vulnerability

Please do not open a public issue for credential leaks, authorization bypasses,
or remote-code-execution problems. Send a private GitHub Security Advisory to
the maintainers with reproduction steps and the affected version.

## Deployment notes

- Keep `.env`, Feishu credentials, model keys, SQLite runtime data, and source
  documents out of Git.
- Replace `SERVICE_API_KEY` before exposing the API to a network.
- This first release uses one tenant-wide knowledge base. It does not enforce
  per-user Feishu ACLs, so only configure folders whose contents may be shared
  with every user of this service.
- Rotate a credential immediately if it ever appears in Git history; deleting
  only the latest copy is not sufficient.
