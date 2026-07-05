# Security Policy

vramwatch is a read-only local tool: it shells out to `nvidia-smi`/`amd-smi`
and makes HTTP requests to inference servers on loopback. It does not write to
your GPU, take any destructive action, or send data off your machine.

## Reporting a vulnerability

If you find a security issue (e.g. a way to make vramwatch execute something
unexpected via crafted `nvidia-smi`/loader output), please report it privately:

- Use GitHub's **"Report a vulnerability"** (Security → Advisories) on this repo, or
- email the maintainer listed on the GitHub profile.

Please do not open a public issue for security reports. You'll get an
acknowledgement within a few days.

## Supported versions

Only the latest released version is supported.
