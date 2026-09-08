# Troubleshooting

Everything below is from the WSL2 install on 2026-09-08. Items marked *fixed in
revv* need only an update; the rest are on your side. Install steps are in
[README.md](README.md); the limits that are not bugs are in [LIMITS.md](LIMITS.md).

## Installing on WSL2

The rule that breaks most first runs: **install the NVIDIA driver on the
Windows host, before touching WSL2. Never install an NVIDIA driver inside the
WSL2 guest** — CUDA is passed through from the host driver. A working
`nvidia-smi` inside Ubuntu does not mean the CUDA toolkit is present.

```
wsl --install -d Ubuntu          # PowerShell (admin), reboot, open Ubuntu
sudo apt update && sudo apt install -y git cmake build-essential
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y cuda-toolkit-13-3    # or the newest cuda-toolkit-13-x listed
export PATH=/usr/local/cuda/bin:$PATH    # add to ~/.bashrc too
# then follow the Linux steps
```

Ubuntu's packaged CUDA and older 12.x toolkits fail against recent glibc/gcc,
which is why the NVIDIA WSL repo is used above. Expect a smaller context:
Windows reserves roughly 1–1.5 GB of the card, revv plans against free VRAM,
and will drop a rung: on our WSL2 box that is 8,192 for the dense build and
12,288 for the MoE one. `revv doctor` shows the reservation and the choice.

The CUDA toolkit lines above are only needed for a source build. The prebuilt
carries its own runtime and needs no toolkit.

- **`apt install` fails with 404s on a fresh WSL image.** The image ships a
  stale package index. Run `sudo apt update` before anything else.
- **`nvidia-smi: command not found` over SSH or in a script, though it works in
  a terminal.** WSL2 puts it in `/usr/lib/wsl/lib`, and only login shells add
  that to PATH. *Fixed in revv:* revv falls back to that path itself.

## Memory and VRAM

- **Ubuntu shows half your RAM.** WSL2 caps the guest at 50% of the host by
  default, which can hide enough to rule out the MoE build (it wants ~8–9 GB
  free, on top of the VRAM). Put `memory=24GB` under `[wsl2]` in
  `%UserProfile%\.wslconfig`, a Windows file you write from PowerShell or from
  Linux at `/mnt/c/Users/<you>/.wslconfig`, then `wsl --shutdown` from
  PowerShell. *Fixed in revv:* `revv doctor` now says this on WSL2.
- **Windows desktop apps hold 0.5–1 GB of the card.** revv plans against what is
  free at launch, so it will refuse or serve a smaller context. Close browsers
  and the like before `revv up`; opening them mid-session can OOM the server.

## The VM and the disk

- **Linux starts throwing I/O errors or "Bus error" and the VM dies.** The WSL
  virtual disk lives on `C:` and grows on demand, so when `C:` fills the guest's
  disk fails under it. Free space on `C:`, or put models on another drive with
  `REVV_HOME`. Interrupted model downloads resume.
- **Closing the last Ubuntu window kills the VM**, and with it a running server
  and any download in flight. Keep one open, or set `vmIdleTimeout=-1`.

## revv's own bugs, fixed

- **`revv up` refuses on a card that should fit, or plans the MoE build far too
  small.** Both were ours, fixed on 2026-09-08 in `3f9542d` and `4d126fc`. If
  you cloned before that date, `git pull` or `revv update`.
- **The installer says to install `cuda-cudart`.** A false warning, fixed in
  `3f9542d`; the CUDA runtime is inside the archive. Ignore it on an old clone.

## Building from source, and which binary you have

The prebuilt is the simplest path: the CUDA *runtime* libraries
`libcudart`/`libcublas`/`libcublasLt` ship inside the archive, so no compiler
and no CUDA toolkit are needed. The binary is built for sm_75/80/86/89/90
(Turing through Hopper/Ada); **RTX 50-series (sm_120) is not included** — use
`--upstream` or `--source` there.

`./install.sh --source` clones llama.cpp at the pinned commit, applies the two
patches in `patches/` (or `--stock` for none), and builds with CUDA. You can
read every patch first. It needs cmake, a CUDA toolkit, and a host compiler
`nvcc` accepts — that chain is the single biggest obstacle to a first working
install, and the reason the prebuilt exists. `./install.sh --upstream` fetches
official llama.cpp binaries instead, but on Linux those are Vulkan, not CUDA:
it runs, the kernel patch does not apply, and none of these numbers were
measured on it. `revv doctor` reports which path built the binary you are
running.

## Ports and an existing llama-server

The server binds only to `127.0.0.1`, on 8080 or the next free port.

- **You already have an `llama-server` on PATH.** revv installs its own anyway
  and says so, because its numbers were measured on the patched build. Pass
  `--system-llama-server` to use yours instead.
- `install.sh` installs its own `llama-server` copy even when one is on your
  PATH, because revv's numbers were measured on its patched build; pass
  `--system-llama-server` if you want yours used instead.

## Read more

[LIMITS.md](LIMITS.md), [BENCHMARKS.md](BENCHMARKS.md), and [TEST_WSL2_RESULTS.md](TEST_WSL2_RESULTS.md).
