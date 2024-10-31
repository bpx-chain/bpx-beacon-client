
## BPX Beacon Client

Official Python implementation of the BPX Beacon Chain: full node, introducer, farmer and timelord

BPX is a new generation Ethereum-compatible blockchain based upon an innovative consensus algorithm, Proof of Space and Time known from the Chia Network. Proof of Space is a cryptographic technique where farmers prove that they allocate unused hard disk space to the network. Proof of Time increases the overall security of the blockchain.

## Installation

### Windows
1. Download and run the installer EXE file.
2. Follow the wizard instructions.
### Debian / Ubuntu
1. Download DEB package
2. Install it using the following command:
```shell
dpkg -i bpx-beacon-client-*.deb
```
### CentOS / Red Hat / Fedora
1. Download RPM package
2. Install it using the following command:
```shell
rpm -i bpx-beacon-client-*.deb
```

## Building the source

This instruction is for MacOS, Ubuntu, CentOS, RedHat, WSL2, Amazon Linux 2 and possibly *NIX like OSes:

```shell
# This installs compatible Python modules
. install.sh
. activate

# If you want to debug/develop GUI app, try to install from source
# This installs supported version NodeJS/npm and npm dependencies.
. install-gui.sh

# To open GUI app, run the following command
. start-gui.sh
```

## Usage

> :warning: **You need both Beacon Client and [Execution Client](https://github.com/bpx-chain/bpx-execution-client) to synchronize with the network.**

### Windows
1. To launch the full node software with a GUI simply use the "BPX Beacon Client" shortcut on your desktop or Start Menu
2. To switch to testnet use the following commands:
```shell
cd "%userprofile%\AppData\Local\Programs\BPX Beacon Client\resources\app.asar.unpacked\daemon"
bpx configure --testnet true
```
3. To switch back to the mainnet use the following commands:
```shell
cd "%userprofile%\AppData\Local\Programs\BPX Beacon Client\resources\app.asar.unpacked\daemon"
bpx configure --testnet false
```

### Linux
1. To launch the full node software with a GUI use the following command:
```shell
bpx-beacon-client &
```
2. To switch to testnet use the following command:
```shell
bpx configure --testnet true
```
3. To switch back to the mainnet use the following command:
```shell
bpx configure --testnet false
```
