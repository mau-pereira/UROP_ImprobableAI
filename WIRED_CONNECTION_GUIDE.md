# Wired Connection Guide: Vision Pro via Developer Strap

## Overview

The wired connection uses the **Vision Pro Developer Strap** to connect your Vision Pro directly to your computer via USB-C. This provides:

- **~10 Gbps local network speeds** (much faster than WiFi)
- **Lower latency** (~50ms vs ~100ms on WiFi)
- **More stable connection** (no WiFi interference)
- **Full internet access** for Vision Pro (via NAT)

## Prerequisites

1. **Vision Pro Developer Strap** - [Purchase from Apple](https://developer.apple.com/visionos/developer-strap/)
2. **USB-C cable** - Connect the developer strap to your computer
3. **macOS or Linux** - Windows is not yet supported
4. **Admin/sudo access** - Required for network configuration

## Step-by-Step Setup

### Step 1: Physical Connection

1. **Attach the Developer Strap** to your Vision Pro
2. **Connect USB-C cable** from the developer strap to your computer
3. **Wait 5-10 seconds** for the computer to recognize the device

### Step 2: Run the Setup Script

The `avp_stream` package includes a command-line tool to set up the network bridge:

```bash
setup-avp-wired
```

**Alternative**: If the command isn't found, you can run it directly:

```bash
# From the repository root
python -m avp_stream.bridge_avp

# Or use the shell script (macOS only)
./avp_stream/bridge_avp.sh
```

### Step 3: Follow the Prompts

The script will:

1. **Check prerequisites** - Verifies required tools are installed
2. **Detect network configuration** - Finds your primary network interface
3. **Show configuration** - Displays detected settings
4. **Set up bridge** - Configures network bridge and NAT

**Example output:**
```
🔧 Vision Pro High-Speed Bridge Setup
======================================

⚠️  IMPORTANT: Plug in your Vision Pro BEFORE running this script!

Is your Vision Pro plugged in? (y/n) y

Detecting configuration...

Configuration:
  Platform: Darwin
  Primary interface: en0
  Your machine's IP: 192.168.1.100
  Network: 169.254.220.0/24
  Bridge IP: 169.254.220.1

Continue? (y/n) y

Setting up bridge...
  • Enabling IP forwarding...
  • Configuring bridge0...
  • Adding en0 to bridge0...
  • Configuring NAT...

✅ Setup complete!

⏳ Network should be active in ~5 seconds...

Your Vision Pro should now have:
  • ~10 Gbps local network speeds (test with iperf)
  • Full internet access
```

### Step 4: Get Vision Pro IP Address

1. **Open Tracking Streamer app** on Vision Pro
2. **Look for the wired IP address** - The app shows separate IPs for WiFi and wired connections
3. **Note the wired IP** - It should be in the `169.254.220.x` range (e.g., `169.254.220.107`)

**Important**: Use the **wired IP address**, not the WiFi IP!

### Step 5: Use the Wired IP in Python

```python
from avp_stream import VisionProStreamer

# Use the WIRED IP address shown in the app
wired_ip = "169.254.220.107"  # Replace with your actual wired IP
streamer = VisionProStreamer(ip=wired_ip)

# Rest of your code...
```

Or from command line:

```bash
python your_script.py --ip 169.254.220.107
```

## Platform-Specific Notes

### macOS

- Uses `bridge0` interface
- Uses `pfctl` for NAT
- Automatically detects primary network interface

**Requirements:**
- `ifconfig`, `route`, `sysctl`, `pfctl` (all built-in)

### Linux

- Uses `br0` bridge interface (wired) or direct USB interface (wireless)
- Uses `iptables` for NAT
- Requires `bridge-utils` package for wired connections

**Install bridge-utils (if needed):**
```bash
sudo apt-get install bridge-utils  # Debian/Ubuntu
sudo yum install bridge-utils       # RHEL/CentOS
```

**Requirements:**
- `ip`, `brctl`, `sysctl`, `iptables`

## Verification

### Test Connection

```bash
# Ping Vision Pro (use the wired IP from the app)
ping 169.254.220.107

# Should see responses like:
# 64 bytes from 169.254.220.107: icmp_seq=0 ttl=64 time=0.123 ms
```

### Test Internet on Vision Pro

1. Open Safari or any browser on Vision Pro
2. Try loading a webpage (e.g., `google.com`)
3. Should work if NAT is configured correctly

### Test Python Connection

```bash
# Run simplest example
python examples/00_hand_streaming.py --ip 169.254.220.107
```

## Troubleshooting

### Issue: "USB network interface not detected"

**Solution:**
1. Unplug and replug the USB-C cable
2. Wait 5-10 seconds
3. Run `ip link show` (Linux) or `ifconfig` (macOS) to see all interfaces
4. Try running the setup script again

### Issue: "Command not found: setup-avp-wired"

**Solution:**
```bash
# Reinstall the package
pip install --upgrade avp_stream

# Or run directly
python -m avp_stream.bridge_avp
```

### Issue: Vision Pro can't access internet

**Solution:**
1. Check that NAT is configured: `sudo pfctl -s nat` (macOS) or `sudo iptables -t nat -L` (Linux)
2. Verify IP forwarding is enabled: `sysctl net.inet.ip.forwarding` (should be 1)
3. Try restarting the setup script

### Issue: Connection works but very slow

**Solution:**
1. Make sure you're using the **wired IP**, not WiFi IP
2. Check USB-C cable quality (should support USB 3.0+)
3. Verify bridge is active: `ifconfig bridge0` (macOS) or `ip link show br0` (Linux)

### Issue: Can't ping Vision Pro

**Solution:**
1. Verify Vision Pro is physically connected
2. Check that bridge is configured: `ifconfig bridge0` (macOS)
3. Make sure you're using the correct wired IP from the app
4. Try unplugging and replugging the cable

## Cleanup

When you're done, you can clean up the network configuration:

```bash
# The setup script creates a cleanup script at:
/tmp/vision-pro-cleanup.sh

# Run it to restore network settings
bash /tmp/vision-pro-cleanup.sh
```

Or manually:

**macOS:**
```bash
sudo ifconfig bridge0 deletem en0
sudo ifconfig bridge0 delete 169.254.220.1
sudo sysctl -w net.inet.ip.forwarding=0
sudo pfctl -F all
sudo pfctl -d
```

**Linux:**
```bash
sudo iptables -t nat -D POSTROUTING -s 169.254.220.0/24 -o en0 -j MASQUERADE
sudo brctl delif br0 en0
sudo ip link set br0 down
sudo brctl delbr br0
sudo sysctl -w net.ipv4.ip_forward=0
```

## Performance Comparison

According to benchmarks:

| Connection Type | Latency (720p) | Latency (4K Stereo) | Speed |
|----------------|---------------|---------------------|-------|
| **WiFi**       | ~100ms        | ~150ms              | ~1 Gbps |
| **Wired**      | ~50ms         | ~50ms               | ~10 Gbps |

Wired connection provides:
- **2x lower latency**
- **10x faster speeds**
- **More stable** (no WiFi interference)

## Important Notes

1. **Always use the wired IP** shown in the Tracking Streamer app, not the WiFi IP
2. **Run setup script AFTER** plugging in Vision Pro
3. **Keep the USB-C cable connected** while using wired mode
4. **Both WiFi and wired can work simultaneously** - just use the correct IP for each
5. **NAT provides internet access** - Vision Pro can access internet through your computer's connection

## Example: Full Workflow

```bash
# 1. Plug in Vision Pro via Developer Strap
# 2. Wait 5 seconds

# 3. Run setup
setup-avp-wired

# 4. Open Tracking Streamer app on Vision Pro
# 5. Note the wired IP (e.g., 169.254.220.107)

# 6. Run your Python script with wired IP
python examples/10_teleop_osc_franka.py --ip 169.254.220.107

# 7. When done, cleanup (optional)
bash /tmp/vision-pro-cleanup.sh
```

## Getting Help

If you encounter issues:

1. Check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide
2. Run with verbose output: `python script.py --ip IP --verbose`
3. Check system logs for network errors
4. Verify all prerequisites are installed

