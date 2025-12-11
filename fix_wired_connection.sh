#!/bin/bash
# Fix wired connection for Vision Pro on macOS

echo "🔧 Fixing Vision Pro Wired Connection"
echo "===================================="
echo ""

# Check if Vision Pro is connected
echo "1. Checking for USB devices..."
USB_DEVICES=$(system_profiler SPUSBDataType 2>/dev/null | grep -i "apple\|vision" | wc -l)
if [ "$USB_DEVICES" -eq 0 ]; then
    echo "   ⚠️  No Apple/Vision Pro USB devices detected"
    echo "   Please ensure Vision Pro is connected via Developer Strap"
    exit 1
fi

echo "   ✅ USB device detected"
echo ""

# Check bridge status
echo "2. Checking bridge configuration..."
if ! ifconfig bridge0 > /dev/null 2>&1; then
    echo "   ❌ Bridge not configured. Run 'setup-avp-wired' first"
    exit 1
fi

BRIDGE_IP=$(ifconfig bridge0 | grep "inet 169.254.220.1" | awk '{print $2}')
if [ -z "$BRIDGE_IP" ]; then
    echo "   ❌ Bridge IP not set correctly"
    exit 1
fi

echo "   ✅ Bridge configured: $BRIDGE_IP"
echo ""

# Check for USB network interfaces
echo "3. Looking for USB network interfaces..."
USB_INTERFACES=$(ifconfig | grep -E "^en[0-9]+" | awk '{print $1}' | tr -d ':')

for iface in $USB_INTERFACES; do
    # Skip interfaces already in bridge
    if ifconfig bridge0 | grep -q "member: $iface"; then
        continue
    fi
    
    # Check if interface has link-local IP (169.254.x.x)
    INTERFACE_IP=$(ifconfig $iface 2>/dev/null | grep "inet 169.254" | awk '{print $2}')
    if [ ! -z "$INTERFACE_IP" ]; then
        echo "   Found USB interface: $iface with IP: $INTERFACE_IP"
        echo "   Adding to bridge..."
        sudo ifconfig bridge0 addm $iface 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "   ✅ Added $iface to bridge"
        else
            echo "   ⚠️  Could not add $iface to bridge (may already be added)"
        fi
    fi
done

echo ""
echo "4. Checking ARP table for Vision Pro..."
VISION_IP=$(arp -a | grep "169.254.220" | grep -v "169.254.220.1" | awk '{print $2}' | tr -d '()' | head -1)

if [ ! -z "$VISION_IP" ]; then
    echo "   Found potential Vision Pro IP: $VISION_IP"
    echo ""
    echo "5. Testing connection..."
    ping -c 2 $VISION_IP > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✅ Connection successful!"
        echo ""
        echo "   Use this IP in your Python scripts:"
        echo "   python script.py --ip $VISION_IP"
    else
        echo "   ⚠️  Ping failed, but IP detected"
        echo "   Try:"
        echo "   1. Unplug and replug USB-C cable"
        echo "   2. Wait 10 seconds"
        echo "   3. Check Vision Pro app for wired IP"
        echo "   4. Run this script again"
    fi
else
    echo "   ⚠️  No Vision Pro IP detected in ARP table"
    echo ""
    echo "   Troubleshooting steps:"
    echo "   1. Make sure Vision Pro is connected via Developer Strap"
    echo "   2. Open Tracking Streamer app on Vision Pro"
    echo "   3. Check if wired IP is shown in the app"
    echo "   4. If IP is shown, try using that IP directly"
    echo "   5. Try unplugging and replugging the USB-C cable"
fi

echo ""
echo "6. Current bridge members:"
ifconfig bridge0 | grep "member:" | awk '{print "   - " $2}'




