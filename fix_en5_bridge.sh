#!/bin/bash
# Fix en5 (Vision Pro USB) interface and add to bridge

echo "🔧 Configuring Vision Pro USB interface (en5) for bridge"
echo "========================================================="
echo ""

# Check if en5 exists
if ! ifconfig en5 > /dev/null 2>&1; then
    echo "❌ en5 interface not found"
    echo "   Make sure Vision Pro is connected via USB"
    exit 1
fi

echo "1. Current en5 configuration:"
ifconfig en5 | grep -E "inet |status"
echo ""

echo "2. Configuring en5 to use 169.254.220.x subnet..."
sudo ifconfig en5 169.254.220.2 netmask 255.255.255.0

if [ $? -eq 0 ]; then
    echo "   ✅ en5 configured"
else
    echo "   ❌ Failed to configure en5"
    exit 1
fi

echo ""
echo "3. Adding en5 to bridge0..."
sudo ifconfig bridge0 addm en5

if [ $? -eq 0 ]; then
    echo "   ✅ en5 added to bridge"
else
    echo "   ⚠️  Could not add en5 (may already be added)"
fi

echo ""
echo "4. Verifying bridge configuration:"
ifconfig bridge0 | grep "member:"
echo ""

echo "5. New en5 configuration:"
ifconfig en5 | grep -E "inet |status"
echo ""

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Open Tracking Streamer app on Vision Pro"
echo "2. Check for wired IP (should be 169.254.220.XXX)"
echo "3. Test connection: ping 169.254.220.XXX"
echo "4. Use that IP in your Python scripts"




