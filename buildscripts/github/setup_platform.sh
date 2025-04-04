#!/bin/bash

set -ex

PLATFORM=$1

echo "Setting up platform-specific requirements for ${PLATFORM}"

case "${PLATFORM}" in
    "osx-64")
        echo "Setting up macOS SDK for osx-64 build"
        sdk_dir="buildscripts/github"
        mkdir -p "${sdk_dir}"

        # Download SDK
        echo "Downloading MacOSX10.10.sdk.tar.xz"
        wget -q https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX10.10.sdk.tar.xz

        # Verify checksum
        echo "Verifying SDK checksum"
        shasum -c "${sdk_dir}/MacOSX10.10.sdk.checksum" || exit 1

        # Extract SDK to /opt
        echo "Extracting SDK to /opt"
        sudo mkdir -p /opt
        sudo tar -xf MacOSX10.10.sdk.tar.xz -C /opt
        echo "macOS SDK setup complete"
        ;;
    *)
        echo "No specific setup required for platform: ${PLATFORM}"
        ;;
esac

echo "Platform setup complete for ${PLATFORM}" 