#!/bin/bash
#
# Cleanup script for HDF5 file locks
#
# This script identifies and terminates zombie processes holding
# HDF5 files open, which can prevent new runs from accessing the data.
#
# Usage:
#   ./cleanup_hdf5_locks.sh [HDF5_FILE_PATH]
#
# If no path is provided, searches for all .h5 files in exploration_results/

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [[ $# -eq 1 ]]; then
    TARGET_FILE="$1"
    
    if [[ ! -f "$TARGET_FILE" ]]; then
        echo -e "${RED}Error: File not found: $TARGET_FILE${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Checking for processes holding: $TARGET_FILE${NC}"
    
    # Get PIDs holding the file
    PIDS=$(lsof "$TARGET_FILE" 2>/dev/null | awk 'NR>1 {print $2}' | sort -u)
    
    if [[ -z "$PIDS" ]]; then
        echo -e "${GREEN}✓ No processes holding the file${NC}"
        exit 0
    fi
    
    echo -e "${YELLOW}Found processes:${NC}"
    lsof "$TARGET_FILE" 2>/dev/null | head -10
    
    echo ""
    echo -e "${RED}Terminating processes: $PIDS${NC}"
    kill -9 $PIDS 2>/dev/null || true
    sleep 1
    
    # Verify
    REMAINING=$(lsof "$TARGET_FILE" 2>/dev/null | wc -l)
    if [[ $REMAINING -eq 0 ]]; then
        echo -e "${GREEN}✓ All processes terminated successfully${NC}"
    else
        echo -e "${RED}⚠ Some processes may still be active${NC}"
        lsof "$TARGET_FILE" 2>/dev/null
    fi
    
else
    # Search for all locked .h5 files in exploration_results
    echo -e "${YELLOW}Searching for locked HDF5 files in exploration_results/${NC}"
    
    FOUND_LOCKS=0
    
    for h5file in exploration_results/**/*.h5; do
        if [[ -f "$h5file" ]]; then
            PIDS=$(lsof "$h5file" 2>/dev/null | awk 'NR>1 {print $2}' | sort -u)
            
            if [[ -n "$PIDS" ]]; then
                FOUND_LOCKS=1
                echo ""
                echo -e "${YELLOW}Locked file: $h5file${NC}"
                echo "Processes: $PIDS"
                
                read -p "Kill these processes? [y/N] " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    kill -9 $PIDS 2>/dev/null || true
                    echo -e "${GREEN}✓ Processes terminated${NC}"
                fi
            fi
        fi
    done
    
    if [[ $FOUND_LOCKS -eq 0 ]]; then
        echo -e "${GREEN}✓ No locked HDF5 files found${NC}"
    fi
fi
