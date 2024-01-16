#!/bin/bash


# keep track of folder processed
folder_count = 0

# check for profile_summary in folder
check_profile_summary() {
    if [ -e "$1/profile_summary.csv" ]; then
			echo "Profile summary found in directory: $1"
		else
        # change directory to the folder
        cd "$1" || exit 1  
        search_xmodel
        
        cd - >/dev/null
    fi
}

search_xmodel(){
    echo "Searching for .xmodel file ..."
    for x_model in *.xmodel; do
        if [ -e "$x_model" ]; then
            echo "Found .xmodel file: $txt_file"
            python3 -m vaitrace_py ../latency_test.py *.xmodel
            ((folder_count++))
            if [ "$folder_count" -eq 10 ]; then
                echo "Command executed in 10 folders. Stopping script execution."
                exit 0
            fi
        fi
    done
}

#search all folders
traverse_folders() {
    echo "Searching for profile_summary.csv in all directories..."
    for dir in */; do
        # ignore non-directory entries
        if [ -d "$dir" ]; then
            check_profile_summary "$dir"
        fi
    done
    echo "Search complete."
}

traverse_folders

